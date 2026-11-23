"""Multi-Seed Publication-Grade Benchmark, Statistical Significance & Ablation Runner.

Executes 6 rigorous experiments across multiple independent random seeds, computing
mean, std, 95% Student-t confidence intervals, and paired Wilcoxon signed-rank p-values:
  Exp 1: Multi-Domain Head-to-Head (8 ODEs incl. Duffing/VdP + 2 PDEs @ 5% Noise, 5 seeds)
         comparing ASDS-OMEGA v2.0 vs PySINDy v2.1 (Official), Weak-SINDy, Standard SINDy,
         Pure GP, and Black-Box Neural MLP (3-sigma OOD extrapolation).
  Exp 2: High-Noise EIV Stress Test & Hermite Cumulant Debiasing Isolation (0%..30%, 5 seeds)
         on Van der Pol, Duffing, and Lorenz-63.
  Exp 3: 6-Factor Systematic Ablation Study (@ 10% Noise, 5 seeds) with paired p-values.
  Exp 4: Popperian Active Falsification (PAFE) vs Random Query Efficiency.
  Exp 5: Online Structural Drift Detection & Surgical Regeneration (SIR -> SIRS).
  Exp 6: Real-World Experimental Records (Hudson Bay Lynx-Hare, 1978 Flu, Mauna Loa CO2).
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from scipy import stats

from asds_omega.benchmarks.baselines import (
    run_neural_mlp_baseline,
    run_pure_gp_baseline,
    run_pysindy_official_baseline,
    run_standard_sindy_baseline,
    run_weak_sindy_baseline,
)
from asds_omega.config import EVGHEConfig, MSAVPConfig, PipelineConfig, VIFCConfig, WLINDConfig
from asds_omega.data import (
    gen_sir_regime_drift,
    generate_synthetic_dataset,
    inject_noise,
    load_real_dataset,
)
from asds_omega.scientist import ASDSOmegaPipeline, SurgicalDriftRegenerator
from asds_omega.symbolic import discover_pde_vesp
from asds_omega.verification import (
    export_discovery_json,
    export_latex_report,
    export_lean4_certificate,
    export_markdown_report,
)


def _stats_summary(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n > 1 and std > 0.0:
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        ci95 = float(t_crit * std / np.sqrt(n))
    else:
        ci95 = 0.0
    return {"mean": round(mean, 4), "std": round(std, 4), "ci95": round(ci95, 4), "n": n}


def _paired_pvalue(a: list[float], b: list[float]) -> float:
    """One-sided paired Wilcoxon signed-rank test (H1: median(a) > median(b))."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(diff) < 3 or np.allclose(diff, 0.0, atol=1e-9):
        return 1.0
    try:
        res = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
        return float(res.pvalue)
    except Exception:
        t_res = stats.ttest_rel(a, b, alternative="greater")
        return float(t_res.pvalue) if np.isfinite(t_res.pvalue) else 1.0


def _compute_symbolic_ood_rmse(equations: list[sp.Expr], dataset, seed: int = 42) -> float | None:
    if dataset.rhs_fn is None:
        return None
    rng = np.random.default_rng(seed)
    X_all = np.vstack(dataset.trajectories)
    mu = np.mean(X_all, axis=0)
    sd = np.maximum(np.std(X_all, axis=0), 0.5)
    X_ood = mu + rng.uniform(-3.0, 3.0, size=(120, dataset.n_vars)) * sd
    if dataset.name in {"sir", "lotka_volterra", "michaelis_menten"}:
        X_ood = np.abs(X_ood) + 0.2
    Y_true = np.array([dataset.rhs_fn(0.0, pt) for pt in X_ood], dtype=float)
    syms = [sp.Symbol(v) for v in dataset.var_names]
    funcs = [sp.lambdify(syms, eq, modules="numpy") for eq in equations]
    Y_pred = np.zeros_like(Y_true)
    for k, pt in enumerate(X_ood):
        for i, fn in enumerate(funcs):
            try:
                val = float(fn(*pt))
                Y_pred[k, i] = val if np.isfinite(val) else 0.0
            except Exception:
                Y_pred[k, i] = 0.0
    return float(np.sqrt(np.mean((Y_true - Y_pred) ** 2)) / (np.std(Y_true) + 1e-8))


def run_full_benchmark_suite(
    output_dir: str | Path = "results",
    seed: int = 42,
    n_seeds: int = 5,
) -> dict[str, Any]:
    """Execute all 6 multi-seed publication experiments and persist verified ledger + artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    seeds = [seed + 100 * k for k in range(n_seeds)]
    ledger: dict[str, Any] = {
        "framework": "ASDS-OMEGA v2.0 (VECSD)",
        "seeds": seeds,
        "n_seeds": n_seeds,
        "experiments": {},
    }

    # ------------------------------------------------------------------
    # Exp 1: Multi-Domain Head-to-Head Benchmark @ 5% Noise (Multi-Seed)
    # ------------------------------------------------------------------
    print(f"[Exp 1] Multi-Domain Head-to-Head Benchmark (8 ODEs + 2 PDEs @ 5% noise, {n_seeds} seeds)...", flush=True)
    ode_names = [
        "lorenz", "lotka_volterra", "sir", "pendulum",
        "kepler", "michaelis_menten", "duffing", "van_der_pol",
    ]
    exp1_odes: dict[str, Any] = {}
    paired_asds_all: list[float] = []
    paired_baselines_all: dict[str, list[float]] = {
        "pysindy_official": [],
        "weak_sindy": [],
        "standard_sindy": [],
        "pure_gp": [],
    }

    for name in ode_names:
        asds_sm, asds_ex, asds_ce, asds_shd, asds_ood, asds_rt = [], [], [], [], [], []
        base_sm: dict[str, list[float]] = {k: [] for k in paired_baselines_all}
        base_ce: dict[str, list[float]] = {k: [] for k in paired_baselines_all}
        mlp_oods: list[float] = []
        rep_summary: dict[str, Any] | None = None
        rep_baselines: dict[str, Any] | None = None

        for s_idx, s in enumerate(seeds):
            ds = inject_noise(generate_synthetic_dataset(name, seed=s), 0.05, seed=s)
            cfg = PipelineConfig(seed=s, evghe=EVGHEConfig(gp_population_size=24, gp_generations=10))
            cfg.scientist.enable_active_falsification = False
            res = ASDSOmegaPipeline(cfg).run(ds, run_active_falsification=False)
            ood_asds = _compute_symbolic_ood_rmse([c.expr for c in res.equations], ds, seed=s)

            asds_sm.append(float(res.evidence.symbolic_match_score) if res.evidence.symbolic_match_score is not None else 0.0)
            asds_ex.append(float(res.evidence.exact_structure_rate) if res.evidence.exact_structure_rate is not None else 0.0)
            asds_ce.append(float(res.evidence.mean_coeff_rel_error) if res.evidence.mean_coeff_rel_error is not None else 1.0)
            asds_shd.append(float(res.causal.metrics.get("shd", 0.0)))
            asds_ood.append(float(ood_asds if ood_asds is not None else 0.0))
            asds_rt.append(float(res.runtime_sec))

            b_ps = run_pysindy_official_baseline(ds, seed=s)
            b_ws = run_weak_sindy_baseline(ds, seed=s)
            b_ss = run_standard_sindy_baseline(ds, seed=s)
            b_gp = run_pure_gp_baseline(ds, seed=s)
            b_mlp = run_neural_mlp_baseline(ds, seed=s)

            for key, b_obj in [
                ("pysindy_official", b_ps),
                ("weak_sindy", b_ws),
                ("standard_sindy", b_ss),
                ("pure_gp", b_gp),
            ]:
                base_sm[key].append(float(b_obj.symbolic_match))
                base_ce[key].append(float(b_obj.mean_coeff_rel_error))
                paired_baselines_all[key].append(float(b_obj.symbolic_match))
            paired_asds_all.append(float(res.evidence.symbolic_match_score) if res.evidence.symbolic_match_score is not None else 0.0)
            mlp_oods.append(float(b_mlp.ood_extrapolation_rmse or 1.0))

            if s_idx == 0:
                rep_summary = res.to_summary_dict()
                rep_summary["ood_extrapolation_rmse"] = round(asds_ood[0], 6)
                rep_baselines = {
                    "pysindy_official": asdict(b_ps),
                    "weak_sindy": asdict(b_ws),
                    "standard_sindy": asdict(b_ss),
                    "pure_gp": asdict(b_gp),
                    "neural_mlp": asdict(b_mlp),
                }
                if res.lean4_certificate:
                    export_lean4_certificate(
                        name, ds.var_names, [c.expr for c in res.equations], res.symmetry, res.verification,
                        output_path=out / "lean4" / f"{name}_certificate.lean",
                    )
                export_markdown_report(rep_summary, out / "reports" / f"{name}_report.md")
                export_latex_report(rep_summary, out / "reports" / f"{name}_report.tex")

        exp1_odes[name] = {
            "asds_omega_v2": rep_summary,
            "baselines": rep_baselines,
            "multiseed": {
                "asds_omega_v2": {
                    "symbolic_match": _stats_summary(asds_sm),
                    "exact_structure": _stats_summary(asds_ex),
                    "coeff_rel_error": _stats_summary(asds_ce),
                    "shd": _stats_summary(asds_shd),
                    "ood_rmse": _stats_summary(asds_ood),
                    "runtime_sec": _stats_summary(asds_rt),
                },
                "baselines": {
                    k: {"symbolic_match": _stats_summary(base_sm[k]), "coeff_rel_error": _stats_summary(base_ce[k])}
                    for k in base_sm
                },
                "neural_mlp_ood_rmse": _stats_summary(mlp_oods),
            },
        }
        print(
            f"  {name:18s}: ASDS={np.mean(asds_sm):.3f}±{np.std(asds_sm):.3f} | "
            f"PySINDy={np.mean(base_sm['pysindy_official']):.3f} | "
            f"WeakSINDy={np.mean(base_sm['weak_sindy']):.3f} | "
            f"SINDy={np.mean(base_sm['standard_sindy']):.3f}",
            flush=True,
        )

    # Aggregate statistical significance (paired Wilcoxon across all 8*n_seeds trials!)
    exp1_stats = {
        "asds_overall_symbolic_match": _stats_summary(paired_asds_all),
        "vs_baselines": {
            k: {
                "baseline_symbolic_match": _stats_summary(v),
                "wilcoxon_p_value": round(_paired_pvalue(paired_asds_all, v), 6),
            }
            for k, v in paired_baselines_all.items()
        },
    }

    # 2 Spatiotemporal PDEs @ 5% noise across seeds
    exp1_pdes: dict[str, Any] = {}
    for pde_name in ["burgers_pde", "advection_diffusion_pde"]:
        r2s_deb, r2s_nd = [], []
        rep_p = None
        for s_idx, s in enumerate(seeds):
            pds = inject_noise(generate_synthetic_dataset(pde_name, seed=s), 0.05, seed=s)
            pc = discover_pde_vesp(pds.x, pds.t, pds.u, cumulant_debiasing=True, seed=s)
            pc_nd = discover_pde_vesp(pds.x, pds.t, pds.u, cumulant_debiasing=False, seed=s)
            r2s_deb.append(pc.weak_r2)
            r2s_nd.append(pc_nd.weak_r2)
            if s_idx == 0:
                rep_p = (pds.ground_truth_equation, pc, pc_nd)
        assert rep_p is not None
        exp1_pdes[pde_name] = {
            "ground_truth": rep_p[0],
            "asds_weak_pde": {
                "expr_str": rep_p[1].expr_str,
                "weak_r2": _stats_summary(r2s_deb),
                "coefficients": rep_p[1].coefficients,
            },
            "no_debias_pde": {
                "expr_str": rep_p[2].expr_str,
                "weak_r2": _stats_summary(r2s_nd),
                "coefficients": rep_p[2].coefficients,
            },
        }
    ledger["experiments"]["exp1_multidomain_5pct"] = {
        "odes": exp1_odes,
        "pdes": exp1_pdes,
        "aggregate_significance": exp1_stats,
    }

    # ------------------------------------------------------------------
    # Exp 2: High-Noise EIV Stress Test & Hermite Cumulant Debiasing (0%..30%, multi-seed)
    # ------------------------------------------------------------------
    print(f"[Exp 2] High-Noise EIV Stress Test & Cumulant Debiasing (Van der Pol, Duffing, Lorenz, {n_seeds} seeds)...", flush=True)
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    exp2: dict[str, Any] = {"noise_levels": noise_levels, "systems": {}}
    for sys_name in ["van_der_pol", "duffing", "lorenz"]:
        curves: dict[str, list[dict[str, Any]]] = {
            "asds_omega_v2": [],
            "asds_no_cumulant_debias": [],
            "pysindy_official": [],
            "weak_sindy": [],
            "standard_sindy": [],
        }
        for sigma in noise_levels:
            bucket_sm: dict[str, list[float]] = {k: [] for k in curves}
            bucket_ce: dict[str, list[float]] = {k: [] for k in curves}
            for s in seeds:
                ds_n = inject_noise(generate_synthetic_dataset(sys_name, seed=s), sigma, seed=s)
                # Fast deterministic V-ESP + VIFC + W-LIND (GP off in pure noise stress sweep for speed)
                cfg_f = PipelineConfig(seed=s, evghe=EVGHEConfig(enable_gp_refinement=False))
                r_f = ASDSOmegaPipeline(cfg_f).run(ds_n, run_active_falsification=False)
                bucket_sm["asds_omega_v2"].append(float(r_f.evidence.symbolic_match_score) if r_f.evidence.symbolic_match_score is not None else 0.0)
                bucket_ce["asds_omega_v2"].append(float(r_f.evidence.mean_coeff_rel_error) if r_f.evidence.mean_coeff_rel_error is not None else 1.0)

                cfg_nd = PipelineConfig(
                    seed=s,
                    ms_avp=MSAVPConfig(cumulant_debiasing=False),
                    evghe=EVGHEConfig(enable_gp_refinement=False),
                )
                r_nd = ASDSOmegaPipeline(cfg_nd).run(ds_n, run_active_falsification=False)
                bucket_sm["asds_no_cumulant_debias"].append(float(r_nd.evidence.symbolic_match_score) if r_nd.evidence.symbolic_match_score is not None else 0.0)
                bucket_ce["asds_no_cumulant_debias"].append(float(r_nd.evidence.mean_coeff_rel_error) if r_nd.evidence.mean_coeff_rel_error is not None else 1.0)

                b_ps = run_pysindy_official_baseline(ds_n, seed=s)
                b_ws = run_weak_sindy_baseline(ds_n, seed=s)
                b_ss = run_standard_sindy_baseline(ds_n, seed=s)
                for mk, bobj in [("pysindy_official", b_ps), ("weak_sindy", b_ws), ("standard_sindy", b_ss)]:
                    bucket_sm[mk].append(float(bobj.symbolic_match))
                    bucket_ce[mk].append(float(bobj.mean_coeff_rel_error))

            for mk in curves:
                curves[mk].append({
                    "sigma": sigma,
                    "symbolic_match": _stats_summary(bucket_sm[mk]),
                    "coeff_rel_error": _stats_summary(bucket_ce[mk]),
                })
        exp2["systems"][sys_name] = curves
    ledger["experiments"]["exp2_noise_stress"] = exp2

    # ------------------------------------------------------------------
    # Exp 3: 6-Factor Systematic Ablation Study @ 10% Noise (Multi-Seed)
    # ------------------------------------------------------------------
    print(f"[Exp 3] 6-Factor Systematic Ablation Study (@ 10% noise, {n_seeds} seeds)...", flush=True)
    ablation_systems = ["lorenz", "sir", "pendulum", "van_der_pol", "duffing"]
    ablation_builders = {
        "full_asds_omega_v2": lambda s: PipelineConfig(seed=s, evghe=EVGHEConfig(enable_gp_refinement=False)),
        "no_cumulant_debias": lambda s: PipelineConfig(seed=s, ms_avp=MSAVPConfig(cumulant_debiasing=False), evghe=EVGHEConfig(enable_gp_refinement=False)),
        "no_weak_form_savgol": lambda s: PipelineConfig(seed=s, ms_avp=MSAVPConfig(use_weak_form=False), evghe=EVGHEConfig(enable_gp_refinement=False)),
        "no_vifc_causal": lambda s: PipelineConfig(seed=s, vifc=VIFCConfig(enabled=False), evghe=EVGHEConfig(enable_gp_refinement=False)),
        "no_wlind_symmetry": lambda s: PipelineConfig(seed=s, wlind=WLINDConfig(enabled=False), evghe=EVGHEConfig(enable_gp_refinement=False)),
        "no_mdl_rationalizer": lambda s: PipelineConfig(seed=s, evghe=EVGHEConfig(enable_rationalizer=False, enable_gp_refinement=False)),
    }
    exp3: dict[str, Any] = {}
    full_trials_sm: list[float] = []
    full_trials_ce: list[float] = []

    for abl_name, cfg_fn in ablation_builders.items():
        sm_all, ex_all, ce_all, r2_all, rt_all = [], [], [], [], []
        for s in seeds:
            for sname in ablation_systems:
                ds_a = inject_noise(generate_synthetic_dataset(sname, seed=s), 0.10, seed=s)
                ra = ASDSOmegaPipeline(cfg_fn(s)).run(ds_a, run_active_falsification=False)
                sm_all.append(float(ra.evidence.symbolic_match_score) if ra.evidence.symbolic_match_score is not None else 0.0)
                ex_all.append(float(ra.evidence.exact_structure_rate) if ra.evidence.exact_structure_rate is not None else 0.0)
                ce_all.append(float(ra.evidence.mean_coeff_rel_error) if ra.evidence.mean_coeff_rel_error is not None else 1.0)
                r2_all.append(float(ra.evidence.mean_weak_r2))
                rt_all.append(float(ra.runtime_sec))
        if abl_name == "full_asds_omega_v2":
            full_trials_sm = sm_all
            full_trials_ce = ce_all
        exp3[abl_name] = {
            "symbolic_match": _stats_summary(sm_all),
            "exact_structure": _stats_summary(ex_all),
            "coeff_rel_error": _stats_summary(ce_all),
            "weak_r2": _stats_summary(r2_all),
            "runtime_sec": _stats_summary(rt_all),
            "wilcoxon_p_vs_full_sm": round(_paired_pvalue(full_trials_sm, sm_all), 6) if abl_name != "full_asds_omega_v2" else 1.0,
            "wilcoxon_p_vs_full_ce": round(_paired_pvalue(ce_all, full_trials_ce), 6) if abl_name != "full_asds_omega_v2" else 1.0,
        }
        print(
            f"  {abl_name:22s}: SM={exp3[abl_name]['symbolic_match']['mean']:.4f}±{exp3[abl_name]['symbolic_match']['ci95']:.4f} | "
            f"Exact={exp3[abl_name]['exact_structure']['mean']:.4f} | "
            f"CoeffErr={exp3[abl_name]['coeff_rel_error']['mean']:.4f}",
            flush=True,
        )
    ledger["experiments"]["exp3_ablation_10pct"] = exp3

    # ------------------------------------------------------------------
    # Exp 4: Popperian Active Falsification (PAFE) vs Random Sampling
    # ------------------------------------------------------------------
    print("[Exp 4] Popperian Active Falsification (PAFE) vs Random Query Campaign...", flush=True)
    exp4: dict[str, Any] = {}
    for sname in ["lorenz", "sir", "lotka_volterra"]:
        ds_pafe = inject_noise(generate_synthetic_dataset(sname, n_trajectories=1, seed=seed), 0.10, seed=seed)
        cfg_pafe = PipelineConfig(seed=seed)
        cfg_pafe.scientist.enable_active_falsification = True
        cfg_pafe.scientist.active_budget = 3
        rp = ASDSOmegaPipeline(cfg_pafe).run(ds_pafe, run_active_falsification=True)
        exp4[sname] = rp.to_summary_dict()["active_falsification"]
    ledger["experiments"]["exp4_active_falsification"] = exp4

    # ------------------------------------------------------------------
    # Exp 5: Online Concept Drift & Surgical Structural Regeneration (SLR)
    # ------------------------------------------------------------------
    print("[Exp 5] Online Structural Drift & Surgical Regeneration (SIR -> SIRS)...", flush=True)
    dds = inject_noise(gen_sir_regime_drift(), 0.02, seed=seed)
    dres = SurgicalDriftRegenerator(seed=seed).run_stream(
        dds.t_list[0], dds.trajectories[0], dds.var_names, true_shift_time=20.0
    )
    ledger["experiments"]["exp5_drift_regeneration"] = asdict(dres)

    # ------------------------------------------------------------------
    # Exp 6: Real-World Experimental Scientific Datasets
    # ------------------------------------------------------------------
    print("[Exp 6] Real-World Experimental Validation (Lynx-Hare, 1978 Influenza, Mauna Loa CO2)...", flush=True)
    exp6: dict[str, Any] = {}
    pipe_real = ASDSOmegaPipeline(PipelineConfig(seed=seed))
    for rname in ["hudson_bay_lynx_hare", "boarding_school_influenza", "mauna_loa_co2"]:
        rds = load_real_dataset(rname)
        rres = pipe_real.run(rds, run_active_falsification=False)
        rsumm = rres.to_summary_dict()
        exp6[rname] = rsumm
        export_markdown_report(rsumm, out / "reports" / f"real_{rname}_report.md")
    ledger["experiments"]["exp6_real_world"] = exp6

    ledger["total_benchmark_runtime_sec"] = round(time.perf_counter() - t_start, 2)
    export_discovery_json(ledger, out / "benchmark_ledger.json")
    print(f"[DONE] Multi-seed verified benchmark ledger saved in {ledger['total_benchmark_runtime_sec']}s", flush=True)
    return ledger


if __name__ == "__main__":
    run_full_benchmark_suite()
