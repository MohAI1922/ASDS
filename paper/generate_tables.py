"""Auto-generate publication-ready LaTeX tables directly from verified benchmark_ledger.json.

Guarantees 100% numeric consistency between experimental execution and the LaTeX manuscript
(zero manual transcription!).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fmt_mean_ci(stat: dict[str, Any], decimals: int = 3) -> str:
    m = stat["mean"]
    c = stat["ci95"]
    if c < 10 ** (-decimals):
        return f"{m:.{decimals}f}"
    return f"{m:.{decimals}f} $\\pm$ {c:.{decimals}f}"


def _fmt_pval(p: float) -> str:
    if p < 1e-5:
        return r"$< 10^{-5}$"
    return f"{p:.2e}"


def generate_latex_tables(
    ledger_path: str | Path = "results/benchmark_ledger.json",
    out_dir: str | Path = "paper/tables",
) -> list[Path]:
    ledger = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # ------------------------------------------------------------------
    # Table 1: Multi-Domain Head-to-Head Symbolic Recovery (8 ODEs, 5 Seeds @ 5% Noise)
    # ------------------------------------------------------------------
    exp1 = ledger["experiments"]["exp1_multidomain_5pct"]
    odes = exp1["odes"]
    agg = exp1["aggregate_significance"]

    t1 = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Multi-domain head-to-head symbolic recovery at $5\%$ Gaussian measurement noise averaged over $n=" + str(ledger["n_seeds"]) + r"$ independent seeds (mean $\pm$ 95\% CI). Best symbolic match in \textbf{bold}.}",
        r"\label{tab:main_benchmark}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{System} & \textbf{ASDS-OMEGA v2.0} & \textbf{PySINDy v2.1} & \textbf{Weak-SINDy} & \textbf{Std. SINDy} & \textbf{Pure GP} & \textbf{OOD RMSE (ASDS / MLP)} \\",
        r"\midrule",
    ]
    pretty = {
        "lorenz": "Lorenz-63",
        "lotka_volterra": "Lotka--Volterra",
        "sir": "SIR Epidemic",
        "pendulum": "Nonlinear Pendulum",
        "kepler": "Kepler 2-Body",
        "michaelis_menten": "Michaelis--Menten",
        "duffing": "Duffing Oscillator",
        "van_der_pol": "Van der Pol",
    }
    for k, v in odes.items():
        ms = v["multiseed"]
        vals = [
            ms["asds_omega_v2"]["symbolic_match"]["mean"],
            ms["baselines"]["pysindy_official"]["symbolic_match"]["mean"],
            ms["baselines"]["weak_sindy"]["symbolic_match"]["mean"],
            ms["baselines"]["standard_sindy"]["symbolic_match"]["mean"],
            ms["baselines"]["pure_gp"]["symbolic_match"]["mean"],
        ]
        best_v = max(vals)
        raw_strs = [
            _fmt_mean_ci(ms["asds_omega_v2"]["symbolic_match"]),
            _fmt_mean_ci(ms["baselines"]["pysindy_official"]["symbolic_match"]),
            _fmt_mean_ci(ms["baselines"]["weak_sindy"]["symbolic_match"]),
            _fmt_mean_ci(ms["baselines"]["standard_sindy"]["symbolic_match"]),
            _fmt_mean_ci(ms["baselines"]["pure_gp"]["symbolic_match"]),
        ]
        fmt_strs = [f"\\textbf{{{s}}}" if val >= best_v - 1e-4 else s for s, val in zip(raw_strs, vals)]
        ood_a = ms["asds_omega_v2"]["ood_rmse"]["mean"]
        ood_m = ms["neural_mlp_ood_rmse"]["mean"]
        t1.append(
            f"{pretty.get(k, k)} & {' & '.join(fmt_strs)} & {ood_a:.4f} / {ood_m:.3f} \\\\"
        )

    vs = agg["vs_baselines"]
    t1.extend([
        r"\midrule",
        f"\\textbf{{Overall Mean ($N={agg['asds_overall_symbolic_match']['n']}$)}} & "
        f"\\textbf{{{_fmt_mean_ci(agg['asds_overall_symbolic_match'])}}} & "
        f"{_fmt_mean_ci(vs['pysindy_official']['baseline_symbolic_match'])} & "
        f"{_fmt_mean_ci(vs['weak_sindy']['baseline_symbolic_match'])} & "
        f"{_fmt_mean_ci(vs['standard_sindy']['baseline_symbolic_match'])} & "
        f"{_fmt_mean_ci(vs['pure_gp']['baseline_symbolic_match'])} & --- \\\\",
        f"\\textit{{Wilcoxon $p$-value (vs. ASDS)}} & --- & "
        f"{_fmt_pval(vs['pysindy_official']['wilcoxon_p_value'])} & "
        f"{_fmt_pval(vs['weak_sindy']['wilcoxon_p_value'])} & "
        f"{_fmt_pval(vs['standard_sindy']['wilcoxon_p_value'])} & "
        f"{_fmt_pval(vs['pure_gp']['wilcoxon_p_value'])} & --- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ])
    p1 = out / "table1_main_benchmark.tex"
    p1.write_text("\n".join(t1) + "\n", encoding="utf-8")
    written.append(p1)

    # ------------------------------------------------------------------
    # Table 2: 6-Factor Systematic Ablation Study @ 10% Noise
    # ------------------------------------------------------------------
    exp3 = ledger["experiments"]["exp3_ablation_10pct"]
    abl_labels = [
        ("full_asds_omega_v2", r"\textbf{Full ASDS-OMEGA v2.0 (VECSD)}"),
        ("no_cumulant_debias", "w/o Hermite Cumulant Debiasing"),
        ("no_weak_form_savgol", "w/o MS-AVP Weak-Form (SavGol Diff.)"),
        ("no_vifc_causal", "w/o VIFC Causal Masking"),
        ("no_wlind_symmetry", "w/o W-LIND Symmetry / KKT"),
        ("no_mdl_rationalizer", "w/o $K_{\\text{eff}}$-MDL Rationalizer"),
    ]
    t2 = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{6-factor systematic ablation at $10\%$ Gaussian noise across 5 benchmarks $\times$ " + str(ledger["n_seeds"]) + r" seeds (mean $\pm$ 95\% CI).}",
        r"\label{tab:ablation}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{Sym. Match} & \textbf{Exact Struct.} & \textbf{Coeff. Rel. Err.} & \textbf{Weak $R^2$} \\",
        r"\midrule",
    ]
    for k, lbl in abl_labels:
        row = exp3[k]
        t2.append(
            f"{lbl} & {_fmt_mean_ci(row['symbolic_match'])} & {_fmt_mean_ci(row['exact_structure'])} & "
            f"{_fmt_mean_ci(row['coeff_rel_error'])} & {_fmt_mean_ci(row['weak_r2'])} \\\\"
        )
    t2.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    p2 = out / "table2_ablation.tex"
    p2.write_text("\n".join(t2) + "\n", encoding="utf-8")
    written.append(p2)

    # ------------------------------------------------------------------
    # Table 3: Hermite Cumulant EIV Debiasing on Van der Pol & Duffing (15%-20% Noise)
    # ------------------------------------------------------------------
    exp2 = ledger["experiments"]["exp2_noise_stress"]["systems"]
    t3 = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Direct isolation of shrinkage-regularized Hermite Cumulant EIV Debiasing on cubic self-power nonlinearities (Van der Pol $-x^2 y$ and Duffing $-x^3$) at $15\%\text{--}30\%$ noise (mean $\pm$ 95\% CI, $n=" + str(ledger["n_seeds"]) + r"$ seeds).}",
        r"\label{tab:cumulant_eiv}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{System} & \textbf{Noise} & \textbf{ASDS (Debiased)} & \textbf{ASDS (No Debias)} & \textbf{PySINDy v2.1} & \textbf{Weak-SINDy} \\",
        r"\midrule",
    ]
    for sk, slbl in [("van_der_pol", "Van der Pol"), ("duffing", "Duffing")]:
        for target_sig in (0.15, 0.20, 0.25, 0.30):
            mkeys = ["asds_omega_v2", "asds_no_cumulant_debias", "pysindy_official", "weak_sindy"]
            stats_row = []
            for mkey in mkeys:
                found = None
                for entry in exp2[sk][mkey]:
                    if abs(entry["sigma"] - target_sig) < 1e-6:
                        found = entry["coeff_rel_error"]
                        break
                stats_row.append(found)
            best_ce = min(st["mean"] for st in stats_row if st is not None)
            cells = []
            for st in stats_row:
                if st is None:
                    cells.append("---")
                else:
                    txt = _fmt_mean_ci(st, 3)
                    cells.append(f"\\textbf{{{txt}}}" if st["mean"] <= best_ce + 1e-4 else txt)
            t3.append(f"{slbl} & {int(target_sig*100)}\\% & {' & '.join(cells)} \\\\")
    t3.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    p3 = out / "table3_cumulant_eiv.tex"
    p3.write_text("\n".join(t3) + "\n", encoding="utf-8")
    written.append(p3)

    return written


if __name__ == "__main__":
    paths = generate_latex_tables()
    print("Generated LaTeX tables:", [str(p) for p in paths])
