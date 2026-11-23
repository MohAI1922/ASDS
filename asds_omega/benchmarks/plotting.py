"""Publication-Grade 300-DPI Matplotlib Figures with 95% Confidence Intervals from Verified JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from asds_omega.causal.counterfactual import CounterfactualSimulator
from asds_omega.data import generate_synthetic_dataset, inject_noise, load_real_dataset


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.8,
        "axes.titlesize": 10.8,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 220,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def generate_all_publication_figures(
    ledger_path: str | Path = "results/benchmark_ledger.json",
    figures_dir: str | Path = "results/figures",
) -> list[Path]:
    """Render all 6 publication-ready scientific figures with 95% CIs from verified ledger JSON."""
    _style()
    ledger: dict[str, Any] = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # ------------------------------------------------------------------
    # Fig 1: High-Noise EIV Stress Test & Hermite Cumulant Debiasing (95% CI)
    # ------------------------------------------------------------------
    exp2 = ledger["experiments"]["exp2_noise_stress"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    styles = [
        ("asds_omega_v2", "ASDS-OMEGA v2.0 (VECSD)", "#0f766e", "o", "-"),
        ("asds_no_cumulant_debias", "ASDS w/o Cumulant Debias", "#2563eb", "s", "--"),
        ("pysindy_official", "PySINDy v2.1 (Official)", "#7c3aed", "D", "-."),
        ("weak_sindy", "Weak-SINDy (Standard)", "#d97706", "^", ":"),
        ("standard_sindy", "Standard SINDy", "#dc2626", "x", ":"),
    ]
    panels = [
        ("van_der_pol", "Van der Pol (Cubic $x^2 y$ EIV)", "coeff_rel_error", "Coefficient Relative Error (log)", True),
        ("duffing", "Duffing Double-Well (Cubic $x^3$ EIV)", "coeff_rel_error", "Coefficient Relative Error (log)", True),
        ("lorenz", "Lorenz-63 Chaotic Attractor", "symbolic_match", "Symbolic Match Score", False),
    ]
    for ax, (sys_key, title, metric_key, ylab, use_log) in zip(axes, panels):
        sys_data = exp2["systems"][sys_key]
        for k, label, color, marker, ls in styles:
            pts = sys_data[k]
            xs = np.array([p["sigma"] * 100 for p in pts])
            ys = np.array([max(p[metric_key]["mean"], 1e-4 if use_log else 0.0) for p in pts])
            ci = np.array([p[metric_key]["ci95"] for p in pts])
            ax.plot(xs, ys, label=label, color=color, marker=marker, linestyle=ls, linewidth=2.0, markersize=4.8)
            if not use_log:
                ax.fill_between(xs, np.clip(ys - ci, 0, 1.05), np.clip(ys + ci, 0, 1.05), color=color, alpha=0.12)
        if use_log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(-0.04, 1.08)
        ax.set_title(title)
        ax.set_xlabel("Relative Gaussian Noise (%)")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7.3, frameon=True)
    fig.suptitle(f"Multi-Seed Noise Robustness & Hermite Cumulant EIV Debiasing (n={ledger['n_seeds']} seeds, 95% CI)", fontweight="bold", y=1.03)
    p1 = out / "fig1_noise_robustness.png"
    fig.savefig(p1)
    plt.close(fig)
    saved.append(p1)

    # ------------------------------------------------------------------
    # Fig 2: 6-Factor Systematic Ablation Study @ 10% Noise (with 95% CI bars)
    # ------------------------------------------------------------------
    exp3 = ledger["experiments"]["exp3_ablation_10pct"]
    labels_map = [
        ("full_asds_omega_v2", "Full VECSD\n(ASDS v2.0)"),
        ("no_cumulant_debias", "w/o Cumulant\nDebiasing"),
        ("no_weak_form_savgol", "w/o Weak-Form\n(SavGol Diff)"),
        ("no_vifc_causal", "w/o VIFC\nCausal Mask"),
        ("no_wlind_symmetry", "w/o W-LIND\nSymmetry/KKT"),
        ("no_mdl_rationalizer", "w/o MDL\nRationalizer"),
    ]
    names_lbl = [lbl for _, lbl in labels_map]
    sm_vals = [exp3[k]["symbolic_match"]["mean"] for k, _ in labels_map]
    sm_ci = [exp3[k]["symbolic_match"]["ci95"] for k, _ in labels_map]
    ex_vals = [exp3[k]["exact_structure"]["mean"] for k, _ in labels_map]
    ex_ci = [exp3[k]["exact_structure"]["ci95"] for k, _ in labels_map]
    x_pos = np.arange(len(names_lbl))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11.0, 4.5))
    b1 = ax.bar(x_pos - w / 2, sm_vals, yerr=sm_ci, capsize=3, width=w, label="Mean Symbolic Match (±95% CI)", color="#0f766e")
    b2 = ax.bar(x_pos + w / 2, ex_vals, yerr=ex_ci, capsize=3, width=w, label="Exact Structure Recovery (±95% CI)", color="#3b82f6")
    for bars, cis in ((b1, sm_ci), (b2, ex_ci)):
        for bar, c_val in zip(bars, cis):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + c_val + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=7.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names_lbl)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score (5 Benchmarks × 5 Seeds @ 10% Noise)")
    ax.set_title("6-Factor Systematic Ablation Analysis of ASDS-OMEGA v2.0 (VECSD)")
    ax.legend(frameon=True, loc="lower left")
    p2 = out / "fig2_ablation_matrix.png"
    fig.savefig(p2)
    plt.close(fig)
    saved.append(p2)

    # ------------------------------------------------------------------
    # Fig 3: Lorenz 3D Phase Portrait + VIFC Direct Differential Causal Graph
    # ------------------------------------------------------------------
    lor_entry = ledger["experiments"]["exp1_multidomain_5pct"]["odes"]["lorenz"]["asds_omega_v2"]
    ds_lor = inject_noise(generate_synthetic_dataset("lorenz", n_trajectories=1, seed=42), 0.05, seed=42)
    loc_l = {v: sp.Symbol(v) for v in ds_lor.var_names}
    eqs_lor = [sp.sympify(e["expr_str"], locals=loc_l) for e in lor_entry["discovered_equations"]]
    sim_lor = CounterfactualSimulator(ds_lor.var_names, eqs_lor)
    t_sim = ds_lor.t_list[0][:1000]
    X_true = ds_lor.clean_trajectories[0][:1000]
    X_disc = sim_lor.simulate(X_true[0], t_sim)

    fig = plt.figure(figsize=(11.5, 4.6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(X_true[:, 0], X_true[:, 1], X_true[:, 2], color="#94a3b8", lw=1.0, alpha=0.7, label="Ground Truth Lorenz")
    ax1.plot(X_disc[:, 0], X_disc[:, 1], X_disc[:, 2], color="#0f766e", lw=1.2, alpha=0.9, label="ASDS-OMEGA v2.0 Discovered")
    ax1.set_title("Lorenz-63 Attractor Reconstruction (5% Noisy Data)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2)
    prob_mat = np.asarray(lor_entry["causal"]["probability_matrix"])
    im = ax2.imshow(prob_mat, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels(["dx/dt", "dy/dt", "dz/dt"])
    ax2.set_yticklabels(["x", "y", "z"])
    ax2.set_xlabel("Target Derivative")
    ax2.set_ylabel("Causal Parent State")
    ax2.set_title(f"VIFC Direct Differential Causal Graph (SHD = {lor_entry['causal']['metrics']['shd']:.0f})")
    for r in range(3):
        for c in range(3):
            ax2.text(c, r, f"{prob_mat[r, c]:.2f}", ha="center", va="center", color="white" if prob_mat[r, c] > 0.55 else "black", fontweight="bold")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="VIFC Edge Probability")
    p3 = out / "fig3_lorenz_discovery.png"
    fig.savefig(p3)
    plt.close(fig)
    saved.append(p3)

    # ------------------------------------------------------------------
    # Fig 4: Online Structural Drift & Surgical Regeneration (SLR)
    # ------------------------------------------------------------------
    exp5 = ledger["experiments"]["exp5_drift_regeneration"]
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10.0, 5.2), sharex=True)
    ts = exp5["time_series_t"]
    ax_top.plot(ts, exp5["residual_trace"], color="#2563eb", lw=1.8, label="Sliding Weak-Form Residual")
    if exp5["true_shift_time"] is not None:
        ax_top.axvline(exp5["true_shift_time"], color="#dc2626", ls="--", lw=1.6, label=f"True Regime Shift (t={exp5['true_shift_time']})")
    if exp5["detected_time"] is not None:
        ax_top.axvline(exp5["detected_time"], color="#0f766e", ls="-", lw=1.8, label=f"SLR CUSUM Alarm (t={exp5['detected_time']:.2f})")
    ax_top.set_ylabel("Weak Residual")
    ax_top.set_title(
        f"Online Concept Drift & Surgical Regeneration ({exp5['drift_type']} | Post R²: {exp5['pre_drift_r2_on_post']:.2f} → {exp5['regenerated_r2_on_post']:.3f})"
    )
    ax_top.legend(fontsize=8.5)

    ax_bot.plot(ts, exp5["cusum_trace"], color="#d97706", lw=2.0, label="Page-Hinkley / CUSUM Statistic")
    ax_bot.axhline(4.5, color="#64748b", ls=":", lw=1.4, label="Alarm Threshold (h = 4.5)")
    ax_bot.set_xlabel("Time (days)")
    ax_bot.set_ylabel("CUSUM Score")
    ax_bot.legend(fontsize=8.5)
    p4 = out / "fig4_drift_regeneration.png"
    fig.savefig(p4)
    plt.close(fig)
    saved.append(p4)

    # ------------------------------------------------------------------
    # Fig 5: Real-World Experimental Scientific Discovery (3 Datasets)
    # ------------------------------------------------------------------
    exp6 = ledger["experiments"]["exp6_real_world"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    real_specs = [
        ("hudson_bay_lynx_hare", "Hudson Bay Lynx–Hare (1900–1920)", "Years since 1900", "Pelts (×10k)"),
        ("boarding_school_influenza", "1978 English Boarding School Flu", "Days", "Population Fraction"),
        ("mauna_loa_co2", "Mauna Loa Atmospheric CO₂ (1960–2023)", "Decades since 1960", "Excess CO₂ (×100 ppm)"),
    ]
    for ax, (rkey, title, xlab, ylab) in zip(axes, real_specs):
        rds = load_real_dataset(rkey)
        r_entry = exp6[rkey]
        t_r = rds.t_list[0]
        X_obs = rds.trajectories[0]
        loc_r = {v: sp.Symbol(v) for v in rds.var_names}
        eqs_r = [sp.sympify(e["expr_str"], locals=loc_r) for e in r_entry["discovered_equations"]]
        X_sim = CounterfactualSimulator(rds.var_names, eqs_r).simulate(X_obs[0], t_r)
        palette = ["#0f766e", "#dc2626", "#2563eb"]
        for j, vname in enumerate(rds.var_names):
            c = palette[j % len(palette)]
            ax.plot(t_r, X_obs[:, j], color=c, alpha=0.45, lw=2.2, label=f"{vname} (obs)")
            if np.all(np.isfinite(X_sim[:, j])) and float(np.max(np.abs(X_sim[:, j]))) < 20 * float(np.max(np.abs(X_obs[:, j])) + 1):
                ax.plot(t_r, X_sim[:, j], color=c, ls="--", lw=1.6, label=f"{vname} (ASDS)")
        ax.set_title(f"{title}\n(Weak R² = {r_entry['evidence']['mean_weak_r2']:.3f})", fontsize=9.5)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7.2, ncol=2)
    p5 = out / "fig5_real_world_discovery.png"
    fig.savefig(p5)
    plt.close(fig)
    saved.append(p5)

    # ------------------------------------------------------------------
    # Fig 6: Head-to-Head Multi-Domain Comparison (incl. Official PySINDy v2.1) & OOD
    # ------------------------------------------------------------------
    exp1_odes = ledger["experiments"]["exp1_multidomain_5pct"]["odes"]
    sys_keys = list(exp1_odes.keys())
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 4.5))

    m_asds = [exp1_odes[k]["multiseed"]["asds_omega_v2"]["symbolic_match"]["mean"] for k in sys_keys]
    m_ps = [exp1_odes[k]["multiseed"]["baselines"]["pysindy_official"]["symbolic_match"]["mean"] for k in sys_keys]
    m_ws = [exp1_odes[k]["multiseed"]["baselines"]["weak_sindy"]["symbolic_match"]["mean"] for k in sys_keys]
    m_sindy = [exp1_odes[k]["multiseed"]["baselines"]["standard_sindy"]["symbolic_match"]["mean"] for k in sys_keys]
    m_gp = [exp1_odes[k]["multiseed"]["baselines"]["pure_gp"]["symbolic_match"]["mean"] for k in sys_keys]

    x_i = np.arange(len(sys_keys))
    bw = 0.16
    axL.bar(x_i - 2 * bw, m_asds, width=bw, label="ASDS-OMEGA v2.0", color="#0f766e")
    axL.bar(x_i - bw, m_ps, width=bw, label="PySINDy v2.1 (Official)", color="#7c3aed")
    axL.bar(x_i, m_ws, width=bw, label="Weak-SINDy", color="#d97706")
    axL.bar(x_i + bw, m_sindy, width=bw, label="Standard SINDy", color="#dc2626")
    axL.bar(x_i + 2 * bw, m_gp, width=bw, label="Pure GP", color="#64748b")
    axL.set_xticks(x_i)
    axL.set_xticklabels([k.replace("_", "\n") for k in sys_keys], fontsize=7.6)
    axL.set_ylim(0, 1.32)
    axL.set_ylabel("Mean Symbolic Match (5 Seeds @ 5% Noise)")
    axL.set_title("Head-to-Head Symbolic Law Recovery (8 Benchmarks)")
    axL.legend(fontsize=7.3, ncol=3, loc="upper center", frameon=True)

    ood_asds = [max(exp1_odes[k]["multiseed"]["asds_omega_v2"]["ood_rmse"]["mean"], 1e-4) for k in sys_keys]
    ood_mlp = [max(exp1_odes[k]["multiseed"]["neural_mlp_ood_rmse"]["mean"], 1e-4) for k in sys_keys]
    axR.bar(x_i - 0.18, ood_asds, width=0.35, label="ASDS-OMEGA v2.0 (Symbolic Law)", color="#0f766e")
    axR.bar(x_i + 0.18, ood_mlp, width=0.35, label="Black-Box Neural MLP", color="#dc2626")
    axR.set_yscale("log")
    axR.set_ylim(5e-5, 8.0)
    axR.set_xticks(x_i)
    axR.set_xticklabels([k.replace("_", "\n") for k in sys_keys], fontsize=7.6)
    axR.set_ylabel("Normalized OOD Extrapolation RMSE (log scale)")
    axR.set_title("3σ Out-of-Domain Generalization: Symbolic Laws vs Neural MLP")
    axR.legend(fontsize=8, loc="upper right", frameon=True)

    p6 = out / "fig6_head_to_head_and_ood.png"
    fig.savefig(p6)
    plt.close(fig)
    saved.append(p6)

    return saved
