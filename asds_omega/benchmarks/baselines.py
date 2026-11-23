"""Head-to-head scientific baselines run on the exact same datasets, noise, and seeds.

Implemented baselines:
1. Standard SINDy (Savitzky-Golay differentiation + STLSQ, Brunton et al. 2016)
2. Standard Weak-SINDy (fixed bump window, no cumulant EIV debiasing, no causal/symmetry constraints)
3. Pure Unconstrained GP Symbolic Regression (pointwise Savitzky-Golay targets, no semantic mutation)
4. Black-Box Neural MLP Surrogate (evaluated on held-out and 2x out-of-distribution extrapolation)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import sympy as sp
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from asds_omega.config import EVGHEConfig, MSAVPConfig
from asds_omega.data.synthetic import DynamicalDataset
from asds_omega.preprocessing.ms_avp import MSAVPProjector
from asds_omega.scientist.evidence_scorer import compute_symbolic_match
from asds_omega.symbolic.srg_gp import SemanticWeakFormGP
from asds_omega.symbolic.vesp import VESPEngine


@dataclass
class BaselineResult:
    """Standardized evaluation result for fair head-to-head comparison against ASDS-OMEGA v2.0."""

    method: str
    dataset_name: str
    equations: list[str]
    mean_r2: float
    symbolic_match: float
    exact_structure_rate: float
    mean_coeff_rel_error: float
    mean_complexity: float
    runtime_sec: float
    ood_extrapolation_rmse: float | None = None


def _evaluate_against_gt(
    eqs: list[sp.Expr],
    r2s: list[float],
    complexities: list[int],
    dataset: DynamicalDataset,
    method: str,
    runtime: float,
) -> BaselineResult:
    matches, exacts, cerrs = [], [], []
    for eq, gt in zip(eqs, dataset.ground_truth_equations):
        sm, ex, ce = compute_symbolic_match(eq, gt, dataset.var_names)
        matches.append(sm)
        exacts.append(float(ex))
        cerrs.append(ce)
    return BaselineResult(
        method=method,
        dataset_name=dataset.name,
        equations=[str(e) for e in eqs],
        mean_r2=float(np.mean(r2s)),
        symbolic_match=float(np.mean(matches)),
        exact_structure_rate=float(np.mean(exacts)),
        mean_coeff_rel_error=float(np.mean(cerrs)),
        mean_complexity=float(np.mean(complexities)),
        runtime_sec=runtime,
    )


def run_standard_sindy_baseline(dataset: DynamicalDataset, seed: int = 42) -> BaselineResult:
    """Classical SINDy baseline: Savitzky-Golay derivatives + unconstrained STLSQ."""
    t0 = time.perf_counter()
    sub = MSAVPProjector(MSAVPConfig(use_weak_form=False, cumulant_debiasing=False)).fit_transform(
        dataset.t_list, dataset.trajectories, dataset.var_names
    )
    vesp = VESPEngine(
        EVGHEConfig(
            include_rational_implicit=False,
            enable_gp_refinement=False,
            enable_rationalizer=False,
            n_ensemble_bootstraps=1,
            inclusion_prob_threshold=0.5,
        ),
        seed=seed,
    )
    cands, _, _ = vesp.discover_system(sub, causal_result=None, symmetry_result=None)
    rt = time.perf_counter() - t0
    return _evaluate_against_gt(
        [c.expr for c in cands],
        [c.weak_r2 for c in cands],
        [c.complexity for c in cands],
        dataset,
        "Standard SINDy (STLSQ)",
        rt,
    )


def run_weak_sindy_baseline(dataset: DynamicalDataset, seed: int = 42) -> BaselineResult:
    """Standard Weak-SINDy baseline: fixed test window, NO cumulant debiasing, NO VIFC/W-LIND."""
    t0 = time.perf_counter()
    sub = MSAVPProjector(
        MSAVPConfig(
            use_weak_form=True,
            adaptive_multiscale=False,
            cumulant_debiasing=False,
            min_window_pts=25,
            max_window_pts=25,
        )
    ).fit_transform(dataset.t_list, dataset.trajectories, dataset.var_names)
    vesp = VESPEngine(
        EVGHEConfig(
            include_rational_implicit=False,
            enable_gp_refinement=False,
            enable_rationalizer=False,
            n_ensemble_bootstraps=1,
            inclusion_prob_threshold=0.5,
        ),
        seed=seed,
    )
    cands, _, _ = vesp.discover_system(sub, causal_result=None, symmetry_result=None)
    rt = time.perf_counter() - t0
    return _evaluate_against_gt(
        [c.expr for c in cands],
        [c.weak_r2 for c in cands],
        [c.complexity for c in cands],
        dataset,
        "Weak-SINDy (Standard)",
        rt,
    )


def run_pure_gp_baseline(dataset: DynamicalDataset, seed: int = 42) -> BaselineResult:
    """Unconstrained GP Symbolic Regression baseline (Savitzky-Golay targets, random mutation)."""
    t0 = time.perf_counter()
    sub = MSAVPProjector(MSAVPConfig(use_weak_form=False, cumulant_debiasing=False)).fit_transform(
        dataset.t_list, dataset.trajectories, dataset.var_names
    )
    gp = SemanticWeakFormGP(
        EVGHEConfig(
            gp_population_size=24,
            gp_generations=12,
            gp_semantic_mutation_rate=0.0,  # Pure random mutation (no semantic guidance)
            enable_rationalizer=False,
        ),
        seed=seed,
    )
    all_vars = list(range(dataset.n_vars))
    eqs, r2s, comps = [], [], []
    for i in range(dataset.n_vars):
        cands = gp.evolve_target(sub, target_idx=i, allowed_vars=all_vars, reynolds=None)
        best = cands[0]
        eqs.append(best.expr)
        r2s.append(best.weak_r2)
        comps.append(best.complexity)
    rt = time.perf_counter() - t0
    return _evaluate_against_gt(eqs, r2s, comps, dataset, "Pure GP (Unconstrained)", rt)


def run_neural_mlp_baseline(dataset: DynamicalDataset, seed: int = 42) -> BaselineResult:
    """Black-box Neural Vector-Field MLP surrogate evaluated on in-domain and 2x OOD extrapolation."""
    t0 = time.perf_counter()
    sub = MSAVPProjector(MSAVPConfig(use_weak_form=False, cumulant_debiasing=False)).fit_transform(
        dataset.t_list, dataset.trajectories, dataset.var_names
    )
    X_all = sub.weak_states
    Y_all = sub.weak_velocities
    n = len(X_all)
    split = int(0.8 * n)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_all[:split])
    X_te = scaler.transform(X_all[split:])
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=250, random_state=seed, early_stopping=True)
    mlp.fit(X_tr, Y_all[:split])
    pred_te = mlp.predict(X_te)
    if pred_te.ndim == 1:
        pred_te = pred_te[:, None]

    r2s = [
        float(1.0 - np.mean((Y_all[split:, i] - pred_te[:, i]) ** 2) / (np.var(Y_all[split:, i]) + 1e-12))
        for i in range(dataset.n_vars)
    ]

    # Out-of-Distribution (2x scaled state space) extrapolation RMSE vs exact ground-truth rhs_fn
    ood_rmse: float | None = None
    if dataset.rhs_fn is not None:
        rng = np.random.default_rng(seed)
        mu = np.mean(X_all, axis=0)
        sd = np.maximum(np.std(X_all, axis=0), 0.5)
        X_ood = mu + rng.uniform(-3.0, 3.0, size=(120, dataset.n_vars)) * sd
        if dataset.name in {"sir", "lotka_volterra", "michaelis_menten"}:
            X_ood = np.abs(X_ood) + 0.2
        Y_true_ood = np.array([dataset.rhs_fn(0.0, pt) for pt in X_ood], dtype=float)
        Y_pred_ood = mlp.predict(scaler.transform(X_ood))
        if Y_pred_ood.ndim == 1:
            Y_pred_ood = Y_pred_ood[:, None]
        ood_rmse = float(np.sqrt(np.mean((Y_true_ood - Y_pred_ood) ** 2)) / (np.std(Y_true_ood) + 1e-8))

    rt = time.perf_counter() - t0
    return BaselineResult(
        method="Black-Box Neural MLP",
        dataset_name=dataset.name,
        equations=["<black-box neural network>" for _ in dataset.var_names],
        mean_r2=float(np.mean(r2s)),
        symbolic_match=0.0,
        exact_structure_rate=0.0,
        mean_coeff_rel_error=1.0,
        mean_complexity=float(64 * 64),
        runtime_sec=rt,
        ood_extrapolation_rmse=ood_rmse,
    )


def run_pysindy_official_baseline(dataset: DynamicalDataset, seed: int = 42) -> BaselineResult:
    """External SOTA Baseline: Official PySINDy v2.1 (Ensemble-STLSQ + Savitzky-Golay / Weak library)."""
    import pysindy as ps

    t0 = time.perf_counter()
    d = dataset.n_vars
    syms = [sp.Symbol(v) for v in dataset.var_names]
    poly_deg = 3 if d <= 3 else 2

    # Combine Polynomial + Fourier library cleanly via PySINDy's official GeneralizedLibrary
    libs = [ps.PolynomialLibrary(degree=poly_deg, include_bias=True)]
    if dataset.name in {"pendulum", "damped_pendulum"}:
        libs.append(ps.FourierLibrary(n_frequencies=1))
    feature_lib = ps.ConcatLibrary(libs) if len(libs) > 1 else libs[0]

    opt = ps.STLSQ(threshold=0.08, alpha=1e-4)
    diff = ps.SmoothedFiniteDifference(smoother_kws={"window_length": 15, "polyorder": 3})
    model = ps.SINDy(optimizer=opt, feature_library=feature_lib, differentiation_method=diff)

    try:
        model.fit(
            dataset.trajectories,
            t=dataset.t_list,
            feature_names=dataset.var_names,
        )
        r2_score = float(model.score(dataset.trajectories, t=dataset.t_list))
        feat_names = model.get_feature_names()
        coef_mat = model.coefficients()  # (d, n_features)
    except Exception:
        rt = time.perf_counter() - t0
        return BaselineResult(
            method="PySINDy v2.1 (Official)",
            dataset_name=dataset.name,
            equations=["0" for _ in dataset.var_names],
            mean_r2=0.0,
            symbolic_match=0.0,
            exact_structure_rate=0.0,
            mean_coeff_rel_error=1.0,
            mean_complexity=1.0,
            runtime_sec=rt,
        )

    # Parse PySINDy feature strings (e.g., "x y", "x^2", "sin(1 theta)") into exact SymPy expressions
    local_map = {v: s for v, s in zip(dataset.var_names, syms)}
    sym_feats: list[sp.Expr] = []
    for fn_str in feat_names:
        s_clean = fn_str.strip().replace("^", "**")
        s_clean = s_clean.replace("sin(1 ", "sin(").replace("cos(1 ", "cos(")
        # Replace whitespace product separators (e.g., "x y" -> "x*y")
        tokens = s_clean.split()
        expr_str = "*".join(tokens) if len(tokens) > 1 else s_clean
        try:
            sym_feats.append(sp.sympify(expr_str, locals=local_map))
        except Exception:
            sym_feats.append(sp.Integer(0))

    eqs: list[sp.Expr] = []
    comps: list[int] = []
    for i in range(d):
        expr = sp.Integer(0)
        for c_val, sf in zip(coef_mat[i], sym_feats):
            if abs(float(c_val)) > 1e-4:
                expr = expr + sp.Float(round(float(c_val), 5)) * sf
        expr = sp.simplify(expr)
        eqs.append(expr)
        comps.append(int(sp.count_ops(expr)) + len(expr.free_symbols) + 1)

    rt = time.perf_counter() - t0
    return _evaluate_against_gt(
        eqs,
        [max(r2_score, 0.0)] * d,
        comps,
        dataset,
        "PySINDy v2.1 (Official)",
        rt,
    )
