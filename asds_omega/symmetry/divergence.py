"""Liouville Phase-Space Divergence & Rosenstein Maximal Lyapunov Chaos Classifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

from asds_omega.preprocessing.ms_avp import WeakFormSubstrate
from asds_omega.symmetry.lie_prolongation import _estimate_local_jacobians


@dataclass
class DynamicalRegimeProfile:
    """Geometric & stability classification of a dynamical system."""

    mean_divergence: float
    std_divergence: float
    symbolic_divergence: str | None
    max_lyapunov_est: float
    regime_class: str  # "conservative_volume_preserving", "dissipative_chaotic", "dissipative_relaxational", "expanding"


def estimate_max_lyapunov_rosenstein(
    t_list: list[np.ndarray],
    trajectories: list[np.ndarray],
) -> float:
    """Estimate maximal Lyapunov exponent lambda_1 via Rosenstein nearest-neighbor divergence."""
    X = trajectories[0]
    t = t_list[0]
    n = len(X)
    if n < 80:
        return 0.0
    dt = float(np.median(np.diff(t)))
    X_std = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-8)

    # Subsample to at most 400 points for speed
    step = max(1, n // 400)
    Xs = X_std[::step]
    dt_s = dt * step
    m = len(Xs)
    if m < 40:
        return 0.0

    # Temporal separation Theiler window to avoid picking temporally adjacent points
    theiler = max(5, m // 15)
    horizon = min(20, m // 5)
    n_query = m - horizon

    D = np.linalg.norm(Xs[:n_query, None, :] - Xs[None, :n_query, :], axis=2)
    idx = np.arange(n_query)
    mask = np.abs(idx[:, None] - idx[None, :]) <= theiler
    D[mask] = np.inf
    nn = np.argmin(D, axis=1)
    d0 = D[idx, nn]
    valid = np.isfinite(d0) & (d0 > 1e-8)
    if np.sum(valid) < 10:
        return 0.0

    log_div = np.zeros(horizon, dtype=float)
    for k in range(horizon):
        dist_k = np.linalg.norm(Xs[idx[valid] + k] - Xs[nn[valid] + k], axis=1)
        log_div[k] = float(np.mean(np.log(np.maximum(dist_k, 1e-10))))

    t_h = np.arange(horizon, dtype=float) * dt_s
    # Fit slope over early linear growth segment
    fit_len = max(4, horizon // 2)
    slope, _ = np.polyfit(t_h[:fit_len], log_div[:fit_len], 1)
    return float(slope)


def analyze_divergence_and_chaos(
    substrate: WeakFormSubstrate,
    symbolic_equations: list[sp.Expr] | None = None,
) -> DynamicalRegimeProfile:
    """Compute Liouville phase-space divergence div(f) and classify dynamical regime."""
    d = substrate.n_vars
    syms = [sp.Symbol(v) for v in substrate.var_names]
    U = substrate.weak_states
    V = substrate.weak_velocities

    sym_div_str: str | None = None
    if symbolic_equations is not None and len(symbolic_equations) == d:
        div_expr = sp.simplify(sum(sp.diff(eq, s) for eq, s in zip(symbolic_equations, syms)))
        sym_div_str = str(div_expr)
        div_fn = sp.lambdify(syms, div_expr, modules="numpy")
        vals = []
        for x_pt in U[: min(120, len(U))]:
            try:
                val = float(div_fn(*x_pt))
                if np.isfinite(val):
                    vals.append(val)
            except Exception:
                continue
        divs = np.asarray(vals, dtype=float) if vals else np.array([0.0])
    else:
        if len(U) >= 3 * d + 6:
            _, _, J_eval = _estimate_local_jacobians(U, V)
            divs = np.trace(J_eval, axis1=1, axis2=2)
        else:
            divs = np.array([0.0])

    mean_div = float(np.mean(divs))
    std_div = float(np.std(divs))
    lyap = estimate_max_lyapunov_rosenstein(substrate.t_list, substrate.trajectories)

    # Scale-aware classification
    v_scale = float(np.mean(np.std(V, axis=0) / np.maximum(np.std(U, axis=0), 1e-6))) + 1e-6
    rel_div = abs(mean_div) / v_scale

    if rel_div < 0.08 and std_div / v_scale < 0.15:
        regime = "conservative_volume_preserving"
    elif mean_div < -0.05 * v_scale and lyap > 0.12:
        regime = "dissipative_chaotic"
    elif mean_div < -0.05 * v_scale:
        regime = "dissipative_relaxational"
    else:
        regime = "non_conservative_general"

    return DynamicalRegimeProfile(
        mean_divergence=mean_div,
        std_divergence=std_div,
        symbolic_divergence=sym_div_str,
        max_lyapunov_est=lyap,
        regime_class=regime,
    )
