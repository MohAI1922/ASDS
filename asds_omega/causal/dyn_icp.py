"""Multi-Trajectory Invariant Causal Prediction (Dyn-ICP) in Weak-Form Space.

Uses a universal nonparametric RBF Kernel Ridge surrogate across observational
environments E_1, ..., E_E (distinct trajectories or temporal regimes) so it handles
arbitrary nonlinearities (-x/r^3, sin(theta), Michaelis-Menten, chaotic products)
without polynomial basis assumptions, paired with the Manifold-Aware Faithfulness Guard
for systems obeying exact conservation laws (e.g., S + I + R = 1).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from asds_omega.preprocessing.ms_avp import WeakFormSubstrate


def _krr_env_cv_mse(
    X_std: np.ndarray,
    y: np.ndarray,
    env_ids: np.ndarray,
    alpha: float = 0.015,
) -> tuple[float, np.ndarray]:
    """Leave-One-Environment-Out RBF Kernel Ridge MSE and per-environment MSEs."""
    n = len(y)
    if X_std.shape[1] == 0:
        var_y = float(np.var(y))
        envs = np.unique(env_ids)
        return var_y, np.full(len(envs), var_y, dtype=float)

    D2 = cdist(X_std, X_std, metric="sqeuclidean")
    tri = D2[np.triu_indices_from(D2, k=1)]
    med = float(np.median(tri[tri > 1e-12])) if np.any(tri > 1e-12) else 1.0
    gamma = 1.0 / (2.0 * max(med, 1e-6))
    K = np.exp(-gamma * D2)

    envs = np.unique(env_ids)
    mses = []
    for e in envs:
        te = env_ids == e
        tr = ~te
        if np.sum(tr) < 6 or np.sum(te) < 3:
            continue
        K_tr = K[np.ix_(tr, tr)]
        K_te = K[np.ix_(te, tr)]
        w = np.linalg.solve(K_tr + alpha * np.eye(K_tr.shape[0]), y[tr])
        pred = K_te @ w
        mses.append(float(np.mean((y[te] - pred) ** 2)))
    arr = np.asarray(mses, dtype=float)
    return float(np.mean(arr)) if len(arr) else float(np.var(y)), arr


def compute_dyn_icp_matrix(
    substrate: WeakFormSubstrate,
    collinearity_cutoff: float = 0.96,
    max_samples: int = 200,
) -> np.ndarray:
    """Compute (d, d) universal nonparametric Dyn-ICP probability matrix for x_j -> dx_i/dt."""
    U = substrate.weak_states
    V = substrate.weak_velocities
    traj_ids = substrate.traj_ids
    K_tot, d = U.shape
    scores = np.zeros((d, d), dtype=float)
    if K_tot < 16:
        return scores

    # Evenly subsample across trajectories if large
    if K_tot > max_samples:
        idx = np.linspace(0, K_tot - 1, max_samples, dtype=int)
        U = U[idx]
        V = V[idx]
        traj_ids = traj_ids[idx]
        K_tot = max_samples

    uniq_trajs = np.unique(traj_ids)
    if len(uniq_trajs) >= 3:
        env_ids = traj_ids
    else:
        env_ids = np.minimum((np.arange(K_tot) * 4) // K_tot, 3)

    U_std = (U - U.mean(axis=0)) / np.maximum(U.std(axis=0), 1e-8)

    # Manifold-Aware Guard: detect exact linear conservation collinearity among state vars
    collinear_flags = np.zeros(d, dtype=bool)
    for j in range(d):
        others = [k for k in range(d) if k != j]
        if others:
            Z = np.column_stack([U_std[:, others], np.ones(K_tot)])
            coef, *_ = np.linalg.lstsq(Z, U_std[:, j], rcond=None)
            r2_lin = 1.0 - float(np.var(U_std[:, j] - Z @ coef) / (np.var(U_std[:, j]) + 1e-12))
            collinear_flags[j] = r2_lin > collinearity_cutoff

    for i in range(d):
        y = V[:, i]
        y_var = float(np.var(y)) + 1e-12
        full_mse, full_per_env = _krr_env_cv_mse(U_std, y, env_ids)

        for j in range(d):
            others = [k for k in range(d) if k != j]
            if not collinear_flags[j]:
                drop_mse, drop_per_env = _krr_env_cv_mse(U_std[:, others], y, env_ids)
                delta_mse = max(drop_mse - full_mse, 0.0)
                rel_gain = delta_mse / max(full_mse + 0.012 * y_var, 1e-12)
                inv_frac = float(np.mean(drop_per_env > full_per_env * 1.03)) if len(drop_per_env) else 0.0
                p_val = float(1.0 - np.exp(-2.4 * rel_gain)) * (0.35 + 0.65 * inv_frac)
                scores[j, i] = float(np.clip(p_val, 0.0, 1.0))
            else:
                # Manifold-Aware Guard active (e.g., S+I+R=1 or S+P=const):
                best_other_mse = np.inf
                for k in others:
                    mse_k, _ = _krr_env_cv_mse(U_std[:, [k]], y, env_ids)
                    best_other_mse = min(best_other_mse, mse_k)

                mse_j, _ = _krr_env_cv_mse(U_std[:, [j]], y, env_ids)
                if best_other_mse <= full_mse + 0.015 * y_var:
                    # Single other variable suffices; j is causal iff j itself is that sufficient var
                    scores[j, i] = 0.95 if (mse_j <= best_other_mse * 1.05 and mse_j <= full_mse + 0.015 * y_var) else 0.02
                else:
                    pair_with_j = np.inf
                    for k in others:
                        m_jk, _ = _krr_env_cv_mse(U_std[:, sorted([j, k])], y, env_ids)
                        pair_with_j = min(pair_with_j, m_jk)
                    gain_over_other = max(best_other_mse - pair_with_j, 0.0) / max(pair_with_j + 0.015 * y_var, 1e-12)
                    r2_j = max(1.0 - mse_j / y_var, 0.0)
                    scores[j, i] = float(np.clip((1.0 - np.exp(-2.0 * gain_over_other)) * (0.4 + 0.6 * r2_j), 0.0, 1.0))

    return scores
