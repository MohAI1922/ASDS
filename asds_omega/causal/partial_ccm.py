"""Multivariate Partial Convergent Cross Mapping (pCCM) in Delay-Differential Space.

Extends Sugihara's CCM to multivariate chaotic systems by projecting out non-collinear
mediator manifolds via local ridge regression before measuring cross-map convergence
skill rho(L_max) and convergence slope rho(L_max) - rho(L_min).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from asds_omega.preprocessing.ms_avp import WeakFormSubstrate


def _delay_embed(series: np.ndarray, dim: int, tau: int) -> np.ndarray:
    n = len(series)
    m = n - (dim - 1) * tau
    if m < 10:
        return series[:, None]
    cols = [series[k * tau : k * tau + m] for k in range(dim)]
    return np.column_stack(cols)


def _ccm_skill(M_target: np.ndarray, source: np.ndarray, lib_size: int, k_nn: int, rng: np.random.Generator) -> float:
    """Compute simplex cross-map Pearson correlation predicting source from shadow manifold M_target."""
    n = len(M_target)
    lib_size = min(lib_size, n)
    idx = rng.choice(n, size=lib_size, replace=False) if lib_size < n else np.arange(n)
    M_sub = M_target[idx]
    s_sub = source[idx]

    D = cdist(M_sub, M_sub, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    k = min(k_nn, lib_size - 1)
    nn_idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    nn_dist = np.take_along_axis(D, nn_idx, axis=1)

    d_min = np.maximum(np.min(nn_dist, axis=1, keepdims=True), 1e-12)
    weights = np.exp(-nn_dist / d_min)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    preds = np.sum(weights * s_sub[nn_idx], axis=1)

    if float(np.std(preds)) < 1e-10 or float(np.std(s_sub)) < 1e-10:
        return 0.0
    corr = float(np.corrcoef(preds, s_sub)[0, 1])
    return float(np.clip(corr, 0.0, 1.0)) if np.isfinite(corr) else 0.0


def compute_partial_ccm_matrix(
    substrate: WeakFormSubstrate,
    embed_dim: int = 3,
    tau: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """Compute (d, d) Partial CCM score matrix for x_j -> dx_i/dt."""
    rng = np.random.default_rng(seed)
    U = substrate.weak_states
    V = substrate.weak_velocities
    K, d = U.shape
    scores = np.zeros((d, d), dtype=float)

    if K < 25:
        return scores

    # Use first trajectory's sequential subdomains to preserve temporal order for Takens embedding
    mask0 = substrate.traj_ids == 0
    U0 = U[mask0]
    V0 = V[mask0]
    n0 = len(U0)
    if n0 < 20:
        U0, V0, n0 = U, V, K

    U0_std = (U0 - U0.mean(axis=0)) / np.maximum(U0.std(axis=0), 1e-8)
    V0_std = (V0 - V0.mean(axis=0)) / np.maximum(V0.std(axis=0), 1e-8)

    dim = min(embed_dim, max(2, n0 // 12))
    k_nn = dim + 1

    for i in range(d):
        # Embed target velocity manifold M_{v_i}
        M_vi = _delay_embed(V0_std[:, i], dim=dim, tau=tau)
        m = len(M_vi)
        if m < 15:
            continue
        L_small = max(k_nn + 3, m // 3)
        L_large = m

        for j in range(d):
            src = U0_std[-m:, j]
            # Partial out non-collinear linear mediators to suppress indirect chains
            others = [k for k in range(d) if k != j]
            if others:
                Z = U0_std[-m:, others]
                coef, *_ = np.linalg.lstsq(Z, src, rcond=None)
                res_src = src - Z @ coef
                r2_col = 1.0 - float(np.var(res_src) / (np.var(src) + 1e-12))
                # Faithfulness guard: only residualize when mediators aren't deterministically collinear
                use_src = res_src if r2_col < 0.94 else src
            else:
                use_src = src

            rho_small = _ccm_skill(M_vi, use_src, L_small, k_nn, rng)
            rho_large = _ccm_skill(M_vi, use_src, L_large, k_nn, rng)
            conv_gain = max(rho_large - rho_small, 0.0)
            # Combine asymptotic cross-map skill and positive convergence with library length
            scores[j, i] = float(np.clip(0.65 * rho_large + 1.4 * conv_gain, 0.0, 1.0))

    return scores
