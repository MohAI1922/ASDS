"""Model-Free Weak-Form Discrete Group Equivariance Discovery (Z_2 Reflections & Inversions).

Tests finite symmetry group actions g: x -> g * x directly on the weak-form state-velocity
manifold (U_k, V_k) BEFORE symbolic regression runs, using exact Reynolds-subspace excess
error on a cumulant-debiased polynomial+trig+radial probe basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement, product

import numpy as np
from sklearn.linear_model import Ridge

from asds_omega.preprocessing.ms_avp import WeakFormSubstrate, debias_trig_feature


@dataclass
class DiscreteSymmetry:
    """A discovered discrete sign-flip equivariance g * f(x) = f(g * x)."""

    signs: tuple[int, ...]
    equivariance_error: float
    description: str


def _build_parity_probe_matrix(substrate: WeakFormSubstrate) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Build debiased degree-2 + sin/cos + radial-inverse probe matrix with exact parity tags."""
    d = substrate.n_vars
    K = substrate.n_subdomains
    cols: list[np.ndarray] = [np.ones(K, dtype=float)]
    parities: list[tuple[int, ...]] = [tuple([0] * d)]

    max_probe_deg = 3 if d <= 3 else 2
    for deg in range(1, max_probe_deg + 1):
        for comb in combinations_with_replacement(range(d), deg):
            powers = [0] * d
            for idx in comb:
                powers[idx] += 1
            cols.append(substrate.project_monomial(tuple(powers)))
            parities.append(tuple(p % 2 for p in powers))

    all_X = np.vstack(substrate.trajectories)
    for j in range(d):
        if float(np.max(all_X[:, j]) - np.min(all_X[:, j])) < 20.0:
            s_trajs = [
                debias_trig_feature(X[:, j], "sin", float(substrate.noise_vars[j]), substrate.cumulant_debiasing)[:, None]
                for X in substrate.trajectories
            ]
            c_trajs = [
                debias_trig_feature(X[:, j], "cos", float(substrate.noise_vars[j]), substrate.cumulant_debiasing)[:, None]
                for X in substrate.trajectories
            ]
            cols.append(substrate.project_pointwise(s_trajs)[:, 0])
            p_odd = [0] * d
            p_odd[j] = 1
            parities.append(tuple(p_odd))
            cols.append(substrate.project_pointwise(c_trajs)[:, 0])
            parities.append(tuple([0] * d))

    # Central-force inverse-cube terms x_0/r^3, x_1/r^3 (odd in x_0 / x_1 respectively, even in others)
    if d >= 4:
        r_vals = np.sqrt(all_X[:, 0] ** 2 + all_X[:, 1] ** 2)
        if float(np.min(r_vals)) > 0.15:
            for coord in (0, 1):
                r3_trajs = [
                    (X[:, coord] / np.maximum(X[:, 0] ** 2 + X[:, 1] ** 2, 1e-4) ** 1.5)[:, None]
                    for X in substrate.trajectories
                ]
                cols.append(substrate.project_pointwise(r3_trajs)[:, 0])
                p_odd = [0] * d
                p_odd[coord] = 1
                parities.append(tuple(p_odd))

    return np.column_stack(cols), parities


def _feature_sign(parity: tuple[int, ...], g_signs: tuple[int, ...]) -> int:
    s = 1
    for p, sg in zip(parity, g_signs):
        if p % 2 == 1:
            s *= sg
    return s


def discover_discrete_symmetries(
    substrate: WeakFormSubstrate,
    eq_threshold: float = 0.18,
) -> list[DiscreteSymmetry]:
    """Discover discrete Z_2 reflection/inversion equivariances g * f(x) = f(g * x)."""
    d = substrate.n_vars
    if d > 6:
        return []

    V = substrate.weak_velocities
    all_X = np.vstack(substrate.trajectories)

    # A coordinate can participate in a sign-flip symmetry only if data visits both signs
    can_flip = [
        (float(np.min(all_X[:, j])) < -0.05 * float(np.std(all_X[:, j])))
        and (float(np.max(all_X[:, j])) > 0.05 * float(np.std(all_X[:, j])))
        for j in range(d)
    ]
    if not any(can_flip):
        return []

    Phi, parities = _build_parity_probe_matrix(substrate)
    mu = Phi.mean(axis=0)
    sd = np.maximum(Phi.std(axis=0), 1e-8)
    Phi_s = (Phi - mu) / sd
    Phi_s[:, 0] = 1.0

    # Baseline unconstrained weak-form residual per equation
    base_err = np.zeros(d, dtype=float)
    y_norm = np.zeros(d, dtype=float)
    for i in range(d):
        y = V[:, i]
        y_norm[i] = float(np.linalg.norm(y)) + 1e-12
        mdl = Ridge(alpha=1e-3, fit_intercept=False).fit(Phi_s, y)
        base_err[i] = float(np.linalg.norm(y - mdl.predict(Phi_s)) / y_norm[i])

    rel_noise = float(substrate.metadata.get("relative_noise_est", 0.0))
    eff_thresh = 0.045 + 0.35 * rel_noise

    discovered: list[DiscreteSymmetry] = []
    choices = [(-1, 1) if can_flip[j] else (1,) for j in range(d)]
    for g_signs in product(*choices):
        if all(s == 1 for s in g_signs):
            continue

        # Exact Reynolds-projected subspace excess error across all d equations
        excess_errs = []
        for i in range(d):
            target_sign = g_signs[i]
            keep = [m for m, par in enumerate(parities) if _feature_sign(par, g_signs) == target_sign]
            if not keep:
                excess_errs.append(1.0)
                continue
            y = V[:, i]
            mdl_eq = Ridge(alpha=1e-3, fit_intercept=False).fit(Phi_s[:, keep], y)
            err_eq = float(np.linalg.norm(y - mdl_eq.predict(Phi_s[:, keep])) / y_norm[i])
            excess_errs.append(max(err_eq - base_err[i], 0.0))

        max_excess = float(np.max(excess_errs))
        if max_excess <= min(eff_thresh, eq_threshold):
            lhs = ", ".join(substrate.var_names)
            rhs = ", ".join(
                (f"-{v}" if s == -1 else v) for v, s in zip(substrate.var_names, g_signs)
            )
            discovered.append(
                DiscreteSymmetry(
                    signs=g_signs,
                    equivariance_error=max_excess,
                    description=f"Z2_reflection: ({lhs}) -> ({rhs})",
                )
            )

    discovered.sort(key=lambda s: s.equivariance_error)
    return discovered
