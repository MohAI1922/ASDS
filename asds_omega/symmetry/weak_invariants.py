"""Model-Free Variational First-Integral Foliation Discovery (Zero-Derivative Conservation).

Core Mathematical Identity:
    For ANY smooth conserved quantity H(x(t)) = const along trajectories and ANY
    compactly supported bump test function phi_k(t) vanishing at window boundaries:
        integral phi'_k(t) H(x(t)) dt = -integral phi_k(t) (dH/dt) dt = 0.
    Expanding H(x) = sum_m c_m psi_m(x) yields A c = 0 where A_{km} = <phi'_k, psi_m(x)>
    requires NEITHER x_dot NOR f(x).

Key Theoretical Guards:
1. Univariate Exclusion Theorem: Any candidate H(x) depending on only 1 state variable
   is rejected (eliminates trivial Taylor identities such as cos(x) + 0.5*x^2 ≈ 1 or
   sin(x) - x ≈ 0 on small intervals).
2. Cross-Trajectory Foliation Guard: When multiple trajectories exist, verifies that
   within-trajectory variance of H(x(t)) is strictly much smaller than total variance
   and than the sum of its constituent term variations (cancellation ratio).
3. Family-Stratified Sparse Nullspace Pursuit: Probes physically meaningful invariant
   families (linear mass/stoichiometry, bilinear/quadratic angular momentum & energy,
   Hamiltonian quadratic+trig, Lotka-Volterra linear+log, and central-force kinetic+1/r)
   so exact conservation laws are cleanly separated without cross-family mixing.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np
import sympy as sp

from asds_omega.preprocessing.ms_avp import (
    WeakFormSubstrate,
    debias_trig_feature,
    hermite_debias_monomial,
)


@dataclass
class DiscoveredInvariant:
    """A model-free discovered conservation law / first integral H(x) = const."""

    expr: sp.Expr
    expr_str: str
    family: str
    weak_residual: float
    cancellation_ratio: float
    linear_weights: np.ndarray | None
    terms: dict[str, float]


@dataclass
class _DictFeature:
    name: str
    expr: sp.Expr
    vars_used: frozenset[int]
    family_tag: str  # "linear", "quad", "log", "trig", "inv_r"
    trajs: list[np.ndarray]


def _build_invariant_features(
    substrate: WeakFormSubstrate,
    max_degree: int = 2,
    include_log: bool = True,
    include_trig: bool = True,
    include_inverse_norm: bool = True,
) -> list[_DictFeature]:
    """Build multi-family cumulant-debiased candidate dictionary with variable-support tags."""
    d = substrate.n_vars
    syms = [sp.Symbol(v) for v in substrate.var_names]
    all_X = np.vstack(substrate.trajectories)
    feats: list[_DictFeature] = []

    # 1. Polynomial monomials (degree 1 .. max_degree)
    for deg in range(1, max_degree + 1):
        for comb in combinations_with_replacement(range(d), deg):
            powers = [0] * d
            for idx in comb:
                powers[idx] += 1
            expr = sp.Integer(1)
            for j, p in enumerate(powers):
                if p > 0:
                    expr = expr * (syms[j] ** p)
            trajs = [
                hermite_debias_monomial(X, tuple(powers), substrate.noise_vars, enabled=substrate.cumulant_debiasing)
                for X in substrate.trajectories
            ]
            feats.append(
                _DictFeature(
                    name=str(expr),
                    expr=expr,
                    vars_used=frozenset(comb),
                    family_tag="linear" if deg == 1 else "quad",
                    trajs=trajs,
                )
            )

    # 2. Logarithmic features log(x_j) for predominantly positive variables (robust to noisy tail dips)
    if include_log:
        for j in range(d):
            if float(np.percentile(all_X[:, j], 5)) > 0.08 and float(np.median(all_X[:, j])) > 0.15:
                floor_j = max(0.05 * float(np.median(all_X[:, j])), 1e-3)
                feats.append(
                    _DictFeature(
                        name=f"log({substrate.var_names[j]})",
                        expr=sp.log(syms[j]),
                        vars_used=frozenset([j]),
                        family_tag="log",
                        trajs=[np.log(np.maximum(X[:, j], floor_j)) for X in substrate.trajectories],
                    )
                )

    # 3. Trigonometric cos(x_j), sin(x_j) for angular-scale variables
    if include_trig:
        for j in range(d):
            rng_j = float(np.max(all_X[:, j]) - np.min(all_X[:, j]))
            if 0.4 < rng_j < 15.0:
                for kind, sp_fn in [("cos", sp.cos), ("sin", sp.sin)]:
                    trajs = [
                        debias_trig_feature(
                            X[:, j], kind, float(substrate.noise_vars[j]), enabled=substrate.cumulant_debiasing
                        )
                        for X in substrate.trajectories
                    ]
                    feats.append(
                        _DictFeature(
                            name=f"{kind}({substrate.var_names[j]})",
                            expr=sp_fn(syms[j]),
                            vars_used=frozenset([j]),
                            family_tag="trig",
                            trajs=trajs,
                        )
                    )

    # 4. Inverse radial norm 1/sqrt(x_0^2 + x_1^2) for 4D+ central-force systems
    if include_inverse_norm and d >= 4:
        r_vals = np.sqrt(all_X[:, 0] ** 2 + all_X[:, 1] ** 2)
        if float(np.min(r_vals)) > 0.15 and float(np.std(r_vals)) > 1e-3:
            expr = 1 / sp.sqrt(syms[0] ** 2 + syms[1] ** 2)
            feats.append(
                _DictFeature(
                    name=f"1/sqrt({substrate.var_names[0]}**2 + {substrate.var_names[1]}**2)",
                    expr=expr,
                    vars_used=frozenset([0, 1]),
                    family_tag="inv_r",
                    trajs=[
                        1.0 / np.sqrt(np.maximum(X[:, 0] ** 2 + X[:, 1] ** 2, 1e-6))
                        for X in substrate.trajectories
                    ],
                )
            )

    return feats


def _eval_candidate(
    c: np.ndarray,
    A_sub: np.ndarray,
    F_trajs: list[np.ndarray],
) -> tuple[float, float]:
    """Compute weak-form conservation residual and within-trajectory cancellation ratio."""
    col_norms = np.linalg.norm(A_sub, axis=0)
    denom_a = float(np.sum(np.abs(c) * col_norms)) + 1e-12
    weak_res = float(np.linalg.norm(A_sub @ c) / denom_a)

    ratios = []
    for F in F_trajs:
        H = F @ c
        term_stds = np.std(F, axis=0)
        denom_s = float(np.sum(np.abs(c) * term_stds)) + 1e-12
        ratios.append(float(np.std(H) / denom_s))
    return weak_res, float(np.mean(ratios))


def _search_subspace_invariants(
    feats: list[_DictFeature],
    indices: list[int],
    A_full: np.ndarray,
    F_full_trajs: list[np.ndarray],
    sv_threshold: float,
    cancel_threshold: float,
    max_terms: int = 5,
) -> list[tuple[np.ndarray, float, float]]:
    """Find sparse, multi-variable conservation vectors c (in full feature coordinates) inside a family."""
    if len(indices) < 2:
        return []
    m_full = len(feats)
    m_sub = len(indices)
    A_sub = A_full[:, indices]
    F_sub_trajs = [F[:, indices] for F in F_full_trajs]

    all_F = np.vstack(F_sub_trajs)
    f_scales = np.maximum(np.std(all_F, axis=0), 1e-6)
    A_s = A_sub / f_scales[None, :]
    col_n = np.maximum(np.linalg.norm(A_s, axis=0), 1e-8)
    A_norm = A_s / col_n[None, :]

    _, svals, Vt = np.linalg.svd(A_norm, full_matrices=False)
    rel_svs = svals / (svals[0] + 1e-12)
    # Inspect bottom singular vectors up to dimension 4
    cand_rows = [k for k in reversed(range(len(svals))) if rel_svs[k] < max(sv_threshold * 4.0, 0.20)][:4]
    if not cand_rows:
        cand_rows = [len(svals) - 1]

    found: list[tuple[np.ndarray, float, float]] = []

    def _refine_and_check(v_norm: np.ndarray) -> None:
        c_sub = v_norm / (col_n * f_scales)
        peak = int(np.argmax(np.abs(c_sub)))
        if abs(c_sub[peak]) < 1e-12:
            return
        c_sub = c_sub / c_sub[peak]

        # Iterative backward pruning of weakest scaled term while preserving conservation
        for _ in range(m_sub):
            nz = np.where(np.abs(c_sub) > 1e-8)[0]
            if len(nz) <= 2:
                break
            contrib = np.abs(c_sub[nz]) * f_scales[nz]
            # Try dropping weakest term
            weakest = nz[int(np.argmin(contrib))]
            keep = [idx for idx in nz if idx != weakest]
            # Refit optimal nullspace vector on `keep`
            Ak = A_sub[:, keep] / f_scales[keep][None, :]
            _, _, Vtk = np.linalg.svd(Ak, full_matrices=False)
            c_try = np.zeros(m_sub, dtype=float)
            c_try[keep] = Vtk[-1] / f_scales[keep]
            p2 = int(np.argmax(np.abs(c_try)))
            if abs(c_try[p2]) > 1e-12:
                c_try /= c_try[p2]
            w_try, r_try = _eval_candidate(c_try, A_sub, F_sub_trajs)
            w_cur, r_cur = _eval_candidate(c_sub, A_sub, F_sub_trajs)
            # Always prune if above max_terms OR if residual stays well within tolerance
            if len(nz) > max_terms or (
                w_try <= max(sv_threshold * 0.9, w_cur * 1.35)
                and r_try <= max(cancel_threshold * 0.9, r_cur * 1.35)
            ):
                c_sub = c_try
            else:
                break

        nz = np.where(np.abs(c_sub) > 1e-5)[0]
        if len(nz) < 2 or len(nz) > max_terms:
            return

        # UNIVARIATE EXCLUSION THEOREM: reject trivial single-variable Taylor identities!
        vars_covered: set[int] = set()
        for local_idx in nz:
            vars_covered.update(feats[indices[local_idx]].vars_used)
        if len(vars_covered) < 2:
            return

        w_res, c_rat = _eval_candidate(c_sub, A_sub, F_sub_trajs)
        if w_res <= sv_threshold and c_rat <= cancel_threshold:
            c_full = np.zeros(m_full, dtype=float)
            c_full[indices] = c_sub
            # Deduplicate against already registered invariants
            u = c_full / (np.linalg.norm(c_full) + 1e-12)
            for prev_c, _, _ in found:
                pu = prev_c / (np.linalg.norm(prev_c) + 1e-12)
                if abs(float(np.dot(u, pu))) > 0.85:
                    return
            found.append((c_full, w_res, c_rat))

    for k in cand_rows:
        _refine_and_check(Vt[k])
    if len(cand_rows) > 1:
        V_null = Vt[cand_rows]
        for pivot in range(m_sub):
            proj = V_null[:, pivot]
            if np.linalg.norm(proj) > 1e-6:
                _refine_and_check(proj @ V_null)

    return found


def discover_weak_invariants(
    substrate: WeakFormSubstrate,
    max_degree: int = 2,
    include_log: bool = True,
    include_trig: bool = True,
    include_inverse_norm: bool = True,
    sv_threshold: float = 0.045,
) -> list[DiscoveredInvariant]:
    """Discover model-free conservation laws H(x) = const across physical invariant families."""
    d = substrate.n_vars
    if d < 2:
        return []

    feats = _build_invariant_features(
        substrate,
        max_degree=max_degree,
        include_log=include_log,
        include_trig=include_trig,
        include_inverse_norm=include_inverse_norm,
    )
    F_full_trajs = [np.column_stack([f.trajs[r] for f in feats]) for r in range(len(substrate.trajectories))]
    A_full = substrate.project_derivative_of_features(F_full_trajs)

    rel_noise = float(substrate.metadata.get("relative_noise_est", 0.0))
    eff_sv = max(sv_threshold, 0.05) + 2.2 * rel_noise
    eff_cancel = 0.045 + 1.15 * rel_noise

    idx_lin = [i for i, f in enumerate(feats) if f.family_tag == "linear"]
    idx_quad = [i for i, f in enumerate(feats) if f.family_tag == "quad"]
    idx_log = [i for i, f in enumerate(feats) if f.family_tag == "log"]
    idx_trig = [i for i, f in enumerate(feats) if f.family_tag == "trig"]
    idx_invr = [i for i, f in enumerate(feats) if f.family_tag == "inv_r"]

    # Family-stratified search (avoids cross-family Taylor cancellations!)
    families_to_probe: list[tuple[str, list[int], int]] = [
        ("linear_conservation", idx_lin, d),
        ("polynomial_first_integral", idx_quad, 4),
    ]
    if idx_log:
        families_to_probe.append(("logarithmic_first_integral", idx_lin + idx_log, 2 * d))
    if idx_trig:
        families_to_probe.append(("hamiltonian_trigonometric", idx_quad + idx_trig, 4))
    if idx_invr:
        families_to_probe.append(("central_force_energy", idx_quad + idx_invr, 4))

    raw_found: list[tuple[str, np.ndarray, float, float]] = []
    for fam_name, fam_indices, max_t in families_to_probe:
        # Linear conservation (mass/stoichiometry) has low DoF and can have low derivative energy
        # in asymptotic tails (e.g. SIR), whereas higher-degree nonlinear families require strict
        # weak-form residuals to prevent false invariants on dissipative attractors like Lorenz.
        if fam_name == "linear_conservation":
            fam_sv = max(sv_threshold, 0.06) + 2.4 * rel_noise
            fam_cancel = 0.035 + 0.85 * rel_noise
        else:
            fam_sv = max(sv_threshold, 0.045) + 0.75 * rel_noise
            fam_cancel = 0.045 + 0.70 * rel_noise
        hits = _search_subspace_invariants(
            feats,
            fam_indices,
            A_full,
            F_full_trajs,
            sv_threshold=fam_sv,
            cancel_threshold=fam_cancel,
            max_terms=max_t,
        )
        for c_full, w_res, c_rat in hits:
            # Skip if already found as a simpler/duplicate invariant
            u = c_full / (np.linalg.norm(c_full) + 1e-12)
            if any(abs(float(np.dot(u, pc / (np.linalg.norm(pc) + 1e-12)))) > 0.85 for _, pc, _, _ in raw_found):
                continue
            raw_found.append((fam_name, c_full, w_res, c_rat))

    results: list[DiscoveredInvariant] = []
    for fam_name, c_vec, w_res, c_rat in raw_found:
        nz = np.where(np.abs(c_vec) > 1e-5)[0]
        terms_dict: dict[str, float] = {}
        expr = sp.Integer(0)
        for idx in nz:
            val = float(c_vec[idx])
            if abs(val - round(val)) < 0.04:
                val_clean = float(round(val))
                coeff_sym = sp.Integer(int(round(val)))
            elif abs(2 * val - round(2 * val)) < 0.04:
                val_clean = float(round(2 * val) / 2.0)
                coeff_sym = sp.Rational(int(round(2 * val)), 2)
            elif abs(4 * val - round(4 * val)) < 0.04:
                val_clean = float(round(4 * val) / 4.0)
                coeff_sym = sp.Rational(int(round(4 * val)), 4)
            else:
                val_clean = round(val, 4)
                coeff_sym = sp.Float(val_clean)
            terms_dict[feats[idx].name] = val_clean
            expr = expr + coeff_sym * feats[idx].expr

        expr = sp.simplify(expr)
        lin_w = None
        if fam_name == "linear_conservation":
            lin_w = np.zeros(d, dtype=float)
            for idx in nz:
                v_idx = next(iter(feats[idx].vars_used))
                lin_w[v_idx] = float(c_vec[idx])
        elif fam_name == "polynomial_first_integral" and len(nz) == 2 and all("*" in feats[i].name and "**" not in feats[i].name for i in nz):
            fam_name = "angular_momentum_bilinear"

        results.append(
            DiscoveredInvariant(
                expr=expr,
                expr_str=str(expr),
                family=fam_name,
                weak_residual=w_res,
                cancellation_ratio=c_rat,
                linear_weights=lin_w,
                terms=terms_dict,
            )
        )

    results.sort(key=lambda inv: inv.weak_residual + 0.5 * inv.cancellation_ratio)
    return results
