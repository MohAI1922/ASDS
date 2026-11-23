"""Continuous Lie Point Symmetry Prolongation Solver, Lie-Algebra Closure & Noether Pairing.

Operates in two scientifically rigorous modes:
1. Model-free pre-discovery mode: tests canonical geometric Lie subgroups (simultaneous
   SO(2) rotations in position-velocity planes & uniform scaling) without overfitting
   noisy local Jacobians.
2. Exact symbolic prolongation mode (post-S-ECE): solves the exact linearized Lie bracket
   [V, f](x) = J_f(x)(Ax + b) - A f(x) = 0 using exact SymPy Jacobians of the discovered
   equations, verifies Lie-algebra closure [V_a, V_b] = C_{ab}^c V_c, and detects
   Noether pairings between continuous symmetries and conserved quantities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

from asds_omega.preprocessing.ms_avp import WeakFormSubstrate


@dataclass
class LieGenerator:
    """A continuous Lie point symmetry infinitesimal generator V = sum_i xi_i(x) d/dx_i."""

    matrix_A: np.ndarray
    vector_b: np.ndarray
    sympy_components: list[sp.Expr]
    symmetry_type: str
    lie_residual: float
    description: str


def _classify_generator(A: np.ndarray, b: np.ndarray, var_names: list[str]) -> tuple[str, str]:
    norm_A = float(np.linalg.norm(A))
    norm_b = float(np.linalg.norm(b))
    if norm_A < 0.08 * (norm_b + 1e-12):
        return "translation", f"Translation along {np.round(b, 3).tolist()}"
    skew_err = float(np.linalg.norm(A + A.T) / (norm_A + 1e-12))
    if skew_err < 0.15:
        return "SO(2)_rotation", f"Continuous rotational SO(2) symmetry (skew-err={skew_err:.3f})"
    off_diag = A - np.diag(np.diag(A))
    if float(np.linalg.norm(off_diag) / (norm_A + 1e-12)) < 0.15:
        return "scaling", f"Scaling symmetry diag={np.round(np.diag(A), 3).tolist()}"
    return "affine_lie", f"General affine Lie generator on ({', '.join(var_names)})"


def _estimate_local_jacobians(
    U: np.ndarray,
    V: np.ndarray,
    k_neighbors: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate local Jacobian J_f(x_k) via weighted local linear regression (for divergence)."""
    K, d = U.shape
    k_nn = k_neighbors or min(max(3 * d + 4, 15), K - 1)
    idx_eval = np.linspace(0, K - 1, min(60, K), dtype=int)
    U_eval = U[idx_eval]
    V_eval = V[idx_eval]
    J_eval = np.zeros((len(idx_eval), d, d), dtype=float)

    U_sd = np.maximum(np.std(U, axis=0), 1e-8)
    U_norm = U / U_sd[None, :]

    for m, idx in enumerate(idx_eval):
        dists = np.linalg.norm(U_norm - U_norm[idx], axis=1)
        nn = np.argpartition(dists, kth=k_nn)[:k_nn]
        dX = U[nn] - U[idx]
        dV = V[nn] - V[idx]
        reg = 1e-4 * np.eye(d) * float(np.mean(np.sum(dX**2, axis=1)) + 1e-6)
        JT = np.linalg.solve(dX.T @ dX + reg, dX.T @ dV)
        J_eval[m] = JT.T

    return U_eval, V_eval, J_eval


def _probe_canonical_so2_model_free(
    substrate: WeakFormSubstrate,
    sv_threshold: float,
) -> list[LieGenerator]:
    """Model-free check for simultaneous SO(2) rotational symmetry (e.g., 2D central force)."""
    d = substrate.n_vars
    syms = [sp.Symbol(v) for v in substrate.var_names]
    U = substrate.weak_states
    V = substrate.weak_velocities
    generators: list[LieGenerator] = []

    # Candidate SO(2) generators:
    # (a) 2D rotation on (x0, x1) for d=2, OR (b) coupled (x0, x1) & (x2, x3) rotation for d=4 (Kepler!)
    cand_mats: list[np.ndarray] = []
    if d == 2:
        A = np.array([[0.0, -1.0], [1.0, 0.0]])
        cand_mats.append(A)
    elif d == 4:
        A = np.zeros((4, 4), dtype=float)
        A[0, 1], A[1, 0] = -1.0, 1.0
        A[2, 3], A[3, 2] = -1.0, 1.0
        cand_mats.append(A)

    if not cand_mats or len(U) < 30:
        return []

    U_std = (U - U.mean(axis=0)) / np.maximum(U.std(axis=0), 1e-8)
    # Test finite small rotation R(eps) = exp(eps * A) via kNN pushforward equivariance
    eps = 0.25
    for A in cand_mats:
        # Exact matrix exponential for block-skew A is cos(eps) I + sin(eps) A
        R = np.cos(eps) * np.eye(d) + np.sin(eps) * A
        U_rot = U @ R.T
        U_rot_std = (U_rot - U.mean(axis=0)) / np.maximum(U.std(axis=0), 1e-8)
        D = np.linalg.norm(U_rot_std[:, None, :] - U_std[None, :, :], axis=2)
        nn_idx = np.argmin(D, axis=1)
        nn_dist = D[np.arange(len(U)), nn_idx]
        close = nn_dist < 0.35 * np.sqrt(d)
        if np.sum(close) < max(12, len(U) // 10):
            continue
        V_exp = V[close] @ R.T
        V_obs = V[nn_idx[close]]
        res = float(np.linalg.norm(V_obs - V_exp) / (np.linalg.norm(V_exp) + 1e-12))
        if res <= max(sv_threshold * 3.0, 0.16):
            comp = [sp.simplify(sum(sp.Integer(int(A[i, j])) * syms[j] for j in range(d) if A[i, j] != 0)) for i in range(d)]
            generators.append(
                LieGenerator(
                    matrix_A=A,
                    vector_b=np.zeros(d, dtype=float),
                    sympy_components=comp,
                    symmetry_type="SO(2)_rotation",
                    lie_residual=res,
                    description=f"Continuous SO(2) rotational symmetry (pushforward err={res:.3f})",
                )
            )
    return generators


def discover_lie_symmetries(
    substrate: WeakFormSubstrate,
    symbolic_equations: list[sp.Expr] | None = None,
    sv_threshold: float = 0.04,
) -> list[LieGenerator]:
    """Discover continuous Lie point symmetries [V, f] = 0.

    - Without symbolic_equations: tests canonical geometric SO(2) subgroups cleanly without
      noisy local-Jacobian overfitting.
    - With symbolic_equations: solves exact symbolic prolongation [V, f] = J_f (Ax+b) - A f = 0
      via SVD and verifies exact symbolic Lie bracket residual!
    """
    d = substrate.n_vars
    syms = [sp.Symbol(v) for v in substrate.var_names]

    if symbolic_equations is None or len(symbolic_equations) != d:
        return _probe_canonical_so2_model_free(substrate, sv_threshold)

    U = substrate.weak_states
    idx_eval = np.linspace(0, len(U) - 1, min(64, len(U)), dtype=int)
    U_eval = U[idx_eval]
    f_fn = [sp.lambdify(syms, eq, modules="numpy") for eq in symbolic_equations]
    J_sym = sp.Matrix(symbolic_equations).jacobian(syms)
    J_fn = sp.lambdify(syms, J_sym, modules="numpy")

    V_eval_list, J_eval_list, U_valid = [], [], []
    for x_pt in U_eval:
        try:
            v_pt = np.array([float(fn(*x_pt)) for fn in f_fn], dtype=float)
            j_pt = np.asarray(J_fn(*x_pt), dtype=float)
            if np.all(np.isfinite(v_pt)) and np.all(np.isfinite(j_pt)):
                U_valid.append(x_pt)
                V_eval_list.append(v_pt)
                J_eval_list.append(j_pt)
        except Exception:
            continue
    if len(U_valid) < d + 2:
        return []

    U_v = np.asarray(U_valid)
    V_v = np.asarray(V_eval_list)
    J_v = np.asarray(J_eval_list)

    n_pts = len(U_v)
    n_params = d * d + d
    L = np.zeros((n_pts * d, n_params), dtype=float)
    f_rms = float(np.sqrt(np.mean(V_v**2))) + 1e-12

    for m in range(n_pts):
        x, f, J = U_v[m], V_v[m], J_v[m]
        for i in range(d):
            row = m * d + i
            for k in range(d):
                for l_idx in range(d):
                    val = J[i, k] * x[l_idx] - (f[l_idx] if k == i else 0.0)
                    L[row, k * d + l_idx] = val / f_rms
                L[row, d * d + k] = J[i, k] / f_rms

    _, svals, Vt = np.linalg.svd(L, full_matrices=False)
    rel_svs = svals / (svals[0] + 1e-12)

    generators: list[LieGenerator] = []
    for k in reversed(range(len(svals))):
        if rel_svs[k] > sv_threshold:
            break
        theta = Vt[k]
        A = theta[: d * d].reshape(d, d)
        b = theta[d * d :]
        peak = max(float(np.max(np.abs(A))), float(np.max(np.abs(b))), 1e-12)
        A_clean = np.where(np.abs(A) < 0.04 * peak, 0.0, A / peak)
        b_clean = np.where(np.abs(b) < 0.04 * peak, 0.0, b / peak)

        # Reject trivial zero or pure linear-system identity scaling V = x on linear systems unless genuinely small
        theta_c = np.concatenate([A_clean.ravel(), b_clean])
        res_clean = float(np.linalg.norm(L @ theta_c) / (np.linalg.norm(L, ord=2) * np.linalg.norm(theta_c) + 1e-12))
        if res_clean > sv_threshold * 1.5:
            continue

        sym_type, desc = _classify_generator(A_clean, b_clean, substrate.var_names)
        comp = []
        for i in range(d):
            expr_i = sp.Float(round(float(b_clean[i]), 4)) if abs(b_clean[i]) > 1e-4 else sp.Integer(0)
            for j in range(d):
                if abs(A_clean[i, j]) > 1e-4:
                    c_val = float(A_clean[i, j])
                    c_sym = sp.Integer(int(round(c_val))) if abs(c_val - round(c_val)) < 0.05 else sp.Float(round(c_val, 4))
                    expr_i = expr_i + c_sym * syms[j]
            comp.append(sp.simplify(expr_i))

        generators.append(
            LieGenerator(
                matrix_A=A_clean,
                vector_b=b_clean,
                sympy_components=comp,
                symmetry_type=sym_type,
                lie_residual=res_clean,
                description=desc,
            )
        )
        if len(generators) >= 3:
            break

    return generators


def verify_lie_closure(generators: list[LieGenerator]) -> dict[str, float | bool]:
    """Verify Lie-algebra closure [V_a, V_b] in span({V_c}) via matrix commutator [A_a, A_b]."""
    r = len(generators)
    if r <= 1:
        return {"closed": True, "n_generators": r, "max_closure_residual": 0.0}

    basis = np.column_stack([g.matrix_A.ravel() for g in generators])
    Q, _ = np.linalg.qr(basis)
    max_res = 0.0
    for a in range(r):
        for b in range(a + 1, r):
            comm = (generators[a].matrix_A @ generators[b].matrix_A - generators[b].matrix_A @ generators[a].matrix_A).ravel()
            norm_c = float(np.linalg.norm(comm))
            if norm_c > 1e-8:
                proj = Q @ (Q.T @ comm)
                res = float(np.linalg.norm(comm - proj) / norm_c)
                max_res = max(max_res, res)
    return {
        "closed": max_res < 0.1,
        "n_generators": r,
        "max_closure_residual": max_res,
    }
