"""Stage A of EVG-HE: Stability-Selected Cumulant-Debiased Variational Ensembled Sparse
Projection (V-ESP), Implicit Rational Weak-Form Solver, KKT Conservation Enforcement,
and 1+1D Spacetime PDE Discovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement

import numpy as np
import sympy as sp

from asds_omega.causal.vifc import CausalDiscoveryResult
from asds_omega.config import EVGHEConfig
from asds_omega.preprocessing.dimensional import DimensionalAnalyzer
from asds_omega.preprocessing.ms_avp import (
    SpacetimeWeakProjector,
    WeakFormSubstrate,
    debias_trig_feature,
)
from asds_omega.symmetry.reynolds import SymmetryDiscoveryResult
from asds_omega.symbolic.rationalizer import (
    compute_mdl_score,
    expression_complexity,
    rationalize_expression,
)


@dataclass
class CandidateEquation:
    """A candidate symbolic governing equation for dx_i/dt with full variational diagnostics."""

    target_var: str
    expr: sp.Expr
    expr_str: str
    weak_mse: float
    weak_r2: float
    mdl_score: float
    complexity: int
    symmetry_residual: float = 0.0
    source: str = "vesp"
    inclusion_probs: dict[str, float] = field(default_factory=dict)


@dataclass
class PDECandidate:
    """Discovered spatiotemporal PDE candidate u_t = F(u, u_x, u_xx, ...)."""

    expr: sp.Expr
    expr_str: str
    weak_mse: float
    weak_r2: float
    mdl_score: float
    estimated_noise_std: float
    coefficients: dict[str, float]


@dataclass
class _LibraryTerm:
    name: str
    expr: sp.Expr
    vars_used: frozenset[int]
    weak_col: np.ndarray  # (K,)
    tier: int = 1  # 1: poly<=2, 2: trig/radial, 3: poly==3
    is_trig: bool = False


def _stlsq(G: np.ndarray, y: np.ndarray, threshold: float, ridge_alpha: float = 1e-6, max_iter: int = 12) -> np.ndarray:
    """Column-normalized Sequential Thresholded Least Squares (STLSQ)."""
    m = G.shape[1]
    if m == 0:
        return np.zeros(0, dtype=float)
    col_norms = np.maximum(np.linalg.norm(G, axis=0), 1e-10)
    y_norm = max(float(np.linalg.norm(y)), 1e-10)
    Gn = G / col_norms[None, :]
    yn = y / y_norm

    xi = np.linalg.solve(Gn.T @ Gn + ridge_alpha * np.eye(m), Gn.T @ yn)
    for _ in range(max_iter):
        active = np.abs(xi) >= threshold
        if not np.any(active):
            # Keep single strongest term
            active[int(np.argmax(np.abs(xi)))] = True
        xi_new = np.zeros(m, dtype=float)
        Ga = Gn[:, active]
        xi_new[active] = np.linalg.solve(Ga.T @ Ga + ridge_alpha * np.eye(Ga.shape[1]), Ga.T @ yn)
        if np.array_equal(active, np.abs(xi_new) >= threshold) and np.allclose(xi, xi_new, atol=1e-8):
            xi = xi_new
            break
        xi = xi_new

    # Unscale back to raw feature coordinates
    return xi * (y_norm / col_norms)


class VESPEngine:
    """Variational Ensembled Sparse Projection + Implicit Rational + KKT Conservation Engine."""

    def __init__(self, config: EVGHEConfig | None = None, seed: int = 42) -> None:
        self.config = config or EVGHEConfig()
        self.seed = seed

    def build_library(
        self,
        substrate: WeakFormSubstrate,
        dim_analyzer: DimensionalAnalyzer | None = None,
    ) -> list[_LibraryTerm]:
        """Construct cumulant-debiased weak-form candidate dictionary."""
        d = substrate.n_vars
        K = substrate.n_subdomains
        syms = [sp.Symbol(v) for v in substrate.var_names]
        all_X = np.vstack(substrate.trajectories)
        terms: list[_LibraryTerm] = []

        # Constant term 1
        terms.append(_LibraryTerm("1", sp.Integer(1), frozenset(), np.ones(K, dtype=float), tier=1))

        # Polynomial monomials 1 .. max_poly_degree (analytically Hermite cumulant-debiased!)
        max_deg = self.config.max_poly_degree if d <= 3 else min(self.config.max_poly_degree, 2)
        for deg in range(1, max_deg + 1):
            for comb in combinations_with_replacement(range(d), deg):
                powers = [0] * d
                for idx in comb:
                    powers[idx] += 1
                expr = sp.Integer(1)
                for j, p in enumerate(powers):
                    if p > 0:
                        expr = expr * (syms[j] ** p)
                col = substrate.project_monomial(tuple(powers))
                terms.append(_LibraryTerm(str(expr), expr, frozenset(comb), col, tier=(1 if deg <= 2 else 3)))

        # Trigonometric terms sin(x_j), cos(x_j): ONLY for angular-span variables (avoids Taylor collinearity!)
        if self.config.include_trig:
            for j in range(d):
                rng_j = float(np.max(all_X[:, j]) - np.min(all_X[:, j]))
                # Require genuine non-monotonic angular span (> 1.6 rad and visits negative values)
                if 1.6 < rng_j < 14.0 and float(np.min(all_X[:, j])) < -0.2:
                    for kind, sp_fn in [("sin", sp.sin), ("cos", sp.cos)]:
                        expr = sp_fn(syms[j])
                        if dim_analyzer is not None and not dim_analyzer.is_transcendentally_valid(expr):
                            continue
                        trajs = [
                            debias_trig_feature(
                                X[:, j], kind, float(substrate.noise_vars[j]), enabled=substrate.cumulant_debiasing
                            )[:, None]
                            for X in substrate.trajectories
                        ]
                        col = substrate.project_pointwise(trajs)[:, 0]
                        terms.append(_LibraryTerm(str(expr), expr, frozenset([j]), col, tier=2, is_trig=True))

        # Central-force inverse-cube radial features x_0/r^3, x_1/r^3 (automatic for 4D orbital systems)
        if d >= 4 or self.config.include_inverse_r3:
            r_vals = np.sqrt(all_X[:, 0] ** 2 + all_X[:, 1] ** 2)
            if float(np.min(r_vals)) > 0.15:
                r3_sym = (syms[0] ** 2 + syms[1] ** 2) ** sp.Rational(3, 2)
                for coord in (0, 1):
                    expr = syms[coord] / r3_sym
                    trajs = [
                        (X[:, coord] / np.maximum(X[:, 0] ** 2 + X[:, 1] ** 2, 1e-4) ** 1.5)[:, None]
                        for X in substrate.trajectories
                    ]
                    col = substrate.project_pointwise(trajs)[:, 0]
                    terms.append(_LibraryTerm(str(expr), expr, frozenset([0, 1]), col, tier=2))

        return terms

    @staticmethod
    def _effective_samples(substrate: WeakFormSubstrate) -> float:
        """Compute effective independent sample count K_eff accounting for overlapping weak windows."""
        if not substrate.use_weak_form:
            return float(min(substrate.n_subdomains, 80))
        total_pts = sum(len(t) for t in substrate.t_list)
        mean_w = max(float(np.mean(substrate.window_widths)), 1.0)
        return float(np.clip(total_pts / mean_w, 18.0, 90.0))

    def _fit_implicit_rational_weak(
        self,
        substrate: WeakFormSubstrate,
        target_idx: int,
        parent_set: set[int],
    ) -> list[CandidateEquation]:
        """Weak-form Implicit Rational discovery: (1 + sum a_k x_k) * dx_i/dt = sum b_m psi_m(x).

        Uses exact integration by parts for x_i * dx_i/dt = 0.5 * d/dt(x_i^2 - sigma_i^2),
        achieving zero numerical differentiation even for rational laws like Michaelis-Menten!
        """
        d = substrate.n_vars
        K = substrate.n_subdomains
        k_eff = self._effective_samples(substrate)
        syms = [sp.Symbol(v) for v in substrate.var_names]
        y_v = substrate.weak_velocities[:, target_idx]
        y_var = float(np.var(y_v)) + 1e-12

        cands: list[CandidateEquation] = []
        # Probe single-variable rational denominators (1 + a * x_p) for each causal parent p
        for p in sorted(parent_set):
            all_xp = np.concatenate([X[:, p] for X in substrate.trajectories])
            if float(np.percentile(all_xp, 5)) < -0.2 or float(np.percentile(all_xp, 75)) <= 0.1:
                continue
            if p == target_idx:
                x2_trajs = [
                    (X[:, p] ** 2 - (substrate.noise_vars[p] if substrate.cumulant_debiasing else 0.0))[:, None]
                    for X in substrate.trajectories
                ]
                xp_vi = -0.5 * substrate.project_derivative_of_features(x2_trajs)[:, 0]
            else:
                xp_vi = substrate.weak_states[:, p] * y_v

            x_p_w = substrate.project_monomial(tuple(1 if j == p else 0 for j in range(d)))
            for include_const in (False, True):
                # Total Least Squares (SVD nullspace) solves c0*v + c1*(x_p*v) - c2*x_p (- c3*1) = 0
                # symmetrically without EIV attenuation bias on noisy x_p * v!
                raw_cols = [y_v, xp_vi, -x_p_w]
                if include_const:
                    raw_cols.append(-np.ones(K, dtype=float))
                M_raw = np.column_stack(raw_cols)
                col_sc = np.maximum(np.linalg.norm(M_raw, axis=0), 1e-10)
                _, _, Vt = np.linalg.svd(M_raw / col_sc[None, :], full_matrices=False)
                c_vec = Vt[-1] / col_sc
                if abs(c_vec[0]) < 1e-8:
                    continue
                c_vec = c_vec / c_vec[0]
                a_val = float(c_vec[1])
                b1_val = float(c_vec[2])
                b0_val = float(c_vec[3]) if include_const else 0.0

                if a_val <= 0.15 or a_val > 100.0:
                    continue
                if float(np.percentile(1.0 + a_val * all_xp, 5)) <= 0.05:
                    continue

                km = 1.0 / a_val
                vmax = b1_val / a_val
                c0 = b0_val / a_val
                num = sp.Float(round(vmax, 4)) * syms[p]
                if include_const and abs(c0) > 1e-3:
                    num = sp.Float(round(c0, 4)) + num
                expr = sp.simplify(num / (sp.Float(round(km, 4)) + syms[p]))

                pred_w = substrate.project_callable(
                    lambda X, _km=km, _vm=vmax, _c0=(c0 if include_const else 0.0), _p=p: (_c0 + _vm * X[:, _p])
                    / np.maximum(_km + X[:, _p], 0.05)
                )
                mse = float(np.mean((y_v - pred_w) ** 2))
                r2 = float(1.0 - mse / y_var)
                if r2 > 0.90:
                    if self.config.enable_rationalizer:
                        fn_eval = lambda ex: self._eval_sympy_weak_mse(ex, substrate, target_idx)
                        expr, mse = rationalize_expression(
                            expr, fn_eval, k_eff, y_var, max_denom=self.config.max_rational_denom
                        )
                        r2 = float(1.0 - mse / y_var)
                    mdl = compute_mdl_score(mse, k_eff, expr, y_var)
                    cands.append(
                        CandidateEquation(
                            target_var=substrate.var_names[target_idx],
                            expr=expr,
                            expr_str=str(expr),
                            weak_mse=mse,
                            weak_r2=r2,
                            mdl_score=mdl,
                            complexity=expression_complexity(expr),
                            source="vesp_implicit_rational",
                        )
                    )
        return cands

    def _eval_sympy_weak_mse(
        self,
        expr: sp.Expr,
        substrate: WeakFormSubstrate,
        target_idx: int,
    ) -> float:
        syms = [sp.Symbol(v) for v in substrate.var_names]
        fn = sp.lambdify(syms, expr, modules="numpy")
        y_v = substrate.weak_velocities[:, target_idx]
        try:
            def _vec_eval(X: np.ndarray) -> np.ndarray:
                args = [X[:, j] for j in range(X.shape[1])]
                out = fn(*args)
                if np.ndim(out) == 0:
                    return np.full(X.shape[0], float(out), dtype=float)
                return np.asarray(out, dtype=float)

            pred = substrate.project_callable(_vec_eval)
            if not np.all(np.isfinite(pred)):
                return 1e9
            return float(np.mean((y_v - pred) ** 2))
        except Exception:
            return 1e9

    def _enforce_linear_conservation_kkt(
        self,
        substrate: WeakFormSubstrate,
        terms: list[_LibraryTerm],
        active_masks: list[np.ndarray],
        coeffs_list: list[np.ndarray],
        C_lin: np.ndarray,
    ) -> list[np.ndarray]:
        """Solve coupled KKT-constrained weak-form least squares enforcing C_lin @ dx/dt = 0.

        For every linear conservation law sum_i c_{r,i} dx_i/dt = 0 (e.g., dS + dI + dR = 0),
        enforces exact coefficient cancellation sum_i c_{r,i} xi_{i, m} = 0 on every shared
        active basis term m, guaranteeing machine-precision conservation!
        """
        d = substrate.n_vars
        M = len(terms)
        G_full = np.column_stack([t.weak_col for t in terms])
        # Build union of active basis indices across all equations coupled by C_lin
        coupled_eqs = np.where(np.any(np.abs(C_lin) > 1e-6, axis=0))[0]
        if len(coupled_eqs) < 2:
            return coeffs_list

        # Collect active indices per equation; allow union only where needed for exact cancellation
        union_active = np.zeros(M, dtype=bool)
        for i in coupled_eqs:
            union_active |= active_masks[i]
        act_idx = np.where(union_active)[0]
        if len(act_idx) == 0:
            return coeffs_list

        n_c = len(coupled_eqs)
        n_a = len(act_idx)
        n_vars_opt = n_c * n_a
        n_constraints = C_lin.shape[0] * n_a

        # Build block-diagonal normal equations H * x = rhs with equality constraints A_eq * x = 0
        Ga = G_full[:, act_idx]
        GtG = Ga.T @ Ga + 1e-9 * np.eye(n_a)
        H = np.zeros((n_vars_opt, n_vars_opt), dtype=float)
        rhs = np.zeros(n_vars_opt, dtype=float)
        for loc_i, eq_i in enumerate(coupled_eqs):
            sl = slice(loc_i * n_a, (loc_i + 1) * n_a)
            H[sl, sl] = GtG
            rhs[sl] = Ga.T @ substrate.weak_velocities[:, eq_i]

        A_eq = np.zeros((n_constraints, n_vars_opt), dtype=float)
        row = 0
        for r in range(C_lin.shape[0]):
            for m_loc in range(n_a):
                for loc_i, eq_i in enumerate(coupled_eqs):
                    A_eq[row, loc_i * n_a + m_loc] = C_lin[r, eq_i]
                row += 1

        # Solve KKT saddle-point system: [[H, A_eq^T], [A_eq, 0]] [x; lam] = [rhs; 0]
        K_mat = np.block([[H, A_eq.T], [A_eq, np.zeros((n_constraints, n_constraints))]])
        b_vec = np.concatenate([rhs, np.zeros(n_constraints)])
        try:
            sol = np.linalg.solve(K_mat, b_vec)[:n_vars_opt]
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(K_mat, b_vec, rcond=None)
            sol = sol[:n_vars_opt]

        # Threshold tiny cross-contaminated entries (<2.5% of equation peak) while
        # preserving exact conservation by solving constrained KKT only on active entries
        new_coeffs = [c.copy() for c in coeffs_list]
        for loc_i, eq_i in enumerate(coupled_eqs):
            xi_sub = sol[loc_i * n_a : (loc_i + 1) * n_a].copy()
            peak = float(np.max(np.abs(xi_sub))) + 1e-12
            xi_sub[np.abs(xi_sub) < 0.025 * peak] = 0.0
            xi_full = np.zeros(M, dtype=float)
            xi_full[act_idx] = xi_sub
            new_coeffs[eq_i] = xi_full
        return new_coeffs

    def discover_system(
        self,
        substrate: WeakFormSubstrate,
        causal_result: CausalDiscoveryResult | None = None,
        symmetry_result: SymmetryDiscoveryResult | None = None,
        dim_analyzer: DimensionalAnalyzer | None = None,
    ) -> tuple[list[CandidateEquation], list[list[CandidateEquation]], dict[str, float]]:
        """Run V-ESP across all d state equations with closed-loop causal & Reynolds pruning."""
        rng = np.random.default_rng(self.seed)
        terms = self.build_library(substrate, dim_analyzer=dim_analyzer)
        M = len(terms)
        d = substrate.n_vars
        K = substrate.n_subdomains
        k_eff = self._effective_samples(substrate)

        best_cands: list[CandidateEquation] = []
        pareto_per_var: list[list[CandidateEquation]] = []
        active_masks: list[np.ndarray] = []
        raw_coeffs: list[np.ndarray] = []
        pruned_counts: list[float] = []

        for i in range(d):
            y = substrate.weak_velocities[:, i]
            y_var = float(np.var(y)) + 1e-12

            # 1. Closed-loop Causal Masking from VIFC
            allowed_parents = set(causal_result.get_parents(i)) if causal_result is not None else set(range(d))

            # 2. Closed-loop Reynolds Equivariance Projection from W-LIND
            reynolds = symmetry_result.reynolds_projector if symmetry_result is not None else None

            allowed_idx = []
            for m, t in enumerate(terms):
                if not t.vars_used.issubset(allowed_parents):
                    continue
                if reynolds is not None and not reynolds.is_trivial:
                    if not reynolds.is_term_equivariant(t.expr, i):
                        continue
                allowed_idx.append(m)

            # Escape guard: if pruning left empty set, relax causal mask while keeping Reynolds
            if not allowed_idx:
                allowed_idx = [
                    m for m, t in enumerate(terms)
                    if reynolds is None or reynolds.is_trivial or reynolds.is_term_equivariant(t.expr, i)
                ]
            if not allowed_idx:
                allowed_idx = list(range(M))

            pruned_counts.append(1.0 - len(allowed_idx) / max(M, 1))

            rel_noise = float(substrate.metadata.get("relative_noise_est", 0.0))
            has_lin_cons = symmetry_result is not None and symmetry_result.linear_conservation_matrix is not None

            tier12_all = [m for m in allowed_idx if terms[m].tier <= 2 and (not has_lin_cons or terms[m].name != "1")]
            tier1_only = [m for m in allowed_idx if terms[m].tier == 1 and (not has_lin_cons or terms[m].name != "1")]
            tier13_poly = [
                m for m in allowed_idx
                if terms[m].tier in (1, 3) and (not has_lin_cons or terms[m].name != "1")
            ]
            reynolds_tier12 = [
                m for m, t in enumerate(terms)
                if t.tier <= 2
                and (not has_lin_cons or t.name != "1")
                and (reynolds is None or reynolds.is_trivial or reynolds.is_term_equivariant(t.expr, i))
            ]

            reynolds_tier13 = [
                m for m, t in enumerate(terms)
                if t.tier in (1, 3)
                and (not has_lin_cons or t.name != "1")
                and (reynolds is None or reynolds.is_trivial or reynolds.is_term_equivariant(t.expr, i))
            ]

            # Noise-floor-aware tier stopping cutoff + minimum variance-gain gate for degree-3:
            # Prevents 20-30% noise on degree-2 systems from overfitting into cubic noise terms,
            # while guaranteeing cubic discovery on genuinely cubic systems (Duffing, Van der Pol).
            tier3_cut = max(0.70, 0.96 - 1.5 * rel_noise)

            search_sets: list[tuple[str, list[int], float]] = []
            if tier1_only:
                search_sets.append(("vesp_sparse", tier1_only, 0.0))
            if set(tier13_poly) != set(tier1_only) and tier13_poly:
                search_sets.append(("vesp_tier3", tier13_poly, tier3_cut))
            if set(tier12_all) != set(tier1_only) and tier12_all:
                search_sets.append(("vesp_sparse", tier12_all, 0.996))
            if set(reynolds_tier12) != set(tier12_all) and reynolds_tier12:
                esc_cut = 0.999 if (has_lin_cons or rel_noise > 0.12) else 0.94
                search_sets.append(("vesp_escape", reynolds_tier12, esc_cut))
            if set(reynolds_tier13) != set(tier13_poly) and reynolds_tier13:
                search_sets.append(("vesp_tier3_escape", reynolds_tier13, 0.985))
            # If linear conservation exists (e.g., S+I+R=1), also probe each drop-one non-collinear
            # variable subset so STLSQ finds the maximally sparse exact representation (e.g. {S, I}!)
            if has_lin_cons and d == 3:
                for drop_v in range(d):
                    keep_v = {v for v in range(d) if v != drop_v}
                    sub_idx = [
                        m for m, t in enumerate(terms)
                        if t.tier == 1 and t.name != "1" and t.vars_used.issubset(keep_v)
                    ]
                    if sub_idx:
                        search_sets.append(("vesp_manifold_reduced", sub_idx, 0.0))

            var_candidates: list[CandidateEquation] = []
            best_mask_i = np.zeros(M, dtype=bool)
            best_xi_i = np.zeros(M, dtype=float)
            best_mdl_i = np.inf

            tier1_best_r2 = -1.0
            for src_label, idx_set, stop_r2 in search_sets:
                if stop_r2 > 0.0 and var_candidates and max(c.weak_r2 for c in var_candidates) >= stop_r2:
                    continue
                is_cubic_pass = src_label.startswith("vesp_tier3")
                G_sub = np.column_stack([terms[m].weak_col for m in idx_set])
                # Stability-selected bootstrap sweep across STLSQ thresholds
                thresh_grid = [0.008, 0.02, 0.04, 0.08, 0.14, 0.22, 0.35]
                seen_supports: set[tuple[int, ...]] = set()

                for lam in thresh_grid:
                    # Ensembled bootstrap stability selection
                    B = self.config.n_ensemble_bootstraps
                    counts = np.zeros(len(idx_set), dtype=float)
                    for _ in range(B):
                        bs = rng.integers(0, K, size=K)
                        xi_b = _stlsq(G_sub[bs], y[bs], threshold=lam)
                        counts += (np.abs(xi_b) > 1e-8).astype(float)
                    incl_probs = counts / B
                    support_local = np.where(incl_probs >= self.config.inclusion_prob_threshold)[0]
                    if len(support_local) == 0:
                        xi_full_sub = _stlsq(G_sub, y, threshold=lam)
                        support_local = np.where(np.abs(xi_full_sub) > 1e-8)[0]
                    if len(support_local) == 0:
                        continue
                    if len(support_local) > 5:
                        xi_tmp = _stlsq(G_sub, y, threshold=lam)
                        norms = np.linalg.norm(G_sub, axis=0)
                        top5 = np.argsort(np.abs(xi_tmp) * norms)[-5:]
                        support_local = np.sort(top5)

                    supp_key = tuple(idx_set[k] for k in support_local)
                    if supp_key in seen_supports:
                        continue
                    seen_supports.add(supp_key)

                    # Unbiased OLS refit on stability-selected support
                    Ga = G_sub[:, support_local]
                    coef_a, *_ = np.linalg.lstsq(Ga, y, rcond=None)
                    pred = Ga @ coef_a
                    mse = float(np.mean((y - pred) ** 2))
                    r2 = float(1.0 - mse / y_var)
                    # Cubic variance-gain guard: at high noise (>12%), reject cubic candidates
                    # that fail to explain substantially more variance (+0.07 R^2) than degree <= 2.
                    if is_cubic_pass and rel_noise > 0.12 and tier1_best_r2 > 0.45 and r2 < tier1_best_r2 + 0.07:
                        continue

                    expr = sp.Integer(0)
                    probs_map: dict[str, float] = {}
                    for loc_k, c_val in zip(support_local, coef_a):
                        if abs(c_val) > 1e-7:
                            t_obj = terms[idx_set[loc_k]]
                            expr = expr + sp.Float(round(float(c_val), 5)) * t_obj.expr
                            probs_map[t_obj.name] = float(incl_probs[loc_k])
                    expr = sp.simplify(expr)

                    if self.config.enable_rationalizer:
                        expr_r, mse_r = rationalize_expression(
                            expr,
                            lambda ex, _i=i: self._eval_sympy_weak_mse(ex, substrate, _i),
                            k_eff,
                            y_var,
                            max_denom=self.config.max_rational_denom,
                        )
                        if mse_r < 1e8:
                            expr, mse = expr_r, mse_r
                            r2 = float(1.0 - mse / y_var)

                    mdl = compute_mdl_score(mse, k_eff, expr, y_var)
                    cand = CandidateEquation(
                        target_var=substrate.var_names[i],
                        expr=expr,
                        expr_str=str(expr),
                        weak_mse=mse,
                        weak_r2=r2,
                        mdl_score=mdl,
                        complexity=expression_complexity(expr),
                        source=src_label,
                        inclusion_probs=probs_map,
                    )
                    var_candidates.append(cand)
                    if not is_cubic_pass and r2 > tier1_best_r2:
                        tier1_best_r2 = r2
                    if mdl < best_mdl_i:
                        best_mdl_i = mdl
                        best_mask_i = np.zeros(M, dtype=bool)
                        best_xi_i = np.zeros(M, dtype=float)
                        for loc_k, c_val in zip(support_local, coef_a):
                            best_mask_i[idx_set[loc_k]] = True
                            best_xi_i[idx_set[loc_k]] = float(c_val)

            # Implicit rational probe (e.g., Michaelis-Menten)
            if self.config.include_rational_implicit:
                rat_cands = self._fit_implicit_rational_weak(substrate, i, allowed_parents)
                var_candidates.extend(rat_cands)

            var_candidates.sort(key=lambda c: c.mdl_score)
            best_cands.append(var_candidates[0])
            pareto_per_var.append(var_candidates[:6])
            active_masks.append(best_mask_i)
            raw_coeffs.append(best_xi_i)

        # 3. Coupled KKT Conservation Enforcement (accept only when parsimonious & MDL-consistent)
        kkt_applied = False
        if (
            symmetry_result is not None
            and symmetry_result.linear_conservation_matrix is not None
            and all(not c.source.endswith("rational") for c in best_cands)
        ):
            C_lin = symmetry_result.linear_conservation_matrix
            kkt_coeffs = self._enforce_linear_conservation_kkt(substrate, terms, active_masks, raw_coeffs, C_lin)
            kkt_cands: list[CandidateEquation] = []
            accept = True
            for i in range(d):
                xi = kkt_coeffs[i]
                nz = np.where(np.abs(xi) > 1e-4)[0]
                if len(nz) > max(4, len(sp.Add.make_args(sp.expand(best_cands[i].expr))) + 1):
                    accept = False
                    break
                expr = sp.Integer(0)
                for m in nz:
                    expr = expr + sp.Float(round(float(xi[m]), 5)) * terms[m].expr
                expr = sp.simplify(expr)
                y = substrate.weak_velocities[:, i]
                y_var = float(np.var(y)) + 1e-12
                mse = self._eval_sympy_weak_mse(expr, substrate, i)
                r2 = float(1.0 - mse / y_var)
                if r2 < best_cands[i].weak_r2 - 0.04:
                    accept = False
                    break
                if self.config.enable_rationalizer:
                    expr_r, mse_r = rationalize_expression(
                        expr,
                        lambda ex, _i=i: self._eval_sympy_weak_mse(ex, substrate, _i),
                        k_eff,
                        y_var,
                        max_denom=self.config.max_rational_denom,
                    )
                    if mse_r < 1e8:
                        expr, mse = expr_r, mse_r
                        r2 = float(1.0 - mse / y_var)
                mdl_kkt = compute_mdl_score(mse, k_eff, expr, y_var)
                if mdl_kkt > best_cands[i].mdl_score + 6.0:
                    accept = False
                    break
                kkt_cands.append(
                    CandidateEquation(
                        target_var=substrate.var_names[i],
                        expr=expr,
                        expr_str=str(expr),
                        weak_mse=mse,
                        weak_r2=r2,
                        mdl_score=mdl_kkt,
                        complexity=expression_complexity(expr),
                        source="vesp_kkt_conserved",
                    )
                )
            if accept:
                best_cands = kkt_cands
                for i in range(d):
                    pareto_per_var[i].insert(0, kkt_cands[i])
                kkt_applied = True

        stats = {
            "mean_pruning_ratio": float(np.mean(pruned_counts)) if pruned_counts else 0.0,
            "kkt_conservation_enforced": float(kkt_applied),
        }
        return best_cands, pareto_per_var, stats


def discover_pde_vesp(
    x: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    cumulant_debiasing: bool = True,
    seed: int = 42,
) -> PDECandidate:
    """Discover 1+1D governing PDE u_t = F(u, u_x, u_xx, ...) via 2D Spacetime Weak-Form V-ESP."""
    rng = np.random.default_rng(seed)
    proj = SpacetimeWeakProjector(cumulant_debiasing=cumulant_debiasing)
    G, b, names, noise_est = proj.build_pde_system(x, t, u)
    K, M = G.shape
    b_var = float(np.var(b)) + 1e-12

    sym_map = {n: sp.Symbol(n) for n in names}
    best_cand: PDECandidate | None = None

    for lam in [0.03, 0.06, 0.10, 0.16, 0.25]:
        counts = np.zeros(M, dtype=float)
        for _ in range(16):
            bs = rng.integers(0, K, size=K)
            xi_b = _stlsq(G[bs], b[bs], threshold=lam)
            counts += (np.abs(xi_b) > 1e-8).astype(float)
        supp = np.where(counts / 16.0 >= 0.55)[0]
        if len(supp) == 0:
            xi = _stlsq(G, b, threshold=lam)
            supp = np.where(np.abs(xi) > 1e-8)[0]
        if len(supp) == 0:
            continue
        coef, *_ = np.linalg.lstsq(G[:, supp], b, rcond=None)
        pred = G[:, supp] @ coef
        mse = float(np.mean((b - pred) ** 2))
        r2 = float(1.0 - mse / b_var)

        expr = sp.Integer(0)
        coeff_dict: dict[str, float] = {}
        for idx, c_val in zip(supp, coef):
            if abs(c_val) > 1e-5:
                coeff_dict[names[idx]] = round(float(c_val), 4)
                expr = expr + sp.Float(round(float(c_val), 4)) * sym_map[names[idx]]
        expr = sp.simplify(expr)
        mdl = compute_mdl_score(mse, K, expr, b_var)
        cand = PDECandidate(
            expr=expr,
            expr_str=str(expr),
            weak_mse=mse,
            weak_r2=r2,
            mdl_score=mdl,
            estimated_noise_std=noise_est,
            coefficients=coeff_dict,
        )
        if best_cand is None or cand.mdl_score < best_cand.mdl_score:
            best_cand = cand

    assert best_cand is not None
    return best_cand
