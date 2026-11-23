"""Unified Multi-Evidence Confidence Scorer & Rigorous Symbolic Match Metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from asds_omega.causal.vifc import CausalDiscoveryResult
from asds_omega.symmetry.reynolds import SymmetryDiscoveryResult
from asds_omega.symbolic.vesp import CandidateEquation


@dataclass
class EvidenceReport:
    """Multi-dimensional scientific evidence assessment for a discovered dynamical system."""

    mean_weak_r2: float
    mean_weak_rmse: float
    mean_complexity: float
    symbolic_match_score: float | None  # [0, 1] exact structure + coefficient match vs GT
    exact_structure_rate: float | None  # Fraction of equations with identical symbolic term support
    mean_coeff_rel_error: float | None
    causal_score: float
    symmetry_score: float
    overall_confidence: float
    per_equation_match: list[float] = field(default_factory=list)


def _extract_additive_terms(expr: sp.Expr, syms: list[sp.Symbol]) -> dict[sp.Expr, float]:
    """Decompose an expanded SymPy expression into {symbolic_basis_monomial: float_coeff}."""
    expanded = sp.expand(expr)
    terms: dict[sp.Expr, float] = {}
    for term in sp.Add.make_args(expanded):
        coeff, rest = term.as_coeff_Mul()
        try:
            c_val = float(coeff.evalf())
        except Exception:
            c_val = 1.0
        if abs(c_val) > 1e-7:
            key = sp.simplify(rest)
            terms[key] = terms.get(key, 0.0) + c_val
    return terms


def compute_symbolic_match(
    pred_expr: sp.Expr,
    gt_expr_str: str,
    var_names: list[str],
) -> tuple[float, bool, float]:
    """Compare discovered SymPy equation against ground-truth equation rigorously.

    Returns:
        (symbolic_match_score in [0, 1], exact_structure_match bool, mean_coeff_rel_error)
    Handles both additive polynomial/trig expressions AND rational forms (e.g. Michaelis-Menten).
    """
    local_dict = {v: sp.Symbol(v) for v in var_names}
    try:
        gt_expr = sp.sympify(gt_expr_str, locals=local_dict)
    except Exception:
        return 0.0, False, 1.0

    # Handle genuinely rational functions (variables in denominator, e.g., Michaelis-Menten or Kepler 1/r^3)
    p_den = pred_expr.as_numer_denom()[1]
    g_den = gt_expr.as_numer_denom()[1]
    if len(p_den.free_symbols) > 0 or len(g_den.free_symbols) > 0:
        # Compare canonicalized numerators & denominators or relative numerical agreement on probe grid
        rng = np.random.default_rng(0)
        syms = [local_dict[v] for v in var_names]
        pts = rng.uniform(0.5, 3.0, size=(80, len(syms)))
        try:
            f_p = sp.lambdify(syms, pred_expr, modules="numpy")
            f_g = sp.lambdify(syms, gt_expr, modules="numpy")
            vp = np.asarray([float(f_p(*p)) for p in pts], dtype=float)
            vg = np.asarray([float(f_g(*p)) for p in pts], dtype=float)
            rel_err = float(np.linalg.norm(vp - vg) / (np.linalg.norm(vg) + 1e-12))
            same_vars = pred_expr.free_symbols == gt_expr.free_symbols
            exact_struct = same_vars and rel_err < 0.08
            score = float(np.clip((1.0 - rel_err) * (1.0 if same_vars else 0.6), 0.0, 1.0))
            return score, exact_struct, min(rel_err, 1.0)
        except Exception:
            return 0.0, False, 1.0

    syms = [local_dict[v] for v in var_names]
    p_terms = _extract_additive_terms(pred_expr, syms)
    g_terms = _extract_additive_terms(gt_expr, syms)

    p_keys = set(p_terms.keys())
    g_keys = set(g_terms.keys())
    union = p_keys | g_keys
    inter = p_keys & g_keys

    if not union:
        return 1.0, True, 0.0

    jaccard = len(inter) / len(union)
    exact_struct = (p_keys == g_keys)

    if inter:
        rel_errs = [
            abs(p_terms[k] - g_terms[k]) / max(abs(g_terms[k]), 1e-6)
            for k in inter
        ]
        mean_rel = float(np.mean(rel_errs))
        coeff_acc = float(np.exp(-1.5 * mean_rel))
    else:
        mean_rel = 1.0
        coeff_acc = 0.0

    score = float(np.clip(0.65 * jaccard + 0.35 * jaccard * coeff_acc, 0.0, 1.0))
    return score, exact_struct, min(mean_rel, 1.0)


def score_discovered_system(
    equations: list[CandidateEquation],
    causal_result: CausalDiscoveryResult,
    symmetry_result: SymmetryDiscoveryResult,
    var_names: list[str],
    ground_truth_equations: list[str] | None = None,
) -> EvidenceReport:
    """Aggregate symbolic, predictive, causal, and symmetry evidence into a unified score."""
    mean_r2 = float(np.mean([max(c.weak_r2, 0.0) for c in equations]))
    mean_rmse = float(np.mean([np.sqrt(max(c.weak_mse, 0.0)) for c in equations]))
    mean_comp = float(np.mean([c.complexity for c in equations]))

    # Causal consistency score: fraction of variables used in discovered equations supported by VIFC
    causal_agreements = []
    for i, cand in enumerate(equations):
        parents_vifc = set(var_names[j] for j in causal_result.get_parents(i))
        used_vars = {str(s) for s in cand.expr.free_symbols if str(s) in var_names}
        if not used_vars:
            causal_agreements.append(1.0)
        else:
            causal_agreements.append(len(used_vars & parents_vifc) / len(used_vars))
    causal_score = float(np.mean(causal_agreements))

    # Symmetry & invariant preservation score
    sym_scores = [1.0 - min(c.symmetry_residual, 1.0) for c in equations]
    if symmetry_result.invariants:
        inv_bonus = float(np.mean([1.0 - min(inv.weak_residual, 1.0) for inv in symmetry_result.invariants]))
        symmetry_score = 0.5 * float(np.mean(sym_scores)) + 0.5 * inv_bonus
    else:
        symmetry_score = float(np.mean(sym_scores))

    # Ground-truth symbolic evaluation (only for synthetic benchmarks with known analytic truth)
    sym_match: float | None = None
    exact_rate: float | None = None
    mean_coeff_err: float | None = None
    per_eq: list[float] = []
    if ground_truth_equations and not any("expected" in gt.lower() for gt in ground_truth_equations):
        exacts = []
        cerrs = []
        for cand, gt in zip(equations, ground_truth_equations):
            sm, ex, ce = compute_symbolic_match(cand.expr, gt, var_names)
            per_eq.append(sm)
            exacts.append(float(ex))
            cerrs.append(ce)
        sym_match = float(np.mean(per_eq))
        exact_rate = float(np.mean(exacts))
        mean_coeff_err = float(np.mean(cerrs))

    overall = float(np.clip(0.50 * mean_r2 + 0.25 * causal_score + 0.25 * symmetry_score, 0.0, 1.0))
    return EvidenceReport(
        mean_weak_r2=mean_r2,
        mean_weak_rmse=mean_rmse,
        mean_complexity=mean_comp,
        symbolic_match_score=sym_match,
        exact_structure_rate=exact_rate,
        mean_coeff_rel_error=mean_coeff_err,
        causal_score=causal_score,
        symmetry_score=symmetry_score,
        overall_confidence=overall,
        per_equation_match=per_eq,
    )
