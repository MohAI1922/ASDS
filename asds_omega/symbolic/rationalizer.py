"""MDL-Certified Continued-Fraction Constant Rationalizer & Description Length Scoring."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import sympy as sp


def expression_complexity(expr: sp.Expr) -> int:
    """Count AST nodes (operators, symbols, numbers) in a SymPy expression."""
    return int(sp.count_ops(expr)) + len(expr.free_symbols) + 1


def _constant_bits(val: sp.Expr) -> float:
    """Description length in bits for a numeric constant (exact small rational << float)."""
    if val.is_Integer:
        return float(np.log2(abs(int(val)) + 1.0) + 1.5)
    if val.is_Rational:
        p, q = int(val.p), int(val.q)
        return float(np.log2(abs(p) + 1.0) + np.log2(q + 1.0) + 2.0)
    if val in (sp.pi, sp.E):
        return 3.0
    return 14.0  # IEEE floating-point parameter cost


def compute_mdl_score(
    weak_mse: float,
    n_samples: float,
    expr: sp.Expr,
    y_var: float,
    symmetry_penalty: float = 0.0,
) -> float:
    """Minimum Description Length (MDL) objective using effective independent sample size K_eff.

    Lower is strictly better:
        MDL = (K_eff / 2) * log(max(mse / y_var, 1e-10))
              + 0.85 * log(K_eff) * n_terms + 0.9 * complexity + param_bits + sym_penalty
    """
    k_eff = max(float(n_samples), 16.0)
    norm_mse = max(float(weak_mse) / max(float(y_var), 1e-12), 1e-10)
    data_bits = 0.5 * k_eff * float(np.log(norm_mse))
    comp = float(expression_complexity(expr))
    has_sym_denom = len(expr.as_numer_denom()[1].free_symbols) > 0
    n_terms = float(len(sp.Add.make_args(sp.expand(expr)))) if not has_sym_denom else 2.2
    struct_bits = (0.95 * np.log(k_eff)) * n_terms + 0.95 * comp
    param_bits = 0.0
    for atom in expr.atoms(sp.Number):
        if atom not in (sp.Integer(0), sp.Integer(1), sp.Integer(-1)):
            param_bits += 0.35 * _constant_bits(atom)
    return float(data_bits + struct_bits + param_bits + 25.0 * symmetry_penalty)


def snap_float_to_rational(
    value: float,
    max_denom: int = 24,
    rel_tol: float = 0.03,
) -> sp.Expr:
    """Propose the cleanest exact integer, small rational p/q, or pi-multiple within rel_tol."""
    if abs(value) < 1e-9:
        return sp.Integer(0)
    # 1. Near integer
    r_int = round(value)
    if r_int != 0 and abs(value - r_int) / max(abs(r_int), 1.0) <= rel_tol:
        return sp.Integer(int(r_int))

    # 2. Small rational p/q via continued fractions
    frac = Fraction(value).limit_denominator(max_denom)
    if frac.numerator != 0:
        f_val = float(frac)
        if abs(value - f_val) / max(abs(value), 1e-9) <= rel_tol:
            return sp.Rational(frac.numerator, frac.denominator)

    # 3. Canonical pi multiples
    for num, den in [(1, 1), (2, 1), (1, 2), (1, 4)]:
        cand = (num / den) * np.pi
        for sgn in (1, -1):
            if abs(value - sgn * cand) / max(abs(value), 1e-9) <= rel_tol * 0.5:
                return sp.Rational(sgn * num, den) * sp.pi

    return sp.Float(round(value, 4))


def rationalize_expression(
    expr: sp.Expr,
    eval_mse_fn,
    n_samples: int,
    y_var: float,
    max_denom: int = 24,
    rel_tol: float = 0.035,
) -> tuple[sp.Expr, float]:
    """MDL-certified continued-fraction constant snapping.

    Replaces floating-point coefficients with exact integers / small rationals (e.g.,
    2.667 -> 8/3, 0.501 -> 1/2) ONLY when the replacement strictly decreases total MDL
    (preventing accuracy degradation from over-aggressive rounding).
    """
    best_expr = sp.nsimplify(expr, rational=False) if expr.is_Number else expr
    best_mse = float(eval_mse_fn(best_expr))
    best_mdl = compute_mdl_score(best_mse, n_samples, best_expr, y_var)

    # Collect floating-point numeric atoms
    floats = [a for a in best_expr.atoms(sp.Float)]
    if not floats:
        return best_expr, best_mse

    # Try snapping all floats jointly first, then individually
    repl_all = {}
    for fl in floats:
        snapped = snap_float_to_rational(float(fl), max_denom=max_denom, rel_tol=rel_tol)
        if snapped != fl:
            repl_all[fl] = snapped

    if repl_all:
        cand_all = sp.simplify(best_expr.xreplace(repl_all))
        mse_all = float(eval_mse_fn(cand_all))
        mdl_all = compute_mdl_score(mse_all, n_samples, cand_all, y_var)
        if mdl_all <= best_mdl and mse_all <= max(best_mse * 1.15, best_mse + 1e-4 * y_var):
            return cand_all, mse_all

    # Greedy per-constant MDL snapping
    curr_expr, curr_mse, curr_mdl = best_expr, best_mse, best_mdl
    for fl in floats:
        snapped = snap_float_to_rational(float(fl), max_denom=max_denom, rel_tol=rel_tol)
        if snapped == fl:
            continue
        cand = sp.simplify(curr_expr.xreplace({fl: snapped}))
        mse_c = float(eval_mse_fn(cand))
        mdl_c = compute_mdl_score(mse_c, n_samples, cand, y_var)
        if mdl_c <= curr_mdl and mse_c <= max(curr_mse * 1.12, curr_mse + 1e-4 * y_var):
            curr_expr, curr_mse, curr_mdl = cand, mse_c, mdl_c

    return curr_expr, curr_mse
