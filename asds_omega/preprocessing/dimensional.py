"""Buckingham Pi Dimensional Analysis & SI Unit Consistency Engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp


@dataclass
class DimensionalResult:
    """Output of Buckingham Pi dimensional analysis."""

    var_names: list[str]
    unit_matrix: np.ndarray  # (n_dims, n_vars)
    pi_groups: list[sp.Expr]
    dimensionless_vars: list[str]
    rank: int
    n_pi_groups: int


class DimensionalAnalyzer:
    """Exact rational Buckingham Pi theorem & AST dimensional homogeneity verifier.

    Dimensions are represented as tuples of rational/integer exponents over base
    SI units (default: [Mass M, Length L, Time T]).
    """

    def __init__(self, units: dict[str, tuple[int, ...]] | None = None) -> None:
        self.units = units or {}

    def analyze(self, var_names: list[str]) -> DimensionalResult | None:
        """Compute exact rational nullspace of dimensional matrix to extract Buckingham Pi groups."""
        if not self.units or not all(v in self.units for v in var_names):
            return None
        dim_cols = [self.units[v] for v in var_names]
        n_dims = len(dim_cols[0])
        U = sp.Matrix([[dim_cols[j][i] for j in range(len(var_names))] for i in range(n_dims)])
        null_vecs = U.nullspace()
        syms = [sp.Symbol(v, positive=True) for v in var_names]
        pi_groups: list[sp.Expr] = []
        for vec in null_vecs:
            # Clear denominators for clean canonical dimensionless products
            denoms = [sp.fraction(sp.Rational(c))[1] for c in vec]
            lcm_d = sp.ilcm(*denoms) if denoms else 1
            scaled = [int(sp.Rational(c) * lcm_d) for c in vec]
            expr = sp.Integer(1)
            for sym, exp in zip(syms, scaled):
                if exp != 0:
                    expr = expr * (sym**exp)
            pi_groups.append(sp.simplify(expr))

        dimless = [v for v in var_names if all(e == 0 for e in self.units[v])]
        return DimensionalResult(
            var_names=var_names,
            unit_matrix=np.array(dim_cols, dtype=float).T,
            pi_groups=pi_groups,
            dimensionless_vars=dimless,
            rank=int(U.rank()),
            n_pi_groups=len(pi_groups),
        )

    def infer_term_dimension(self, expr: sp.Expr) -> tuple[sp.Rational, ...] | None:
        """Infer SI unit exponent vector of a SymPy expression, or None if dimensionally inconsistent."""
        if not self.units:
            return None
        n_dims = len(next(iter(self.units.values())))
        zero = tuple(sp.Rational(0) for _ in range(n_dims))

        def _rec(node: sp.Expr) -> tuple[sp.Rational, ...] | None:
            if node.is_Number:
                return zero
            if isinstance(node, sp.Symbol):
                name = str(node)
                if name in self.units:
                    return tuple(sp.Rational(e) for e in self.units[name])
                return zero
            if isinstance(node, sp.Mul):
                acc = list(zero)
                for arg in node.args:
                    d = _rec(arg)
                    if d is None:
                        return None
                    for k in range(n_dims):
                        acc[k] += d[k]
                return tuple(acc)
            if isinstance(node, sp.Pow):
                base, exp = node.args
                db = _rec(base)
                de = _rec(exp)
                if db is None or de is None or de != zero:
                    return None
                if not exp.is_Number:
                    return None
                r = sp.Rational(exp)
                return tuple(r * e for e in db)
            if isinstance(node, sp.Add):
                # Terms in a sum must match dimensions unless a coefficient absorbs units;
                # for strictly dimensionless transcendental arguments we enforce exact match.
                dims = [_rec(arg) for arg in node.args]
                if any(d is None for d in dims):
                    return None
                non_zero = [d for d in dims if d != zero]
                if not non_zero:
                    return zero
                first = non_zero[0]
                if all(d == first for d in non_zero):
                    return first
                return None
            if node.func in (sp.sin, sp.cos, sp.tan, sp.exp, sp.log):
                # Transcendental functions MUST receive dimensionless arguments!
                d_arg = _rec(node.args[0])
                if d_arg != zero:
                    return None
                return zero
            return zero

        return _rec(expr)

    def is_transcendentally_valid(self, expr: sp.Expr) -> bool:
        """Verify that every transcendental function receives strictly dimensionless arguments."""
        if not self.units:
            return True
        n_dims = len(next(iter(self.units.values())))
        zero = tuple(sp.Rational(0) for _ in range(n_dims))
        for node in sp.preorder_traversal(expr):
            if node.func in (sp.sin, sp.cos, sp.tan, sp.exp, sp.log):
                d_arg = self.infer_term_dimension(node.args[0])
                if d_arg != zero:
                    return False
        return True
