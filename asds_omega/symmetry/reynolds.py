"""Exact Reynolds Equivariant Projection Operator P_G & Unified W-LIND Engine.

Constructs the group-averaging Reynolds Operator from discovered discrete symmetries:
    P_G[f](x) = (1/|G|) sum_{g in G} g^{-1} f(g * x)
which analytically prunes every non-equivariant basis term BEFORE symbolic search,
and bundles linear conservation orthogonality constraints c^T dx/dt = 0 for coupled
KKT-constrained optimization in S-ECE v2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from asds_omega.config import WLINDConfig
from asds_omega.preprocessing.ms_avp import WeakFormSubstrate
from asds_omega.symmetry.discrete_groups import DiscreteSymmetry, discover_discrete_symmetries
from asds_omega.symmetry.divergence import DynamicalRegimeProfile, analyze_divergence_and_chaos
from asds_omega.symmetry.lie_prolongation import (
    LieGenerator,
    discover_lie_symmetries,
    verify_lie_closure,
)
from asds_omega.symmetry.weak_invariants import DiscoveredInvariant, discover_weak_invariants


class ReynoldsEquivariantProjector:
    """Exact discrete group-averaging Reynolds projection operator P_G on symbolic expressions."""

    def __init__(self, var_names: list[str], discrete_symmetries: list[DiscreteSymmetry]) -> None:
        self.var_names = var_names
        self.symbols = [sp.Symbol(v) for v in var_names]
        self.symmetries = discrete_symmetries

    @property
    def is_trivial(self) -> bool:
        return len(self.symmetries) == 0

    def is_term_equivariant(self, expr: sp.Expr, target_idx: int) -> bool:
        """Check whether candidate term `expr` lies in Im(P_G) for target equation dx_{target_idx}/dt."""
        if not self.symmetries:
            return True
        for sym in self.symmetries:
            target_sign = sym.signs[target_idx]
            subs_map = {s: sg * s for s, sg in zip(self.symbols, sym.signs)}
            transformed = expr.subs(subs_map)
            # Check exact algebraic equivariance: expr(g*x) == target_sign * expr(x)
            diff = sp.simplify(transformed - target_sign * expr)
            if diff != 0:
                return False
        return True

    def project_expression(self, expr: sp.Expr, target_idx: int) -> sp.Expr:
        """Apply exact Reynolds operator P_G[expr](x) = (1/|G|) sum_g g_i^{-1} expr(g*x)."""
        if not self.symmetries:
            return expr
        # Generate subgroup elements (identity + discovered involutions)
        elements = [tuple([1] * len(self.var_names))] + [s.signs for s in self.symmetries]
        acc = sp.Integer(0)
        for signs in elements:
            subs_map = {s: sg * s for s, sg in zip(self.symbols, signs)}
            acc = acc + signs[target_idx] * expr.subs(subs_map)
        return sp.simplify(acc / len(elements))

    def filter_basis_indices(self, basis_exprs: list[sp.Expr], target_idx: int) -> list[int]:
        """Return indices of basis terms surviving exact Reynolds projection for equation `target_idx`."""
        return [idx for idx, expr in enumerate(basis_exprs) if self.is_term_equivariant(expr, target_idx)]


@dataclass
class SymmetryDiscoveryResult:
    """Unified output of Layer 3 (CSIL-Symmetry / W-LIND)."""

    invariants: list[DiscoveredInvariant]
    discrete_symmetries: list[DiscreteSymmetry]
    lie_generators: list[LieGenerator]
    lie_closure: dict[str, float | bool]
    regime_profile: DynamicalRegimeProfile
    reynolds_projector: ReynoldsEquivariantProjector
    linear_conservation_matrix: np.ndarray | None = None  # (n_lin_inv, d): C @ dx/dt = 0
    pruning_ratio: float = 0.0
    metadata: dict = field(default_factory=dict)


class WLINDEngine:
    """Layer 3: Model-Free Weak-Form Lie-Noether & Foliation Discovery (W-LIND)."""

    def __init__(self, config: WLINDConfig | None = None) -> None:
        self.config = config or WLINDConfig()

    def discover(
        self,
        substrate: WeakFormSubstrate,
        symbolic_equations: list[sp.Expr] | None = None,
    ) -> SymmetryDiscoveryResult:
        """Execute full W-LIND geometric, invariant, and symmetry discovery."""
        if not self.config.enabled:
            empty_proj = ReynoldsEquivariantProjector(substrate.var_names, [])
            dummy_regime = DynamicalRegimeProfile(0.0, 0.0, None, 0.0, "disabled")
            return SymmetryDiscoveryResult(
                invariants=[],
                discrete_symmetries=[],
                lie_generators=[],
                lie_closure={"closed": True, "n_generators": 0, "max_closure_residual": 0.0},
                regime_profile=dummy_regime,
                reynolds_projector=empty_proj,
                linear_conservation_matrix=None,
            )

        invariants = discover_weak_invariants(
            substrate,
            max_degree=self.config.max_invariant_degree,
            include_log=self.config.include_log_invariants,
            include_trig=self.config.include_trig_invariants,
            include_inverse_norm=self.config.include_inverse_norm,
            sv_threshold=self.config.invariant_sv_threshold,
        )
        discrete_syms = discover_discrete_symmetries(
            substrate,
            eq_threshold=self.config.discrete_eq_threshold,
        )
        lie_gens = discover_lie_symmetries(
            substrate,
            symbolic_equations=symbolic_equations,
            sv_threshold=self.config.lie_sv_threshold,
        )
        closure = verify_lie_closure(lie_gens)
        regime = analyze_divergence_and_chaos(substrate, symbolic_equations=symbolic_equations)

        # Build exact Reynolds projector (only when enabled)
        use_syms = discrete_syms if self.config.enforce_reynolds_pruning else []
        reynolds = ReynoldsEquivariantProjector(substrate.var_names, use_syms)

        # Collect exact linear conservation constraints C @ dx/dt = 0 (e.g., dS + dI + dR = 0)
        lin_rows = [inv.linear_weights for inv in invariants if inv.linear_weights is not None]
        C_lin = np.vstack(lin_rows) if (lin_rows and self.config.enforce_conservation_kkt) else None

        return SymmetryDiscoveryResult(
            invariants=invariants,
            discrete_symmetries=discrete_syms,
            lie_generators=lie_gens,
            lie_closure=closure,
            regime_profile=regime,
            reynolds_projector=reynolds,
            linear_conservation_matrix=C_lin,
        )
