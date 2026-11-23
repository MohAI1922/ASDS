"""Layer 4: Equivariant Variational Grammar-Guided Hybrid Evolution (EVG-HE)."""

from asds_omega.symbolic.meta_controller import (
    MetaControllerDecision,
    ThompsonMetaController,
)
from asds_omega.symbolic.rationalizer import (
    compute_mdl_score,
    expression_complexity,
    rationalize_expression,
)
from asds_omega.symbolic.srg_gp import (
    GPIndividual,
    SemanticWeakFormGP,
)
from asds_omega.symbolic.vesp import (
    CandidateEquation,
    PDECandidate,
    VESPEngine,
    discover_pde_vesp,
)

__all__ = [
    "VESPEngine",
    "CandidateEquation",
    "PDECandidate",
    "discover_pde_vesp",
    "SemanticWeakFormGP",
    "GPIndividual",
    "ThompsonMetaController",
    "MetaControllerDecision",
    "compute_mdl_score",
    "expression_complexity",
    "rationalize_expression",
]
