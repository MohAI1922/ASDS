"""Layer 5: Autonomous Scientific Agency — Evidence Scoring, Active Falsification, Drift Regeneration & Pipeline."""

from asds_omega.scientist.active_falsifier import (
    ActiveExperimentStep,
    ActiveFalsificationResult,
    PopperianActiveFalsifier,
)
from asds_omega.scientist.drift_regenerator import (
    DriftRegenerationResult,
    SurgicalDriftRegenerator,
)
from asds_omega.scientist.evidence_scorer import (
    EvidenceReport,
    compute_symbolic_match,
    score_discovered_system,
)
from asds_omega.scientist.pipeline import (
    ASDSOmegaPipeline,
    DiscoveryResult,
)

__all__ = [
    "ASDSOmegaPipeline",
    "DiscoveryResult",
    "EvidenceReport",
    "compute_symbolic_match",
    "score_discovered_system",
    "PopperianActiveFalsifier",
    "ActiveFalsificationResult",
    "ActiveExperimentStep",
    "SurgicalDriftRegenerator",
    "DriftRegenerationResult",
]
