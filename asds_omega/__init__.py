"""ASDS-OMEGA v2.0 — Variational Equivariant Causal-Symbolic Discovery (VECSD).

A unified, research-grade framework for autonomous discovery of governing
differential equations, direct differential causal graphs, discrete/continuous
symmetries, conservation laws, active experimental falsification, online
structural drift regeneration, and formal symbolic/Lean-4 verification.
"""

from asds_omega.config import (
    EVGHEConfig,
    MSAVPConfig,
    PipelineConfig,
    ScientistConfig,
    VIFCConfig,
    WLINDConfig,
    set_global_seed,
)

__version__ = "2.0.0"

__all__ = [
    "__version__",
    "MSAVPConfig",
    "VIFCConfig",
    "WLINDConfig",
    "EVGHEConfig",
    "ScientistConfig",
    "PipelineConfig",
    "set_global_seed",
]
