"""Configuration dataclasses, deterministic seeding, and provenance hashing."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    """Set deterministic RNG states across Python and NumPy and return a Generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


@dataclass
class MSAVPConfig:
    """Layer 1: Multi-Scale Adaptive Variational Projection (MS-AVP) settings."""

    poly_order: int = 4  # Test function bump smoothness: phi(u) = (1 - u^2)^p
    n_subdomains: int = 90  # Number of variational integration windows per trajectory
    min_window_pts: int = 11  # Minimum odd support size in samples
    max_window_pts: int = 81  # Maximum odd support size in samples
    adaptive_multiscale: bool = True  # Local curvature-optimal W_k* vs fixed window
    cumulant_debiasing: bool = True  # Analytical Hermite / Wick EIV bias correction
    noise_override: float | None = None  # Optional known noise std (else Wavelet-MAD)
    use_weak_form: bool = True  # False triggers Savitzky-Golay finite-diff ablation


@dataclass
class VIFCConfig:
    """Layer 2: Variational Interventional Flow Causality (VIFC) settings."""

    enabled: bool = True
    fdr_level: float = 0.15  # Target false discovery rate for temporal knockoffs
    n_knockoffs: int = 12  # Block-permuted knockoff ensembles for WF-cHSIC
    ccm_embed_dim: int = 3  # Takens delay embedding dimension for pCCM
    ccm_tau: int = 2  # Delay lag in samples
    ccm_enabled: bool = True
    icp_enabled: bool = True  # Multi-trajectory / regime Invariant Causal Prediction
    edge_threshold: float = 0.42  # Calibrated posterior threshold for causal parent mask
    soft_escape_prob: float = 0.08  # Escape rate in GP grammar for false-negative safety


@dataclass
class WLINDConfig:
    """Layer 3: Weak-Form Lie-Noether & Foliation Discovery (W-LIND) settings."""

    enabled: bool = True
    max_invariant_degree: int = 2
    include_log_invariants: bool = True
    include_trig_invariants: bool = True
    include_inverse_norm: bool = True
    invariant_sv_threshold: float = 0.045  # Normalized nullspace residual cutoff
    discrete_eq_threshold: float = 0.18  # Relative weak-form pushforward equivariance cutoff
    lie_max_degree: int = 1  # Affine/rotational infinitesimal generators
    lie_sv_threshold: float = 0.06
    enforce_reynolds_pruning: bool = True
    enforce_conservation_kkt: bool = True


@dataclass
class EVGHEConfig:
    """Layer 4: Equivariant Variational Grammar-Guided Hybrid Evolution (EVG-HE)."""

    max_poly_degree: int = 3
    include_trig: bool = True
    include_rational_implicit: bool = True
    include_inverse_r3: bool = False  # Enabled automatically for central-force / auto
    stlsq_threshold: float = 0.08
    n_ensemble_bootstraps: int = 20
    inclusion_prob_threshold: float = 0.55
    enable_gp_refinement: bool = True
    gp_population_size: int = 36
    gp_generations: int = 18
    gp_max_depth: int = 4
    gp_semantic_mutation_rate: float = 0.65
    enable_rationalizer: bool = True
    max_rational_denom: int = 24
    use_meta_controller: bool = True


@dataclass
class ScientistConfig:
    """Layer 5 & 6: Active Falsification (PAFE), Drift Regeneration (SLR) & Verification."""

    enable_active_falsification: bool = True
    active_budget: int = 3
    active_n_candidates: int = 24
    enable_drift_regeneration: bool = True
    cusum_threshold: float = 4.5
    drift_window_pts: int = 40
    export_lean4: bool = True


@dataclass
class PipelineConfig:
    """Top-level reproducible experiment configuration for ASDS-OMEGA v2.0."""

    seed: int = 42
    ms_avp: MSAVPConfig = field(default_factory=MSAVPConfig)
    vifc: VIFCConfig = field(default_factory=VIFCConfig)
    wlind: WLINDConfig = field(default_factory=WLINDConfig)
    evghe: EVGHEConfig = field(default_factory=EVGHEConfig)
    scientist: ScientistConfig = field(default_factory=ScientistConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]
