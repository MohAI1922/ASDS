"""Unified Variational Interventional Flow Causality (VIFC) Engine.

Fuses three orthogonal dynamical causal signatures in weak-form operator space:
1. WF-cHSIC with Manifold-Aware Faithfulness Guard & Temporal Knockoff FDR control
2. Multivariate Partial Convergent Cross Mapping (pCCM) in delay-differential space
3. Multi-Trajectory Invariant Causal Prediction (Dyn-ICP)

Produces a calibrated direct differential causal graph G_diff: x_j(t) -> dx_i/dt,
exact SHD/Precision/Recall/F1 metrics, and causal parent masks for S-ECE v2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from asds_omega.causal.dyn_icp import compute_dyn_icp_matrix
from asds_omega.causal.partial_ccm import compute_partial_ccm_matrix
from asds_omega.causal.wf_chsic import compute_wf_chsic_knockoff_matrix
from asds_omega.config import VIFCConfig
from asds_omega.preprocessing.ms_avp import WeakFormSubstrate


@dataclass
class CausalDiscoveryResult:
    """Structured output of Layer 2 (CSIL-DynCausal / VIFC)."""

    var_names: list[str]
    adjacency_matrix: np.ndarray  # Binary (d, d): A[j, i] = 1 iff x_j -> dx_i/dt
    probability_matrix: np.ndarray  # Calibrated P_causal[j, i] in [0, 1]
    wf_chsic_matrix: np.ndarray
    pccm_matrix: np.ndarray
    dyn_icp_matrix: np.ndarray
    edges: list[tuple[str, str, float]]  # (parent_var, target_deriv_var, prob)
    graph: nx.DiGraph
    metrics: dict[str, float] = field(default_factory=dict)

    def get_parents(self, target_idx: int) -> list[int]:
        """Return list of inferred causal parent variable indices for dx_{target_idx}/dt."""
        parents = [int(j) for j in range(len(self.var_names)) if self.adjacency_matrix[j, target_idx] == 1]
        # Safety fallback: never return an empty parent set unless all probabilities are near zero
        if not parents:
            best = int(np.argmax(self.probability_matrix[:, target_idx]))
            if self.probability_matrix[best, target_idx] > 0.1:
                parents = [best]
        return parents


def compute_causal_metrics(pred_adj: np.ndarray, true_adj: np.ndarray) -> dict[str, float]:
    """Compute exact Differential Structural Hamming Distance (SHD), Precision, Recall, F1."""
    pred = (np.asarray(pred_adj) > 0).astype(int)
    true = (np.asarray(true_adj) > 0).astype(int)
    shd = int(np.sum(pred != true))
    tp = int(np.sum((pred == 1) & (true == 1)))
    fp = int(np.sum((pred == 1) & (true == 0)))
    fn = int(np.sum((pred == 0) & (true == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = float(np.mean(pred == true))
    return {
        "shd": float(shd),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "edge_accuracy": acc,
        "missing_edges": float(fn),
        "extra_edges": float(fp),
    }


class VIFCEngine:
    """Layer 2: Variational Interventional Flow Causality (VIFC)."""

    def __init__(self, config: VIFCConfig | None = None, seed: int = 42) -> None:
        self.config = config or VIFCConfig()
        self.seed = seed

    def discover(
        self,
        substrate: WeakFormSubstrate,
        ground_truth_adj: np.ndarray | None = None,
    ) -> CausalDiscoveryResult:
        """Infer direct differential causal graph x_j -> dx_i/dt from weak-form substrate."""
        d = substrate.n_vars
        var_names = substrate.var_names

        if not self.config.enabled:
            adj = np.ones((d, d), dtype=int)
            prob = np.ones((d, d), dtype=float)
            return self._package(var_names, adj, prob, prob, prob, prob, ground_truth_adj)

        _, p_chsic = compute_wf_chsic_knockoff_matrix(
            substrate,
            n_knockoffs=self.config.n_knockoffs,
            fdr_level=self.config.fdr_level,
            seed=self.seed,
        )
        p_icp = compute_dyn_icp_matrix(substrate) if self.config.icp_enabled else np.zeros((d, d))
        p_ccm = (
            compute_partial_ccm_matrix(
                substrate,
                embed_dim=self.config.ccm_embed_dim,
                tau=self.config.ccm_tau,
                seed=self.seed,
            )
            if self.config.ccm_enabled
            else np.zeros((d, d))
        )

        # Calibrated fusion:
        # Dyn-ICP directly tests out-of-regime differential predictive necessity (highest weight),
        # WF-cHSIC provides nonparametric nonlinear independence with knockoff FDR control,
        # pCCM provides Takens manifold confirmation for chaotic attractors.
        if self.config.icp_enabled and self.config.ccm_enabled:
            prob = 0.55 * p_icp + 0.30 * p_chsic + 0.15 * p_ccm
            # Strong single-estimator override only when ICP + cHSIC agree or ICP is decisive
            prob = np.maximum(prob, np.minimum(p_icp, 0.85) * (p_icp > 0.78))
        elif self.config.icp_enabled:
            prob = 0.65 * p_icp + 0.35 * p_chsic
        else:
            prob = 0.70 * p_chsic + 0.30 * p_ccm

        prob = np.clip(prob, 0.0, 1.0)
        # Noise-adaptive recall preservation: maintain high precision at low noise (0-10%)
        # while gently relaxing threshold at extreme noise (15-30%) to prevent parent drops.
        rel_noise = float(substrate.metadata.get("relative_noise_est", 0.0))
        eff_edge_thresh = max(0.20, self.config.edge_threshold - 1.15 * max(rel_noise - 0.08, 0.0))
        adj = (prob >= eff_edge_thresh).astype(int)

        # Ensure every dynamically evolving variable has at least one parent
        for i in range(d):
            if np.sum(adj[:, i]) == 0 and float(np.std(substrate.weak_velocities[:, i])) > 1e-6:
                adj[int(np.argmax(prob[:, i])), i] = 1

        return self._package(var_names, adj, prob, p_chsic, p_ccm, p_icp, ground_truth_adj)

    def _package(
        self,
        var_names: list[str],
        adj: np.ndarray,
        prob: np.ndarray,
        p_chsic: np.ndarray,
        p_ccm: np.ndarray,
        p_icp: np.ndarray,
        ground_truth_adj: np.ndarray | None,
    ) -> CausalDiscoveryResult:
        d = len(var_names)
        G = nx.DiGraph()
        for v in var_names:
            G.add_node(v)
        edges: list[tuple[str, str, float]] = []
        for j in range(d):
            for i in range(d):
                if adj[j, i] == 1:
                    w = float(prob[j, i])
                    G.add_edge(var_names[j], var_names[i], weight=w)
                    edges.append((var_names[j], f"d({var_names[i]})/dt", w))

        metrics = compute_causal_metrics(adj, ground_truth_adj) if ground_truth_adj is not None else {}
        return CausalDiscoveryResult(
            var_names=var_names,
            adjacency_matrix=adj,
            probability_matrix=prob,
            wf_chsic_matrix=p_chsic,
            pccm_matrix=p_ccm,
            dyn_icp_matrix=p_icp,
            edges=edges,
            graph=G,
            metrics=metrics,
        )
