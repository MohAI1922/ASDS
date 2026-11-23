"""End-to-End ASDS-OMEGA v2.0 (VECSD) Autonomous Scientific Discovery Pipeline.

Orchestrates all 6 research layers seamlessly:
  Layer 1: MS-AVP (Multi-Scale Adaptive Variational Projection + Cumulant EIV Debiasing + Buckingham Pi)
  Layer 2: CSIL-DynCausal (VIFC: WF-cHSIC Knockoffs + Partial CCM + Multi-Trajectory Dyn-ICP)
  Layer 3: CSIL-Symmetry (W-LIND: Model-Free Invariants + Discrete Groups + Lie Prolongation + Reynolds)
  Layer 4: S-ECE v2 (EVG-HE: Pruned V-ESP + Implicit Rational + KKT Conservation + Semantic Weak-Form GP + Meta)
  Layer 5: ASDS-Core (Evidence Scoring + Popperian Active Falsification + Online Surgical Drift Regeneration)
  Layer 6: ASDS-Verify (SymPy Formal Prover + Symbolic Lie Refinement + Lean 4 Certificates + LaTeX/JSON Export)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

from asds_omega.causal.vifc import CausalDiscoveryResult, VIFCEngine
from asds_omega.config import PipelineConfig, set_global_seed
from asds_omega.data.synthetic import DynamicalDataset
from asds_omega.preprocessing.dimensional import DimensionalAnalyzer, DimensionalResult
from asds_omega.preprocessing.ms_avp import MSAVPProjector, WeakFormSubstrate
from asds_omega.scientist.active_falsifier import ActiveFalsificationResult, PopperianActiveFalsifier
from asds_omega.scientist.evidence_scorer import EvidenceReport, score_discovered_system
from asds_omega.symmetry.divergence import analyze_divergence_and_chaos
from asds_omega.symmetry.lie_prolongation import discover_lie_symmetries, verify_lie_closure
from asds_omega.symmetry.reynolds import SymmetryDiscoveryResult, WLINDEngine
from asds_omega.symbolic.meta_controller import ThompsonMetaController
from asds_omega.symbolic.srg_gp import SemanticWeakFormGP
from asds_omega.symbolic.vesp import CandidateEquation, VESPEngine
from asds_omega.verification.lean_exporter import export_lean4_certificate
from asds_omega.verification.sympy_prover import FormalProofCertificate, SymPyFormalProver


@dataclass
class DiscoveryResult:
    """Complete reproducible scientific discovery result returned by ASDSOmegaPipeline."""

    dataset_name: str
    domain: str
    var_names: list[str]
    config_hash: str
    seed: int
    runtime_sec: float
    substrate: WeakFormSubstrate
    dimensional: DimensionalResult | None
    causal: CausalDiscoveryResult
    symmetry: SymmetryDiscoveryResult
    equations: list[CandidateEquation]
    pareto_fronts: list[list[CandidateEquation]]
    evidence: EvidenceReport
    verification: FormalProofCertificate
    lean4_certificate: str | None
    active_falsification: ActiveFalsificationResult | None
    meta_summary: dict[str, Any]
    pruning_ratio: float
    kkt_conservation_enforced: bool

    def to_summary_dict(self) -> dict[str, Any]:
        """Serialize into a clean JSON-ready provenance & scientific summary dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "domain": self.domain,
            "var_names": self.var_names,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "runtime_sec": round(self.runtime_sec, 4),
            "estimated_noise_stds": [round(float(x), 5) for x in self.substrate.noise_stds],
            "n_subdomains": self.substrate.n_subdomains,
            "pruning_ratio": round(self.pruning_ratio, 4),
            "kkt_conservation_enforced": self.kkt_conservation_enforced,
            "dimensional_pi_groups": [str(g) for g in self.dimensional.pi_groups] if self.dimensional else [],
            "discovered_equations": [
                {
                    "target_var": c.target_var,
                    "expr_str": c.expr_str,
                    "weak_r2": round(c.weak_r2, 5),
                    "weak_mse": float(c.weak_mse),
                    "mdl_score": round(c.mdl_score, 3),
                    "complexity": c.complexity,
                    "source": c.source,
                }
                for c in self.equations
            ],
            "causal": {
                "adjacency_matrix": self.causal.adjacency_matrix.tolist(),
                "probability_matrix": np.round(self.causal.probability_matrix, 3).tolist(),
                "edges": [(u, v, round(w, 3)) for u, v, w in self.causal.edges],
                "metrics": self.causal.metrics,
            },
            "symmetry": {
                "regime_class": self.symmetry.regime_profile.regime_class,
                "mean_divergence": round(self.symmetry.regime_profile.mean_divergence, 4),
                "symbolic_divergence": self.symmetry.regime_profile.symbolic_divergence,
                "max_lyapunov_est": round(self.symmetry.regime_profile.max_lyapunov_est, 4),
                "discrete_symmetries": [
                    {"description": s.description, "signs": list(s.signs), "error": round(s.equivariance_error, 4)}
                    for s in self.symmetry.discrete_symmetries
                ],
                "invariants": [
                    {
                        "expr": inv.expr_str,
                        "family": inv.family,
                        "weak_residual": round(inv.weak_residual, 4),
                        "cancellation_ratio": round(inv.cancellation_ratio, 4),
                    }
                    for inv in self.symmetry.invariants
                ],
                "lie_generators": [
                    {
                        "type": g.symmetry_type,
                        "components": [str(c) for c in g.sympy_components],
                        "residual": round(g.lie_residual, 4),
                    }
                    for g in self.symmetry.lie_generators
                ],
                "lie_closure": self.symmetry.lie_closure,
            },
            "evidence": {
                "mean_weak_r2": round(self.evidence.mean_weak_r2, 5),
                "mean_weak_rmse": round(self.evidence.mean_weak_rmse, 5),
                "mean_complexity": round(self.evidence.mean_complexity, 2),
                "symbolic_match_score": (
                    round(self.evidence.symbolic_match_score, 4)
                    if self.evidence.symbolic_match_score is not None
                    else None
                ),
                "exact_structure_rate": (
                    round(self.evidence.exact_structure_rate, 4)
                    if self.evidence.exact_structure_rate is not None
                    else None
                ),
                "mean_coeff_rel_error": (
                    round(self.evidence.mean_coeff_rel_error, 4)
                    if self.evidence.mean_coeff_rel_error is not None
                    else None
                ),
                "causal_score": round(self.evidence.causal_score, 4),
                "symmetry_score": round(self.evidence.symmetry_score, 4),
                "overall_confidence": round(self.evidence.overall_confidence, 4),
            },
            "verification": {
                "pass_rate": round(self.verification.pass_rate, 4),
                "symbolic_divergence": self.verification.symbolic_divergence,
                "equilibria": self.verification.equilibria,
                "conservation_proofs": [
                    {
                        "theorem": p.theorem_name,
                        "statement": p.statement,
                        "proved": p.proved_exact,
                        "residual": p.symbolic_residual,
                    }
                    for p in self.verification.conservation_proofs
                ],
                "discrete_symmetry_proofs": [
                    {
                        "theorem": p.theorem_name,
                        "statement": p.statement,
                        "proved": p.proved_exact,
                    }
                    for p in self.verification.discrete_symmetry_proofs
                ],
                "lie_symmetry_proofs": [
                    {
                        "theorem": p.theorem_name,
                        "statement": p.statement,
                        "proved": p.proved_exact,
                    }
                    for p in self.verification.lie_symmetry_proofs
                ],
                "derived_scientific_insights": self.verification.derived_scientific_insights,
            },
            "meta_controller": self.meta_summary,
            "active_falsification": (
                {
                    "n_hypotheses_initial": self.active_falsification.n_hypotheses_initial,
                    "entropy_reduction": round(self.active_falsification.entropy_reduction, 4),
                    "random_baseline_entropy_reduction": round(
                        self.active_falsification.random_baseline_entropy_reduction, 4
                    ),
                    "final_posteriors": self.active_falsification.final_posteriors,
                    "steps": [
                        {
                            "step": st.step,
                            "selected_x0": st.selected_x0,
                            "disagreement": round(st.max_disagreement, 4),
                            "entropy_after": round(st.posterior_entropy_after, 4),
                            "winner": st.winning_hypothesis_idx,
                        }
                        for st in self.active_falsification.steps
                    ],
                }
                if self.active_falsification is not None
                else None
            ),
        }


class ASDSOmegaPipeline:
    """Unified Autonomous Scientific Discovery System — ASDS-OMEGA v2.0 (VECSD)."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def run(
        self,
        dataset: DynamicalDataset,
        run_active_falsification: bool | None = None,
    ) -> DiscoveryResult:
        """Execute the full 6-layer VECSD autonomous scientific discovery pipeline."""
        t0 = time.perf_counter()
        set_global_seed(self.config.seed)

        # Layer 1: Multi-Scale Adaptive Variational Projection + Dimensional Analysis
        projector = MSAVPProjector(self.config.ms_avp)
        substrate = projector.fit_transform(dataset.t_list, dataset.trajectories, dataset.var_names)
        dim_analyzer = DimensionalAnalyzer(dataset.units) if dataset.units else None
        dim_result = dim_analyzer.analyze(dataset.var_names) if dim_analyzer else None

        # Layer 2: Variational Interventional Flow Causality (VIFC)
        causal_engine = VIFCEngine(self.config.vifc, seed=self.config.seed)
        causal_res = causal_engine.discover(substrate, ground_truth_adj=dataset.ground_truth_causal_adj)

        # Layer 3 (Model-Free Pass): W-LIND Invariants, Discrete Symmetries, Divergence, Reynolds
        wlind = WLINDEngine(self.config.wlind)
        sym_res = wlind.discover(substrate, symbolic_equations=None)

        # Layer 4: EVG-HE Symbolic Discovery (V-ESP + Implicit Rational + KKT + Semantic GP + Meta)
        vesp = VESPEngine(self.config.evghe, seed=self.config.seed)
        best_eqs, pareto_fronts, vesp_stats = vesp.discover_system(
            substrate,
            causal_result=causal_res,
            symmetry_result=sym_res,
            dim_analyzer=dim_analyzer,
        )

        meta = ThompsonMetaController(seed=self.config.seed, enabled=self.config.evghe.use_meta_controller)
        rel_noise = float(substrate.metadata.get("relative_noise_est", 0.0))

        if self.config.evghe.enable_gp_refinement:
            gp = SemanticWeakFormGP(self.config.evghe, seed=self.config.seed)
            all_X_obs = np.vstack(dataset.trajectories)
            has_angular_var = any(
                (1.6 < float(np.max(all_X_obs[:, j]) - np.min(all_X_obs[:, j])) < 14.0)
                and float(np.min(all_X_obs[:, j])) < -0.2
                for j in range(dataset.n_vars)
            )
            for i in range(dataset.n_vars):
                dec = meta.decide_refinement(
                    target_var=dataset.var_names[i],
                    vesp_r2=best_eqs[i].weak_r2,
                    rel_noise=rel_noise,
                    base_gp_generations=self.config.evghe.gp_generations,
                )
                if dec.trigger_gp and dec.gp_generations > 0:
                    allowed = causal_res.get_parents(i)
                    gp_cands = gp.evolve_target(
                        substrate,
                        target_idx=i,
                        allowed_vars=allowed,
                        reynolds=sym_res.reynolds_projector,
                        allow_trig=self.config.evghe.include_trig and has_angular_var,
                        generations=dec.gp_generations,
                    )
                    if gp_cands:
                        mdl_before = best_eqs[i].mdl_score
                        # Filter out degenerate constant collapses from GP
                        non_trivial_gp = [
                            c for c in gp_cands
                            if len(c.expr.free_symbols) > 0 and c.weak_r2 >= best_eqs[i].weak_r2 - 0.12
                        ]
                        if non_trivial_gp:
                            pareto_fronts[i].extend(non_trivial_gp)
                            pareto_fronts[i].sort(key=lambda c: c.mdl_score)
                            if non_trivial_gp[0].mdl_score < best_eqs[i].mdl_score:
                                best_eqs[i] = non_trivial_gp[0]
                        meta.update_reward("srg_gp_semantic", mdl_before, best_eqs[i].mdl_score)

        # Layer 3 (Exact Symbolic Refinement Pass): Analytical Divergence & Exact Lie Prolongation
        discovered_sympy = [c.expr for c in best_eqs]
        if self.config.wlind.enabled:
            sym_res.regime_profile = analyze_divergence_and_chaos(substrate, symbolic_equations=discovered_sympy)
            exact_lies = discover_lie_symmetries(
                substrate,
                symbolic_equations=discovered_sympy,
                sv_threshold=self.config.wlind.lie_sv_threshold,
            )
            if exact_lies:
                sym_res.lie_generators = exact_lies
                sym_res.lie_closure = verify_lie_closure(exact_lies)

        # Layer 5: Evidence Scoring & Optional Popperian Active Falsification (PAFE)
        evidence = score_discovered_system(
            best_eqs,
            causal_res,
            sym_res,
            dataset.var_names,
            ground_truth_equations=dataset.ground_truth_equations,
        )

        do_active = (
            run_active_falsification
            if run_active_falsification is not None
            else self.config.scientist.enable_active_falsification
        )
        active_res: ActiveFalsificationResult | None = None
        if do_active and dataset.rhs_fn is not None:
            falsifier = PopperianActiveFalsifier(
                seed=self.config.seed,
                n_candidates=self.config.scientist.active_n_candidates,
            )
            active_res = falsifier.run_campaign(
                dataset=dataset,
                pareto_per_var=pareto_fronts,
                linear_conservation=sym_res.linear_conservation_matrix,
                budget=self.config.scientist.active_budget,
                noise_std=float(dataset.metadata.get("noise_level", 0.05)),
            )

        # Layer 6: Formal Symbolic Verification (SymPy Prover + Lean 4 Proof Certificate)
        prover = SymPyFormalProver(dataset.var_names)
        proof_cert = prover.prove_system(discovered_sympy, sym_res, np.vstack(dataset.trajectories))
        lean_cert = (
            export_lean4_certificate(
                dataset.name,
                dataset.var_names,
                discovered_sympy,
                sym_res,
                proof_cert,
            )
            if self.config.scientist.export_lean4
            else None
        )

        runtime = time.perf_counter() - t0
        return DiscoveryResult(
            dataset_name=dataset.name,
            domain=dataset.domain,
            var_names=dataset.var_names,
            config_hash=self.config.config_hash(),
            seed=self.config.seed,
            runtime_sec=runtime,
            substrate=substrate,
            dimensional=dim_result,
            causal=causal_res,
            symmetry=sym_res,
            equations=best_eqs,
            pareto_fronts=pareto_fronts,
            evidence=evidence,
            verification=proof_cert,
            lean4_certificate=lean_cert,
            active_falsification=active_res,
            meta_summary=meta.summary(),
            pruning_ratio=vesp_stats["mean_pruning_ratio"],
            kkt_conservation_enforced=bool(vesp_stats["kkt_conservation_enforced"]),
        )
