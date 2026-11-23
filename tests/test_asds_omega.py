"""Comprehensive unit, mathematical-property, integration, and determinism tests for ASDS-OMEGA v2.0."""

from __future__ import annotations

import numpy as np
import sympy as sp

from asds_omega.causal import CounterfactualSimulator, VIFCEngine
from asds_omega.config import PipelineConfig
from asds_omega.data import (
    gen_burgers_pde,
    gen_duffing,
    gen_kepler_2body,
    gen_lorenz,
    gen_lotka_volterra,
    gen_michaelis_menten,
    gen_pendulum,
    gen_sir,
    gen_sir_regime_drift,
    gen_van_der_pol,
    inject_noise,
    load_boarding_school_influenza,
    load_hudson_bay_lynx_hare,
    load_mauna_loa_co2,
)
from asds_omega.preprocessing import (
    DimensionalAnalyzer,
    MSAVPProjector,
    estimate_noise_wavelet_mad,
    hermite_poly_prob,
)
from asds_omega.scientist import ASDSOmegaPipeline, SurgicalDriftRegenerator
from asds_omega.symmetry import WLINDEngine
from asds_omega.symbolic import discover_pde_vesp


# ------------------------------------------------------------------
# Layer 1: MS-AVP & Hermite Cumulant Debiasing Mathematical Theorems
# ------------------------------------------------------------------
def test_hermite_cumulant_unbiasedness():
    """Verify E[H_n(X + eps; sigma^2)] == X^n for degrees 2, 3, 4 under high Gaussian noise."""
    rng = np.random.default_rng(123)
    x_true = np.linspace(-2.0, 2.0, 20000)
    sigma = 0.6
    x_noisy = x_true + rng.normal(0.0, sigma, size=x_true.shape)
    for deg in (2, 3, 4):
        biased_err = abs(float(np.mean(x_noisy**deg - x_true**deg)))
        debiased_err = abs(float(np.mean(hermite_poly_prob(x_noisy, deg, sigma**2) - x_true**deg)))
        assert debiased_err < 0.12 * max(biased_err, 0.1), f"Deg {deg}: debiased {debiased_err} vs biased {biased_err}"


def test_wavelet_mad_noise_estimator():
    ds = inject_noise(gen_lorenz(n_trajectories=2, seed=7), 0.10, seed=7)
    est = estimate_noise_wavelet_mad(ds.trajectories)
    true_std = 0.10 * np.std(np.vstack(ds.clean_trajectories), axis=0)
    rel_err = np.abs(est - true_std) / true_std
    assert np.all(rel_err < 0.25), f"Wavelet-MAD relative errors too high: {rel_err}"


def test_buckingham_pi_dimensional_analysis():
    kep = gen_kepler_2body()
    da = DimensionalAnalyzer(kep.units)
    res = da.analyze(kep.var_names)
    assert res is not None
    assert res.n_pi_groups == 2
    # Transcendental validity: theta (dimensionless) allowed, x (Length) rejected
    pen = gen_pendulum()
    da_pen = DimensionalAnalyzer(pen.units)
    assert da_pen.is_transcendentally_valid(sp.sin(sp.Symbol("theta")))
    assert not da_pen.is_transcendentally_valid(sp.sin(sp.Symbol("omega")))


# ------------------------------------------------------------------
# Layer 2: VIFC Direct Differential Causal Discovery
# ------------------------------------------------------------------
def test_vifc_exact_cyclic_causality_lorenz_and_pendulum():
    for gen_fn in (gen_lorenz, gen_pendulum, gen_lotka_volterra):
        ds = inject_noise(gen_fn(seed=42), 0.05, seed=42)
        sub = MSAVPProjector().fit_transform(ds.t_list, ds.trajectories, ds.var_names)
        cres = VIFCEngine(seed=42).discover(sub, ground_truth_adj=ds.ground_truth_causal_adj)
        assert cres.metrics["shd"] == 0.0, f"{ds.name} expected SHD=0, got {cres.metrics}"


# ------------------------------------------------------------------
# Layer 3: W-LIND Model-Free Symmetries, Invariants & Divergence
# ------------------------------------------------------------------
def test_wlind_lorenz_z2_and_dissipative():
    ds = inject_noise(gen_lorenz(seed=42), 0.05, seed=42)
    sub = MSAVPProjector().fit_transform(ds.t_list, ds.trajectories, ds.var_names)
    sres = WLINDEngine().discover(sub)
    signs_list = [s.signs for s in sres.discrete_symmetries]
    assert (-1, -1, 1) in signs_list, f"Expected Lorenz Z2 (-1,-1,1), got {signs_list}"
    assert sres.regime_profile.mean_divergence < 0.0


def test_wlind_sir_linear_conservation():
    ds = inject_noise(gen_sir(seed=42), 0.02, seed=42)
    sub = MSAVPProjector().fit_transform(ds.t_list, ds.trajectories, ds.var_names)
    sres = WLINDEngine().discover(sub)
    assert sres.linear_conservation_matrix is not None
    w = sres.linear_conservation_matrix[0]
    w_norm = w / w[0]
    assert np.allclose(w_norm, np.ones(3), atol=0.1)


# ------------------------------------------------------------------
# Layer 4-6: End-to-End Symbolic Discovery, Formal Prover & Lean 4
# ------------------------------------------------------------------
def test_end_to_end_pendulum_and_kepler_exact_recovery():
    pipe = ASDSOmegaPipeline(PipelineConfig(seed=42))
    for gen_fn in (gen_pendulum, gen_kepler_2body):
        ds = inject_noise(gen_fn(seed=42), 0.05, seed=42)
        res = pipe.run(ds, run_active_falsification=False)
        assert res.evidence.symbolic_match_score is not None
        assert res.evidence.symbolic_match_score >= 0.98
        assert res.evidence.exact_structure_rate == 1.0
        assert res.verification.symbolic_divergence == "0"
        assert res.lean4_certificate is not None and "namespace ASDSOmega." in res.lean4_certificate


def test_end_to_end_lorenz_and_sir_and_michaelis_menten():
    pipe = ASDSOmegaPipeline(PipelineConfig(seed=42))
    ds_lor = inject_noise(gen_lorenz(seed=42), 0.05, seed=42)
    r_lor = pipe.run(ds_lor, run_active_falsification=False)
    assert r_lor.evidence.symbolic_match_score is not None and r_lor.evidence.symbolic_match_score >= 0.90
    assert r_lor.pruning_ratio >= 0.50

    ds_sir = inject_noise(gen_sir(seed=42), 0.05, seed=42)
    r_sir = pipe.run(ds_sir, run_active_falsification=False)
    assert r_sir.evidence.symbolic_match_score is not None and r_sir.evidence.symbolic_match_score >= 0.85

    ds_mm = inject_noise(gen_michaelis_menten(seed=42), 0.05, seed=42)
    r_mm = pipe.run(ds_mm, run_active_falsification=False)
    assert r_mm.evidence.mean_weak_r2 >= 0.97


def test_spacetime_weak_burgers_pde():
    pds = inject_noise(gen_burgers_pde(), 0.05, seed=42)
    cand = discover_pde_vesp(pds.x, pds.t, pds.u, cumulant_debiasing=True, seed=42)
    assert cand.weak_r2 >= 0.97
    assert "u*u_x" in cand.coefficients and "u_xx" in cand.coefficients
    assert abs(cand.coefficients["u*u_x"] + 1.0) < 0.08
    assert abs(cand.coefficients["u_xx"] - 0.15) < 0.03


def test_active_falsification_and_counterfactual_simulator():
    ds = inject_noise(gen_sir(n_trajectories=1, seed=42), 0.05, seed=42)
    cfg = PipelineConfig(seed=42)
    cfg.scientist.active_budget = 2
    res = ASDSOmegaPipeline(cfg).run(ds, run_active_falsification=True)
    assert res.active_falsification is not None
    assert len(res.active_falsification.final_posteriors) >= 1

    sim = CounterfactualSimulator(ds.var_names, [c.expr for c in res.equations])
    cf = sim.run_intervention(ds.clean_trajectories[0][0], ds.t_list[0][:100], scale_equation={"S": 0.5})
    assert cf.baseline_trajectory.shape == (100, 3)


def test_online_drift_detection_and_regeneration():
    dds = inject_noise(gen_sir_regime_drift(), 0.02, seed=42)
    dres = SurgicalDriftRegenerator(seed=42).run_stream(
        dds.t_list[0], dds.trajectories[0], dds.var_names, true_shift_time=20.0
    )
    assert dres.drift_detected
    assert dres.regenerated_r2_on_post > dres.pre_drift_r2_on_post


def test_real_world_datasets_run_cleanly():
    pipe = ASDSOmegaPipeline(PipelineConfig(seed=42))
    for loader in (load_hudson_bay_lynx_hare, load_boarding_school_influenza, load_mauna_loa_co2):
        rds = loader()
        r = pipe.run(rds, run_active_falsification=False)
        assert r.evidence.mean_weak_r2 >= 0.70
        assert r.evidence.overall_confidence >= 0.75


def test_deterministic_reproducibility():
    ds = inject_noise(gen_pendulum(seed=99), 0.05, seed=99)
    r1 = ASDSOmegaPipeline(PipelineConfig(seed=99)).run(ds, run_active_falsification=False)
    r2 = ASDSOmegaPipeline(PipelineConfig(seed=99)).run(ds, run_active_falsification=False)
    assert [c.expr_str for c in r1.equations] == [c.expr_str for c in r2.equations]
    assert np.array_equal(r1.causal.adjacency_matrix, r2.causal.adjacency_matrix)


def test_cubic_oscillators_cumulant_debiasing():
    """Verify exact Z2 + cubic recovery on Van der Pol and Duffing under high noise."""
    from asds_omega.config import EVGHEConfig, MSAVPConfig

    ds_vdp = inject_noise(gen_van_der_pol(seed=42), 0.20, seed=42)
    r_deb = ASDSOmegaPipeline(PipelineConfig(seed=42, evghe=EVGHEConfig(enable_gp_refinement=False))).run(
        ds_vdp, run_active_falsification=False
    )
    r_nd = ASDSOmegaPipeline(
        PipelineConfig(seed=42, ms_avp=MSAVPConfig(cumulant_debiasing=False), evghe=EVGHEConfig(enable_gp_refinement=False))
    ).run(ds_vdp, run_active_falsification=False)
    assert r_deb.evidence.symbolic_match_score is not None and r_deb.evidence.symbolic_match_score >= 0.98
    assert (r_deb.evidence.mean_coeff_rel_error or 1.0) <= (r_nd.evidence.mean_coeff_rel_error or 1.0) + 1e-6

    ds_duf = inject_noise(gen_duffing(seed=42), 0.15, seed=42)
    r_duf = ASDSOmegaPipeline(PipelineConfig(seed=42, evghe=EVGHEConfig(enable_gp_refinement=False))).run(
        ds_duf, run_active_falsification=False
    )
    assert r_duf.evidence.symbolic_match_score is not None and r_duf.evidence.symbolic_match_score >= 0.95
