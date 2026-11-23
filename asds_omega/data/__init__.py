"""Synthetic ground-truth dynamical benchmarks, real experimental datasets, and perturbations."""

from asds_omega.data.perturbations import (
    add_gaussian_noise,
    add_outlier_noise,
    add_periodic_bias,
    add_uniform_noise,
    inject_noise,
)
from asds_omega.data.real_world import (
    load_boarding_school_influenza,
    load_hudson_bay_lynx_hare,
    load_mauna_loa_co2,
    load_real_dataset,
)
from asds_omega.data.synthetic import (
    DynamicalDataset,
    PDEDataset,
    gen_advection_diffusion_pde,
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
    generate_synthetic_dataset,
)

__all__ = [
    "DynamicalDataset",
    "PDEDataset",
    "gen_lorenz",
    "gen_lotka_volterra",
    "gen_sir",
    "gen_sir_regime_drift",
    "gen_pendulum",
    "gen_kepler_2body",
    "gen_michaelis_menten",
    "gen_duffing",
    "gen_van_der_pol",
    "gen_burgers_pde",
    "gen_advection_diffusion_pde",
    "generate_synthetic_dataset",
    "load_hudson_bay_lynx_hare",
    "load_boarding_school_influenza",
    "load_mauna_loa_co2",
    "load_real_dataset",
    "add_gaussian_noise",
    "add_uniform_noise",
    "add_outlier_noise",
    "add_periodic_bias",
    "inject_noise",
]
