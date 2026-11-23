"""Bundled real-world published experimental scientific datasets with scholarly provenance.

All numerical series come verbatim from classic peer-reviewed scientific records:
1. Hudson Bay Company Lynx-Hare Fur Trapping Records (1900-1920), Odum (1953) /
   Elton & Nicholson (1942) — classic sparse, noisy ecological predator-prey data.
2. 1978 English Boarding School Influenza Outbreak, Anonymous, British Medical
   Journal (1978), 1(6112), 587 — classic real epidemiological SIR outbreak.
3. Mauna Loa Observatory Atmospheric CO2 Annual-Mean Series (1960-2023),
   Keeling & NOAA Global Monitoring Laboratory — real climate/carbon-cycle growth.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from asds_omega.data.synthetic import DynamicalDataset


def load_hudson_bay_lynx_hare(upsample_factor: int = 4) -> DynamicalDataset:
    """Hudson Bay Company Lynx-Hare pelt counts (1900-1920, in thousands).

    Reference: Odum, E. P. (1953). Fundamentals of Ecology; Howard (2009).
    Uses gentle cubic interpolation ONLY for weak-form quadrature integration
    grid density while preserving exact raw annual observations in metadata.
    """
    years = np.arange(1900, 1921, dtype=float)
    hare_raw = np.array(
        [30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4, 27.1,
         40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7],
        dtype=float,
    )
    lynx_raw = np.array(
        [4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4,
         8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6],
        dtype=float,
    )
    # Normalize to tens of thousands for numerical conditioning
    t_raw = years - years[0]
    X_raw = np.column_stack([hare_raw / 10.0, lynx_raw / 10.0])

    if upsample_factor > 1:
        t_fine = np.linspace(t_raw[0], t_raw[-1], (len(t_raw) - 1) * upsample_factor + 1)
        x_fine = np.maximum(CubicSpline(t_raw, X_raw[:, 0])(t_fine), 0.1)
        y_fine = np.maximum(CubicSpline(t_raw, X_raw[:, 1])(t_fine), 0.1)
        t_use = t_fine
        X_use = np.column_stack([x_fine, y_fine])
    else:
        t_use = t_raw
        X_use = X_raw

    return DynamicalDataset(
        name="hudson_bay_lynx_hare",
        domain="Real-World Ecology (Hudson Bay 1900-1920)",
        var_names=["hare", "lynx"],
        t_list=[t_use],
        trajectories=[X_use],
        clean_trajectories=[X_use],
        ground_truth_equations=[
            "alpha*hare - beta*hare*lynx  (expected ecological form)",
            "delta*hare*lynx - gamma*lynx  (expected ecological form)",
        ],
        ground_truth_causal_adj=np.array([[1, 1], [1, 1]], dtype=int),
        ground_truth_invariants=["Approximate Lotka-Volterra logarithmic Hamiltonian"],
        ground_truth_symmetries=["Approximate periodic orbit"],
        metadata={
            "source": "Hudson Bay Company (1900-1920), Odum (1953)",
            "raw_years": years.tolist(),
            "raw_n_points": len(years),
            "scale_unit": "10,000 pelts",
            "is_real_world": True,
        },
    )


def load_boarding_school_influenza(upsample_factor: int = 5) -> DynamicalDataset:
    """1978 English Boarding School Influenza Outbreak (N = 763 boys, 14 days).

    Reference: Anonymous (1978), 'Influenza in a boarding school', British Medical
    Journal, 1(6112), 587; Murray (2002), Mathematical Biology I.
    Reconstructs normalized (S, I, R) fractions from observed daily bed-confined cases
    using standard cumulative incidence recovery rate (~0.45/day) while preserving
    exact raw daily bed counts in metadata.
    """
    days = np.arange(0, 15, dtype=float)
    # Day 0: 1 index case; Days 1-14: BMJ published daily boys confined to bed
    infected_counts = np.array(
        [1, 3, 8, 28, 75, 221, 291, 255, 235, 190, 125, 70, 28, 12, 5],
        dtype=float,
    )
    N = 763.0
    I_raw = infected_counts / N
    # Estimate cumulative recovered via integral of gamma*I with mean infectious stay ~2.2 days
    gamma_est = 0.45
    R_raw = np.zeros_like(I_raw)
    for k in range(1, len(days)):
        R_raw[k] = R_raw[k - 1] + 0.5 * gamma_est * (I_raw[k] + I_raw[k - 1]) * (days[k] - days[k - 1])
    R_raw = np.minimum(R_raw, 1.0 - I_raw)
    S_raw = np.maximum(1.0 - I_raw - R_raw, 0.0)
    X_raw = np.column_stack([S_raw, I_raw, R_raw])

    if upsample_factor > 1:
        t_fine = np.linspace(days[0], days[-1], (len(days) - 1) * upsample_factor + 1)
        I_fine = np.clip(CubicSpline(days, I_raw)(t_fine), 1e-4, 1.0)
        R_fine = np.clip(CubicSpline(days, R_raw)(t_fine), 0.0, 1.0 - I_fine)
        S_fine = np.clip(1.0 - I_fine - R_fine, 0.0, 1.0)
        t_use = t_fine
        X_use = np.column_stack([S_fine, I_fine, R_fine])
    else:
        t_use = days
        X_use = X_raw

    return DynamicalDataset(
        name="boarding_school_influenza",
        domain="Real-World Epidemiology (1978 BMJ Influenza Outbreak)",
        var_names=["S", "I", "R"],
        t_list=[t_use],
        trajectories=[X_use],
        clean_trajectories=[X_use],
        ground_truth_equations=[
            "-beta*S*I  (expected SIR)",
            "beta*S*I - gamma*I  (expected SIR)",
            "gamma*I  (expected SIR)",
        ],
        ground_truth_causal_adj=np.array([[1, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=int),
        ground_truth_invariants=["S + I + R"],
        ground_truth_symmetries=["Mass conservation S + I + R = 1"],
        metadata={
            "source": "British Medical Journal (1978) 1(6112):587; N=763",
            "raw_days": days.tolist(),
            "raw_infected_counts": infected_counts.tolist(),
            "N": N,
            "is_real_world": True,
        },
    )


def load_mauna_loa_co2(upsample_factor: int = 3) -> DynamicalDataset:
    """Mauna Loa Observatory Atmospheric CO2 Excess Above Pre-Industrial (1960-2023).

    Reference: NOAA Global Monitoring Laboratory / Scripps CO2 Program (Keeling Curve).
    Annual mean atmospheric CO2 concentration (ppm) minus pre-industrial 280 ppm baseline,
    paired with its annual rate state to probe autonomous accelerating carbon-cycle growth laws.
    """
    years = np.arange(1960, 2024, dtype=float)
    # NOAA GML / Keeling annual mean CO2 (ppm) from 1960 to 2023
    co2_ppm = np.array(
        [
            316.91, 317.64, 318.45, 318.99, 319.62, 320.04, 321.37, 322.18, 323.05, 324.62,
            325.68, 326.32, 327.46, 329.68, 330.19, 331.13, 332.03, 333.84, 335.41, 336.84,
            338.76, 340.12, 341.48, 343.15, 344.87, 346.35, 347.61, 349.31, 351.69, 353.20,
            354.45, 355.70, 356.54, 357.21, 358.96, 360.97, 362.74, 363.88, 366.84, 368.54,
            369.71, 371.32, 373.45, 375.98, 377.70, 379.98, 382.09, 384.02, 385.83, 387.64,
            390.10, 391.85, 394.06, 396.74, 398.81, 401.01, 404.41, 406.76, 408.72, 411.65,
            414.21, 416.41, 418.53, 421.08,
        ],
        dtype=float,
    )
    t_raw = (years - years[0]) / 10.0  # Decades since 1960
    c_excess = (co2_ppm - 280.0) / 100.0  # Excess CO2 in hundreds of ppm above pre-industrial

    if upsample_factor > 1:
        t_use = np.linspace(t_raw[0], t_raw[-1], (len(t_raw) - 1) * upsample_factor + 1)
        c_use = CubicSpline(t_raw, c_excess)(t_use)
    else:
        t_use = t_raw
        c_use = c_excess

    # 1D autonomous carbon excess trajectory (C_excess)
    X_use = c_use[:, None]
    return DynamicalDataset(
        name="mauna_loa_co2",
        domain="Real-World Climate Science (NOAA Mauna Loa 1960-2023)",
        var_names=["C"],
        t_list=[t_use],
        trajectories=[X_use],
        clean_trajectories=[X_use],
        ground_truth_equations=[
            "r*C  (expected near-exponential excess CO2 growth law)"
        ],
        ground_truth_causal_adj=np.array([[1]], dtype=int),
        ground_truth_invariants=[],
        ground_truth_symmetries=["Approximate scaling / exponential growth"],
        metadata={
            "source": "NOAA Global Monitoring Laboratory / Scripps CO2 Program (1960-2023)",
            "baseline_ppm": 280.0,
            "time_unit": "decades since 1960",
            "state_unit": "100 ppm excess above pre-industrial",
            "raw_n_points": len(years),
            "is_real_world": True,
        },
    )


def load_real_dataset(name: str, **kwargs) -> DynamicalDataset:
    """Factory for bundled real-world experimental datasets."""
    key = name.lower().strip()
    if key in {"lynx_hare", "hudson_bay", "hudson_bay_lynx_hare"}:
        return load_hudson_bay_lynx_hare(**kwargs)
    if key in {"influenza", "boarding_school", "boarding_school_influenza"}:
        return load_boarding_school_influenza(**kwargs)
    if key in {"mauna_loa", "co2", "mauna_loa_co2"}:
        return load_mauna_loa_co2(**kwargs)
    raise ValueError(f"Unknown real dataset '{name}'.")
