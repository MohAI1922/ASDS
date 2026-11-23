"""Reproducible noise models (Gaussian, uniform, outlier, periodic) for robustness tests."""

from __future__ import annotations

import copy
import numpy as np

from asds_omega.data.synthetic import DynamicalDataset, PDEDataset


def add_gaussian_noise(
    X: np.ndarray,
    noise_level: float,
    rng: np.random.Generator,
    relative: bool = True,
) -> np.ndarray:
    """Inject zero-mean Gaussian measurement noise scaled relative to per-channel std."""
    if noise_level <= 0.0:
        return X.copy()
    if relative:
        scale = np.maximum(np.std(X, axis=0, keepdims=True), 1e-6)
    else:
        scale = 1.0
    return X + rng.normal(0.0, noise_level * scale, size=X.shape)


def add_uniform_noise(
    X: np.ndarray,
    noise_level: float,
    rng: np.random.Generator,
    relative: bool = True,
) -> np.ndarray:
    """Inject uniform bounded measurement noise with matched standard deviation."""
    if noise_level <= 0.0:
        return X.copy()
    scale = np.maximum(np.std(X, axis=0, keepdims=True), 1e-6) if relative else 1.0
    half_width = np.sqrt(3.0) * noise_level * scale
    return X + rng.uniform(-half_width, half_width, size=X.shape)


def add_outlier_noise(
    X: np.ndarray,
    noise_level: float,
    rng: np.random.Generator,
    outlier_frac: float = 0.03,
    outlier_mult: float = 5.0,
) -> np.ndarray:
    """Inject Gaussian baseline noise plus sparse heavy-tailed sensor spikes."""
    noisy = add_gaussian_noise(X, noise_level, rng, relative=True)
    if noise_level <= 0.0 or outlier_frac <= 0.0:
        return noisy
    scale = np.maximum(np.std(X, axis=0, keepdims=True), 1e-6)
    mask = rng.random(size=X.shape) < outlier_frac
    spikes = rng.normal(0.0, outlier_mult * noise_level * scale, size=X.shape)
    return noisy + mask * spikes


def add_periodic_bias(
    X: np.ndarray,
    t: np.ndarray,
    noise_level: float,
    freq: float = 12.0,
) -> np.ndarray:
    """Inject deterministic high-frequency periodic sensor interference."""
    if noise_level <= 0.0:
        return X.copy()
    scale = np.maximum(np.std(X, axis=0, keepdims=True), 1e-6)
    wave = np.sin(2.0 * np.pi * freq * t)[:, None]
    return X + (noise_level * scale) * wave


def inject_noise(
    dataset: DynamicalDataset | PDEDataset,
    noise_level: float,
    noise_type: str = "gaussian",
    seed: int = 42,
) -> DynamicalDataset | PDEDataset:
    """Return a deep-copied dataset with controlled noise applied across trajectories."""
    rng = np.random.default_rng(seed)
    ds = copy.deepcopy(dataset)
    if isinstance(ds, PDEDataset):
        if noise_level > 0.0:
            scale = max(float(np.std(ds.u_clean)), 1e-6)
            ds.u = ds.u_clean + rng.normal(0.0, noise_level * scale, size=ds.u_clean.shape)
        ds.metadata["noise_level"] = noise_level
        ds.metadata["noise_type"] = noise_type
        return ds

    noisy_trajs: list[np.ndarray] = []
    for t_arr, X_clean in zip(ds.t_list, ds.clean_trajectories):
        if noise_type == "gaussian":
            Xn = add_gaussian_noise(X_clean, noise_level, rng)
        elif noise_type == "uniform":
            Xn = add_uniform_noise(X_clean, noise_level, rng)
        elif noise_type == "outliers":
            Xn = add_outlier_noise(X_clean, noise_level, rng)
        elif noise_type == "periodic":
            Xn = add_periodic_bias(X_clean, t_arr, noise_level)
        else:
            raise ValueError(f"Unsupported noise_type '{noise_type}'.")
        noisy_trajs.append(Xn)
    ds.trajectories = noisy_trajs
    ds.metadata["noise_level"] = noise_level
    ds.metadata["noise_type"] = noise_type
    return ds
