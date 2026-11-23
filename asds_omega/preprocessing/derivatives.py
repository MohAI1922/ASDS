"""Classical numerical derivative estimators (Savitzky-Golay and central differences) for ablation."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def estimate_derivatives_central(t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Second-order central finite differences (non-uniform spacing safe)."""
    return np.gradient(X, t, axis=0, edge_order=2)


def estimate_derivatives_savgol(
    t: np.ndarray,
    X: np.ndarray,
    window_length: int | None = None,
    polyorder: int = 3,
) -> np.ndarray:
    """Savitzky-Golay smoothed derivative estimator."""
    n = len(t)
    dt = float(np.median(np.diff(t)))
    if window_length is None:
        win = min(max(7, (n // 25) | 1), 31)
    else:
        win = int(window_length) | 1
    win = min(win, n if n % 2 == 1 else n - 1)
    if win < 5:
        return estimate_derivatives_central(t, X)
    order = min(polyorder, win - 2)
    out = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        out[:, j] = savgol_filter(X[:, j], window_length=win, polyorder=order, deriv=1, delta=dt)
    return out
