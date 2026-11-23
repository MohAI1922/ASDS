"""Layer 1: Multi-Scale Adaptive Variational Projection (MS-AVP) with Cumulant EIV Debiasing.

Novel algorithmic contributions:
1. Wavelet-MAD per-channel noise variance estimation directly from raw noisy trajectories.
2. Locally adaptive multi-scale test-function support selection W_k* balancing local
   signal curvature truncation error against observational noise variance.
3. Analytical Hermite / Wick cumulant debiasing of nonlinear weak-form inner products,
   eliminating Errors-in-Variables (EIV) attenuation bias in polynomial and trigonometric
   features under high measurement noise (sigma = 10-30%).
4. Unified 1D ODE and 1+1D PDE spacetime integration-by-parts projection substrate
   shared across causal discovery (VIFC), symmetry/invariant discovery (W-LIND), and
   symbolic evolution (EVG-HE).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.signal import savgol_filter

from asds_omega.config import MSAVPConfig
from asds_omega.preprocessing.derivatives import estimate_derivatives_savgol


def estimate_noise_wavelet_mad(trajectories: list[np.ndarray]) -> np.ndarray:
    """Estimate per-channel additive noise std via Donoho-Johnstone Haar wavelet MAD.

    For a smooth sampled signal corrupted by white noise N(0, sigma^2), second-order
    finite differences d2[k] = (x[k+1] - 2*x[k] + x[k-1]) / sqrt(6) annihilate local
    linear trends while preserving unit noise variance, making MAD highly accurate even
    on rapidly varying chaotic trajectories.
    """
    d = trajectories[0].shape[1]
    sigmas = np.zeros(d, dtype=float)
    for j in range(d):
        details = []
        for X in trajectories:
            if len(X) >= 5:
                # Normalized 2nd-order wavelet-like detail (annihilates smooth linear trends)
                d2 = (X[2:, j] - 2.0 * X[1:-1, j] + X[:-2, j]) / np.sqrt(6.0)
                details.append(d2)
            elif len(X) >= 2:
                d1 = np.diff(X[:, j]) / np.sqrt(2.0)
                details.append(d1)
        if details:
            cat = np.concatenate(details)
            mad = np.median(np.abs(cat - np.median(cat)))
            sigmas[j] = float(mad / 0.67448975)
    return sigmas


def bump_test_function(
    t_window: np.ndarray,
    poly_order: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct compactly supported C^p bump test function phi(t) and d(phi)/dt, d2(phi)/dt2.

    Defined on [t_a, t_b] via u = (t - t_mid) / h in [-1, 1]:
        phi(u) = (1 - u^2)^p
        phi'(t) = (-2p / h) * u * (1 - u^2)^(p-1)
        phi''(t) = (2p / h^2) * (1 - u^2)^(p-2) * ((2p - 1) u^2 - 1)
    Normalized such that integral_{t_a}^{t_b} phi(t) dt = 1.
    """
    t_a, t_b = float(t_window[0]), float(t_window[-1])
    h = 0.5 * (t_b - t_a)
    if h <= 0.0:
        raise ValueError("Test-function window must have positive duration.")
    t_mid = 0.5 * (t_a + t_b)
    u = np.clip((t_window - t_mid) / h, -1.0, 1.0)
    one_minus_u2 = np.maximum(1.0 - u * u, 0.0)

    p = max(int(poly_order), 2)
    phi = one_minus_u2**p
    dphi = (-2.0 * p / h) * u * (one_minus_u2 ** (p - 1))
    d2phi = (2.0 * p / (h * h)) * (one_minus_u2 ** (p - 2)) * ((2.0 * p - 1.0) * u * u - 1.0)

    # Trapezoidal integration weights on t_window
    w_trap = np.zeros_like(t_window)
    dt = np.diff(t_window)
    w_trap[0] = 0.5 * dt[0]
    w_trap[-1] = 0.5 * dt[-1]
    w_trap[1:-1] = 0.5 * (dt[:-1] + dt[1:])

    norm = float(np.sum(w_trap * phi))
    if norm <= 1e-15:
        raise ValueError("Degenerate bump test function integral.")
    phi /= norm
    dphi /= norm
    d2phi /= norm
    return phi * w_trap, dphi * w_trap, d2phi * w_trap


def hermite_poly_prob(x: np.ndarray, degree: int, variance: float) -> np.ndarray:
    """Evaluate scaled probabilists' Hermite polynomial H_n(x; v) with E[H_n(X+eps; v)] = X^n."""
    if degree == 0:
        return np.ones_like(x, dtype=float)
    if degree == 1 or variance <= 0.0:
        return np.power(x, degree)
    h_prev2 = np.ones_like(x, dtype=float)
    h_prev1 = x.astype(float, copy=False)
    for k in range(1, degree):
        h_curr = x * h_prev1 - (k * variance) * h_prev2
        h_prev2, h_prev1 = h_prev1, h_curr
    return h_prev1


def hermite_debias_monomial(
    X: np.ndarray,
    powers: tuple[int, ...],
    noise_vars: np.ndarray,
    enabled: bool = True,
) -> np.ndarray:
    """Compute analytically EIV-debiased monomial prod_j x_j^{p_j} via factor-wise Hermite expansion."""
    out = np.ones(X.shape[0], dtype=float)
    for j, deg in enumerate(powers):
        if deg == 0:
            continue
        var_j = float(noise_vars[j]) if enabled else 0.0
        out = out * hermite_poly_prob(X[:, j], deg, var_j)
    return out


def debias_trig_feature(
    x_col: np.ndarray,
    kind: str,
    noise_var: float,
    enabled: bool = True,
    max_gain: float = 1.65,
) -> np.ndarray:
    """Characteristic-function EIV debiasing for sin(x) and cos(x): E[sin(x+e)] = exp(-v/2) sin(x)."""
    raw = np.sin(x_col) if kind == "sin" else np.cos(x_col)
    if not enabled or noise_var <= 0.0:
        return raw
    gain = float(min(np.exp(0.5 * noise_var), max_gain))
    return gain * raw


@dataclass
class SubdomainSpec:
    """Specification of a single variational quadrature window."""

    traj_idx: int
    start_idx: int
    end_idx: int
    center_time: float
    w_phi: np.ndarray  # Quadrature weights * phi(t)
    w_dphi: np.ndarray  # Quadrature weights * phi'(t)
    w_d2phi: np.ndarray  # Quadrature weights * phi''(t)


@dataclass
class WeakFormSubstrate:
    """Unified variational substrate shared across VIFC, W-LIND, and EVG-HE."""

    t_list: list[np.ndarray]
    trajectories: list[np.ndarray]
    var_names: list[str]
    noise_stds: np.ndarray  # Estimated per-channel noise std
    noise_vars: np.ndarray  # Estimated per-channel noise variance
    subdomains: list[SubdomainSpec]
    weak_velocities: np.ndarray  # (K, d): -<phi'_k, x_i>
    weak_states: np.ndarray  # (K, d): <phi_k, x_i>
    traj_ids: np.ndarray  # (K,) trajectory index for each subdomain
    window_widths: np.ndarray  # (K,) adaptive support widths in samples
    use_weak_form: bool = True
    cumulant_debiasing: bool = True
    metadata: dict = field(default_factory=dict)

    @property
    def n_subdomains(self) -> int:
        return int(self.weak_velocities.shape[0])

    @property
    def n_vars(self) -> int:
        return len(self.var_names)

    def project_pointwise(self, feature_trajs: list[np.ndarray]) -> np.ndarray:
        """Project trajectory-level feature arrays F_r of shape (T_r, M) -> weak matrix (K, M)."""
        if not self.use_weak_form:
            return np.vstack(feature_trajs)
        m = feature_trajs[0].shape[1]
        G = np.empty((len(self.subdomains), m), dtype=float)
        for k, sub in enumerate(self.subdomains):
            F_seg = feature_trajs[sub.traj_idx][sub.start_idx : sub.end_idx]
            G[k] = sub.w_phi @ F_seg
        return G

    def project_derivative_of_features(self, feature_trajs: list[np.ndarray]) -> np.ndarray:
        """Compute <phi'_k, F_m(x)> of shape (K, M) for model-free conservation discovery."""
        if not self.use_weak_form:
            # Finite-difference fallback for ablation
            derivs = []
            for t_arr, F_arr in zip(self.t_list, feature_trajs):
                derivs.append(-estimate_derivatives_savgol(t_arr, F_arr))
            return np.vstack(derivs)
        m = feature_trajs[0].shape[1]
        A = np.empty((len(self.subdomains), m), dtype=float)
        for k, sub in enumerate(self.subdomains):
            F_seg = feature_trajs[sub.traj_idx][sub.start_idx : sub.end_idx]
            A[k] = sub.w_dphi @ F_seg
        return A

    def project_callable(self, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Evaluate pointwise function fn(X) -> (T,) or (T, M) and project onto weak basis."""
        feats = []
        for X in self.trajectories:
            val = np.asarray(fn(X), dtype=float)
            if val.ndim == 1:
                val = val[:, None]
            feats.append(val)
        proj = self.project_pointwise(feats)
        return proj[:, 0] if proj.shape[1] == 1 else proj

    def project_monomial(self, powers: tuple[int, ...]) -> np.ndarray:
        """Compute Cumulant-Debiased weak-form projection of a single monomial."""
        feats = [
            hermite_debias_monomial(X, powers, self.noise_vars, enabled=self.cumulant_debiasing)[:, None]
            for X in self.trajectories
        ]
        return self.project_pointwise(feats)[:, 0]


class MSAVPProjector:
    """Layer 1 engine: builds the Multi-Scale Adaptive Variational Projection substrate."""

    def __init__(self, config: MSAVPConfig | None = None) -> None:
        self.config = config or MSAVPConfig()

    def _estimate_local_curvature(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Estimate smooth local second-derivative magnitude per sample for adaptive W_k*."""
        n = len(t)
        dt = float(np.median(np.diff(t)))
        win = min(max(7, (n // 20) | 1), n if n % 2 == 1 else n - 1)
        if win < 5:
            return np.ones(n, dtype=float)
        X_norm = X / np.maximum(np.std(X, axis=0, keepdims=True), 1e-6)
        curv = np.zeros(n, dtype=float)
        for j in range(X.shape[1]):
            d2 = savgol_filter(X_norm[:, j], window_length=win, polyorder=min(3, win - 2), deriv=2, delta=dt)
            curv += d2 * d2
        return np.sqrt(curv / X.shape[1] + 1e-6)

    def _select_window_lengths(
        self,
        t: np.ndarray,
        X: np.ndarray,
        centers: np.ndarray,
        rel_noise: float,
    ) -> np.ndarray:
        """Choose locally optimal odd window length W_k* balancing bias and noise variance."""
        n = len(t)
        w_min = min(self.config.min_window_pts, max(5, (n // 4) | 1))
        w_max = min(self.config.max_window_pts, max(w_min, (n // 2) | 1))
        if w_min % 2 == 0:
            w_min += 1
        if w_max % 2 == 0:
            w_max -= 1
        w_max = max(w_max, w_min)

        if not self.config.adaptive_multiscale or w_min == w_max:
            # Fixed mid-scale window scaled gently by noise level
            alpha = np.clip(rel_noise / 0.15, 0.0, 1.0)
            w_fixed = int(round(w_min + alpha * (w_max - w_min))) | 1
            return np.full(len(centers), min(max(w_fixed, w_min), w_max), dtype=int)

        curv = self._estimate_local_curvature(t, X)
        med_curv = float(np.median(curv)) + 1e-6
        p = self.config.poly_order
        exponent = 1.0 / (2.0 * p + 5.0)

        # Base scale grows with noise level so high-noise runs average more samples
        noise_scale = np.clip(0.35 + 3.2 * rel_noise, 0.25, 1.0)
        w_base = w_min + noise_scale * (w_max - w_min)

        widths = np.empty(len(centers), dtype=int)
        for idx, c in enumerate(centers):
            # Higher local curvature -> narrower window; smoother region -> wider window
            ratio = med_curv / (float(curv[c]) + 1e-6)
            w_raw = w_base * (ratio ** (2.0 * exponent))
            w_clamped = int(round(np.clip(w_raw, w_min, w_max)))
            if w_clamped % 2 == 0:
                w_clamped += 1
            widths[idx] = min(max(w_clamped, w_min), w_max)
        return widths

    def fit_transform(
        self,
        t_list: list[np.ndarray],
        trajectories: list[np.ndarray],
        var_names: list[str],
    ) -> WeakFormSubstrate:
        """Construct the full MS-AVP weak-form substrate from raw trajectories."""
        d = trajectories[0].shape[1]
        if self.config.noise_override is not None:
            scales = np.maximum(np.std(np.vstack(trajectories), axis=0), 1e-6)
            noise_stds = np.full(d, float(self.config.noise_override)) * scales
        else:
            noise_stds = estimate_noise_wavelet_mad(trajectories)
            # Suppress tiny discretization curvature floor on noiseless trajectories
            scales = np.maximum(np.std(np.vstack(trajectories), axis=0), 1e-6)
            rel_est = noise_stds / scales
            noise_stds = np.where(rel_est < 0.004, 0.0, noise_stds)
        noise_vars = noise_stds**2
        scales = np.maximum(np.std(np.vstack(trajectories), axis=0), 1e-6)
        rel_noise = float(np.mean(noise_stds / scales))

        if not self.config.use_weak_form:
            # Pointwise Savitzky-Golay derivative fallback (for ablation benchmarking)
            dX_all = [estimate_derivatives_savgol(t, X) for t, X in zip(t_list, trajectories)]
            V = np.vstack(dX_all)
            W_states = np.vstack(trajectories)
            traj_ids = np.concatenate([np.full(len(X), idx, dtype=int) for idx, X in enumerate(trajectories)])
            return WeakFormSubstrate(
                t_list=t_list,
                trajectories=trajectories,
                var_names=var_names,
                noise_stds=noise_stds,
                noise_vars=noise_vars,
                subdomains=[],
                weak_velocities=V,
                weak_states=W_states,
                traj_ids=traj_ids,
                window_widths=np.ones(len(V), dtype=int),
                use_weak_form=False,
                cumulant_debiasing=False,
                metadata={"relative_noise_est": rel_noise},
            )

        subdomains: list[SubdomainSpec] = []
        v_rows: list[np.ndarray] = []
        x_rows: list[np.ndarray] = []
        traj_ids_list: list[int] = []
        widths_list: list[int] = []

        for traj_idx, (t_arr, X_arr) in enumerate(zip(t_list, trajectories)):
            n = len(t_arr)
            if n < 7:
                raise ValueError(f"Trajectory {traj_idx} too short ({n} samples) for weak-form projection.")
            n_sub = min(self.config.n_subdomains, max(12, n - 6))
            margin = min(self.config.min_window_pts // 2 + 1, n // 5)
            centers = np.unique(np.linspace(margin, n - 1 - margin, n_sub, dtype=int))
            widths = self._select_window_lengths(t_arr, X_arr, centers, rel_noise)

            for c, w in zip(centers, widths):
                half = min(int(w) // 2, int(c), int(n - 1 - c))
                if half < 2:
                    continue
                s, e = int(c - half), int(c + half + 1)
                w_phi, w_dphi, w_d2phi = bump_test_function(t_arr[s:e], self.config.poly_order)
                X_seg = X_arr[s:e]
                # Integration by parts: <phi, dx/dt> = -<phi', x>
                v_k = -(w_dphi @ X_seg)
                x_k = w_phi @ X_seg
                subdomains.append(
                    SubdomainSpec(
                        traj_idx=traj_idx,
                        start_idx=s,
                        end_idx=e,
                        center_time=float(t_arr[c]),
                        w_phi=w_phi,
                        w_dphi=w_dphi,
                        w_d2phi=w_d2phi,
                    )
                )
                v_rows.append(v_k)
                x_rows.append(x_k)
                traj_ids_list.append(traj_idx)
                widths_list.append(e - s)

        return WeakFormSubstrate(
            t_list=t_list,
            trajectories=trajectories,
            var_names=var_names,
            noise_stds=noise_stds,
            noise_vars=noise_vars,
            subdomains=subdomains,
            weak_velocities=np.vstack(v_rows),
            weak_states=np.vstack(x_rows),
            traj_ids=np.asarray(traj_ids_list, dtype=int),
            window_widths=np.asarray(widths_list, dtype=int),
            use_weak_form=True,
            cumulant_debiasing=self.config.cumulant_debiasing,
            metadata={"relative_noise_est": rel_noise},
        )


class SpacetimeWeakProjector:
    """2D Spacetime Variational Weak-Form Projector for 1+1D PDEs u(x, t).

    Uses separable compactly supported C^p test functions Phi(x, t) = phi_x(x) * phi_t(t)
    so ALL spatial and temporal derivatives are transferred onto smooth analytical test
    functions via integration by parts, with analytical Hermite cumulant debiasing on u^2, u^3.
    """

    def __init__(self, poly_order: int = 4, nx_win: int = 15, nt_win: int = 15, cumulant_debiasing: bool = True) -> None:
        self.poly_order = poly_order
        self.nx_win = nx_win | 1
        self.nt_win = nt_win | 1
        self.cumulant_debiasing = cumulant_debiasing

    def build_pde_system(
        self,
        x: np.ndarray,
        t: np.ndarray,
        u: np.ndarray,
        n_x_sub: int = 18,
        n_t_sub: int = 18,
    ) -> tuple[np.ndarray, np.ndarray, list[str], float]:
        """Build weak-form PDE regression system G * xi = b for candidate PDE terms.

        Candidate basis: [u, u^2, u_x, u_xx, u*u_x, u^2*u_x].
        Note exact conservative integration-by-parts identities:
          <Phi, u_t>    = -<phi_t' * phi_x, u>
          <Phi, u_x>    = -<phi_t  * phi_x', u>
          <Phi, u_xx>   = +<phi_t  * phi_x'', u>
          <Phi, u u_x>  = -0.5 * <phi_t * phi_x', u^2 - sigma^2>
          <Phi, u^2 u_x>= -(1/3) * <phi_t * phi_x', u^3 - 3 sigma^2 u>
        """
        nt, nx = u.shape
        # Estimate 2D noise std via 2D Laplacian-like MAD
        if nt >= 5 and nx >= 5:
            lap = (u[1:-1, 2:] + u[1:-1, :-2] + u[2:, 1:-1] + u[:-2, 1:-1] - 4.0 * u[1:-1, 1:-1]) / np.sqrt(20.0)
            noise_std = float(np.median(np.abs(lap - np.median(lap))) / 0.67448975)
            if noise_std / (float(np.std(u)) + 1e-6) < 0.004:
                noise_std = 0.0
        else:
            noise_std = 0.0
        noise_var = noise_std**2

        u2_deb = hermite_poly_prob(u, 2, noise_var) if self.cumulant_debiasing else u**2
        u3_deb = hermite_poly_prob(u, 3, noise_var) if self.cumulant_debiasing else u**3

        hx = min(self.nx_win // 2, nx // 4)
        ht = min(self.nt_win // 2, nt // 4)
        cx_list = np.unique(np.linspace(hx, nx - 1 - hx, n_x_sub, dtype=int))
        ct_list = np.unique(np.linspace(ht, nt - 1 - ht, n_t_sub, dtype=int))

        names = ["u", "u**2", "u_x", "u_xx", "u*u_x", "u**2*u_x"]
        G_rows = []
        b_rows = []

        for ct in ct_list:
            ts, te = int(ct - ht), int(ct + ht + 1)
            wt, dwt, _ = bump_test_function(t[ts:te], self.poly_order)
            for cx in cx_list:
                xs, xe = int(cx - hx), int(cx + hx + 1)
                wx, dwx, d2wx = bump_test_function(x[xs:xe], self.poly_order)
                U_seg = u[ts:te, xs:xe]
                U2_seg = u2_deb[ts:te, xs:xe]
                U3_seg = u3_deb[ts:te, xs:xe]

                # Bilinear contractions: wt^T @ U_seg @ wx
                b_val = -(dwt @ U_seg @ wx)  # <Phi, u_t>
                f_u = wt @ U_seg @ wx
                f_u2 = wt @ U2_seg @ wx
                f_ux = -(wt @ U_seg @ dwx)
                f_uxx = wt @ U_seg @ d2wx
                f_uux = -0.5 * (wt @ U2_seg @ dwx)
                f_u2ux = -(1.0 / 3.0) * (wt @ U3_seg @ dwx)

                G_rows.append([f_u, f_u2, f_ux, f_uxx, f_uux, f_u2ux])
                b_rows.append(b_val)

        return np.asarray(G_rows, dtype=float), np.asarray(b_rows, dtype=float), names, noise_std
