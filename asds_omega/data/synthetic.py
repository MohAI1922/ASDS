"""Controlled synthetic ODE and PDE dynamical systems with exact ground truth.

Every dataset exposes:
- Multi-trajectory state observations simulated via high-precision RK45/DOP853.
- Exact ground-truth symbolic equations, differential causal adjacency matrices
  A[j, i] = 1 iff d(f_i)/d(x_j) != 0, true conservation laws / first integrals,
  discrete/continuous symmetries, exact phase-space divergence, and an oracle
  simulator callable for Popperian Active Falsification (PAFE).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class DynamicalDataset:
    """Standard container for multi-trajectory ODE / dynamical system datasets."""

    name: str
    domain: str
    var_names: list[str]
    t_list: list[np.ndarray]
    trajectories: list[np.ndarray]  # Possibly noisy observations, each (T, d)
    clean_trajectories: list[np.ndarray]  # Ground-truth noiseless trajectories
    ground_truth_equations: list[str]
    ground_truth_causal_adj: np.ndarray  # A[j, i] = 1 iff x_j -> dx_i/dt
    ground_truth_invariants: list[str] = field(default_factory=list)
    ground_truth_symmetries: list[str] = field(default_factory=list)
    true_divergence: str | None = None
    units: dict[str, tuple[int, ...]] | None = None  # [M, L, T] SI exponents
    rhs_fn: Callable[[float, np.ndarray], np.ndarray] | None = None
    state_bounds: list[tuple[float, float]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_vars(self) -> int:
        return len(self.var_names)

    @property
    def n_trajectories(self) -> int:
        return len(self.trajectories)

    def simulate_oracle(
        self,
        x0: np.ndarray,
        t_span: tuple[float, float] = (0.0, 10.0),
        dt: float = 0.02,
        noise_std: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute a targeted oracle experiment from initial condition x0."""
        if self.rhs_fn is None:
            raise ValueError(f"Dataset '{self.name}' has no ground-truth oracle rhs_fn.")
        t_eval = np.arange(t_span[0], t_span[1] + 1e-12, dt)
        sol = solve_ivp(
            self.rhs_fn,
            (t_eval[0], t_eval[-1]),
            np.asarray(x0, dtype=float),
            t_eval=t_eval,
            method="DOP853",
            rtol=1e-10,
            atol=1e-12,
        )
        X = sol.y.T
        if noise_std > 0.0:
            gen = rng if rng is not None else np.random.default_rng(0)
            scale = np.maximum(np.std(X, axis=0), 1e-6)
            X = X + gen.normal(0.0, noise_std * scale, size=X.shape)
        return t_eval, X


@dataclass
class PDEDataset:
    """Container for 1+1D spatiotemporal PDE datasets u(x, t)."""

    name: str
    domain: str
    var_name: str
    x: np.ndarray  # (Nx,)
    t: np.ndarray  # (Nt,)
    u: np.ndarray  # (Nt, Nx) observed field
    u_clean: np.ndarray  # (Nt, Nx) noiseless field
    ground_truth_equation: str
    ground_truth_symmetries: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _integrate_multi(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    ic_list: list[np.ndarray],
    t_span: tuple[float, float],
    dt: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    t_eval = np.arange(t_span[0], t_span[1] + 1e-12, dt)
    t_list, trajs = [], []
    for x0 in ic_list:
        sol = solve_ivp(
            rhs,
            (t_eval[0], t_eval[-1]),
            np.asarray(x0, dtype=float),
            t_eval=t_eval,
            method="DOP853",
            rtol=1e-10,
            atol=1e-12,
        )
        if not sol.success or sol.y.shape[1] != len(t_eval):
            raise RuntimeError(f"ODE integration failed for IC {x0}: {sol.message}")
        t_list.append(t_eval.copy())
        trajs.append(sol.y.T.copy())
    return t_list, trajs


def gen_lorenz(
    n_trajectories: int = 4,
    t_end: float = 15.0,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    seed: int = 42,
) -> DynamicalDataset:
    """Lorenz-63 chaotic dissipative system with exact discrete Z_2 symmetry."""
    rng = np.random.default_rng(seed)

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    base_ics = [
        np.array([-8.0, 7.0, 27.0]),
        np.array([8.0, -7.0, 27.0]),  # Z_2 symmetric partner
        np.array([1.0, 1.0, 1.0]),
        np.array([-5.0, -10.0, 20.0]),
    ]
    ics = base_ics[:n_trajectories]
    while len(ics) < n_trajectories:
        ics.append(rng.uniform([-12.0, -12.0, 10.0], [12.0, 12.0, 35.0]))

    t_list, trajs = _integrate_multi(rhs, ics, (0.0, t_end), dt)
    # A[j, i] = 1 iff x_j appears in dx_i/dt
    # dx/dt = 10*(y - x)      -> parents: x, y
    # dy/dt = 28*x - y - x*z  -> parents: x, y, z
    # dz/dt = x*y - (8/3)*z   -> parents: x, y, z
    causal_adj = np.array(
        [
            [1, 1, 1],  # x -> dx, dy, dz
            [1, 1, 1],  # y -> dx, dy, dz
            [0, 1, 1],  # z -> dy, dz (NOT dx!)
        ],
        dtype=int,
    )
    return DynamicalDataset(
        name="lorenz",
        domain="Physics (Chaotic Convection)",
        var_names=["x", "y", "z"],
        t_list=t_list,
        trajectories=[tr.copy() for tr in trajs],
        clean_trajectories=trajs,
        ground_truth_equations=[
            "10*y - 10*x",
            "28*x - y - x*z",
            "x*y - 8/3*z",
        ],
        ground_truth_causal_adj=causal_adj,
        ground_truth_invariants=[],  # Dissipative system: NO energy/volume conservation
        ground_truth_symmetries=["Z2_reflection: (x, y, z) -> (-x, -y, z)"],
        true_divergence=f"-{sigma + 1.0 + beta:.6f}",
        rhs_fn=rhs,
        state_bounds=[(-18.0, 18.0), (-22.0, 22.0), (5.0, 45.0)],
        metadata={"sigma": sigma, "rho": rho, "beta": beta, "dissipative": True},
    )


def gen_lotka_volterra(
    n_trajectories: int = 4,
    t_end: float = 20.0,
    dt: float = 0.02,
    alpha: float = 1.1,
    beta: float = 0.4,
    delta: float = 0.4,
    gamma: float = 0.1,
    seed: int = 42,
) -> DynamicalDataset:
    """Lotka-Volterra ecological predator-prey system with exact logarithmic first integral."""
    rng = np.random.default_rng(seed)

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        x, y = state
        return np.array([alpha * x - beta * x * y, delta * x * y - gamma * y])

    base_ics = [
        np.array([1.2, 1.5]),
        np.array([2.5, 1.0]),
        np.array([0.6, 2.8]),
        np.array([1.8, 3.2]),
    ]
    ics = base_ics[:n_trajectories]
    while len(ics) < n_trajectories:
        ics.append(rng.uniform([0.4, 0.5], [3.0, 3.5]))

    t_list, trajs = _integrate_multi(rhs, ics, (0.0, t_end), dt)
    causal_adj = np.array([[1, 1], [1, 1]], dtype=int)
    return DynamicalDataset(
        name="lotka_volterra",
        domain="Ecology (Predator-Prey Dynamics)",
        var_names=["x", "y"],
        t_list=t_list,
        trajectories=[tr.copy() for tr in trajs],
        clean_trajectories=trajs,
        ground_truth_equations=[
            f"{alpha}*x - {beta}*x*y",
            f"{delta}*x*y - {gamma}*y",
        ],
        ground_truth_causal_adj=causal_adj,
        ground_truth_invariants=[f"{delta}*x + {beta}*y - {gamma}*log(x) - {alpha}*log(y)"],
        ground_truth_symmetries=["Time-translation invariance"],
        true_divergence="alpha - gamma + (delta - beta)*x - beta*y",
        rhs_fn=rhs,
        state_bounds=[(0.2, 4.0), (0.2, 5.0)],
        metadata={"alpha": alpha, "beta": beta, "delta": delta, "gamma": gamma},
    )


def gen_sir(
    n_trajectories: int = 4,
    t_end: float = 40.0,
    dt: float = 0.05,
    beta: float = 0.5,
    gamma: float = 0.1,
    seed: int = 42,
) -> DynamicalDataset:
    """Kermack-McKendrick SIR epidemic model with exact conservation S + I + R = 1."""
    rng = np.random.default_rng(seed)

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        S, I, R = state
        return np.array([-beta * S * I, beta * S * I - gamma * I, gamma * I])

    base_ics = [
        np.array([0.95, 0.05, 0.00]),
        np.array([0.80, 0.15, 0.05]),
        np.array([0.70, 0.25, 0.05]),
        np.array([0.90, 0.02, 0.08]),
    ]
    ics = base_ics[:n_trajectories]
    while len(ics) < n_trajectories:
        s0 = rng.uniform(0.6, 0.95)
        i0 = rng.uniform(0.02, 1.0 - s0)
        ics.append(np.array([s0, i0, 1.0 - s0 - i0]))

    t_list, trajs = _integrate_multi(rhs, ics, (0.0, t_end), dt)
    # dS/dt = -beta*S*I       -> parents: S, I
    # dI/dt = beta*S*I - g*I  -> parents: S, I
    # dR/dt = g*I             -> parents: I ONLY (NOT S, NOT R!)
    causal_adj = np.array(
        [
            [1, 1, 0],  # S -> dS, dI
            [1, 1, 1],  # I -> dS, dI, dR
            [0, 0, 0],  # R -> none
        ],
        dtype=int,
    )
    return DynamicalDataset(
        name="sir",
        domain="Epidemiology (Compartmental Disease Spread)",
        var_names=["S", "I", "R"],
        t_list=t_list,
        trajectories=[tr.copy() for tr in trajs],
        clean_trajectories=trajs,
        ground_truth_equations=[
            "-1/2*S*I",
            "1/2*S*I - 1/10*I",
            "1/10*I",
        ],
        ground_truth_causal_adj=causal_adj,
        ground_truth_invariants=["S + I + R"],
        ground_truth_symmetries=["Mass conservation S + I + R = const"],
        rhs_fn=rhs,
        state_bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        metadata={"beta": beta, "gamma": gamma, "R0": beta / gamma},
    )


def gen_sir_regime_drift(
    t_end: float = 60.0,
    dt: float = 0.05,
    t_shift: float = 20.0,
    beta_before: float = 0.6,
    beta_after: float = 0.2,
    gamma: float = 0.1,
    waning_after: float = 0.04,
) -> DynamicalDataset:
    """Streaming SIR -> SIRS regime drift dataset (lockdown + immunity waning)."""

    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        S, I, R = state
        if t < t_shift:
            b = beta_before
            w = 0.0
        else:
            b = beta_after
            w = waning_after
        return np.array([-b * S * I + w * R, b * S * I - gamma * I, gamma * I - w * R])

    t_list, trajs = _integrate_multi(rhs, [np.array([0.96, 0.04, 0.0])], (0.0, t_end), dt)
    causal_adj = np.array(
        [
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 1],
        ],
        dtype=int,
    )
    return DynamicalDataset(
        name="sir_regime_drift",
        domain="Epidemiology (Non-Stationary Policy & Immunity Drift)",
        var_names=["S", "I", "R"],
        t_list=t_list,
        trajectories=[tr.copy() for tr in trajs],
        clean_trajectories=trajs,
        ground_truth_equations=[
            f"-{beta_after}*S*I + {waning_after}*R",
            f"{beta_after}*S*I - {gamma}*I",
            f"{gamma}*I - {waning_after}*R",
        ],
        ground_truth_causal_adj=causal_adj,
        ground_truth_invariants=["S + I + R"],
        ground_truth_symmetries=["Mass conservation S + I + R = const across drift"],
        rhs_fn=rhs,
        state_bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        metadata={
            "t_shift": t_shift,
            "beta_before": beta_before,
            "beta_after": beta_after,
            "gamma": gamma,
            "waning_after": waning_after,
        },
    )


def gen_pendulum(
    n_trajectories: int = 4,
    t_end: float = 16.0,
    dt: float = 0.02,
    g_over_L: float = 2.0,
    damping: float = 0.0,
    seed: int = 42,
) -> DynamicalDataset:
    """Nonlinear pendulum with exact Hamiltonian energy & Z_2 parity symmetry."""
    rng = np.random.default_rng(seed)

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        theta, omega = state
        return np.array([omega, -g_over_L * np.sin(theta) - damping * omega])

    base_ics = [
        np.array([1.2, 0.0]),
        np.array([-1.2, 0.0]),  # Z_2 parity partner
        np.array([2.1, 0.5]),
        np.array([-1.8, -0.7]),
    ]
    ics = base_ics[:n_trajectories]
    while len(ics) < n_trajectories:
        ics.append(rng.uniform([-2.4, -1.5], [2.4, 1.5]))

    t_list, trajs = _integrate_multi(rhs, ics, (0.0, t_end), dt)
    causal_adj = np.array(
        [
            [0, 1],  # theta -> d(omega)/dt
            [1, 1 if damping != 0.0 else 0],  # omega -> d(theta)/dt (and damping if c>0)
        ],
        dtype=int,
    )
    invs = [f"1/2*omega**2 - {g_over_L}*cos(theta)"] if damping == 0.0 else []
    eq2 = f"-{g_over_L}*sin(theta)" if damping == 0.0 else f"-{g_over_L}*sin(theta) - {damping}*omega"
    return DynamicalDataset(
        name="pendulum" if damping == 0.0 else "damped_pendulum",
        domain="Classical Mechanics (Nonlinear Oscillator)",
        var_names=["theta", "omega"],
        t_list=t_list,
        trajectories=[tr.copy() for tr in trajs],
        clean_trajectories=trajs,
        ground_truth_equations=["omega", eq2],
        ground_truth_causal_adj=causal_adj,
        ground_truth_invariants=invs,
        ground_truth_symmetries=["Z2_parity: (theta, omega) -> (-theta, -omega)"],
        true_divergence=f"{-damping:.6f}",
        units={"theta": (0, 0, 0), "omega": (0, 0, -1)},  # [M, L, T]
        rhs_fn=rhs,
        state_bounds=[(-2.8, 2.8), (-3.0, 3.0)],
        metadata={"g_over_L": g_over_L, "damping": damping},
    )


def gen_kepler_2body(
    n_trajectories: int = 4,
    t_end: float = 12.0,
    dt: float = 0.02,
    mu: float = 1.0,
    seed: int = 42,
) -> DynamicalDataset:
    """2D Kepler central-force orbit with continuous SO(2) symmetry, L and E conservation."""
    rng = np.random.default_rng(seed)

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        x, y, vx, vy = state
        r3 = (x * x + y * y) ** 1.5
        return np.array([vx, vy, -mu * x / r3, -mu * y / r3])

    base_ics = [
        np.array([1.0, 0.0, 0.0, 1.15]),
        np.array([0.0, 1.0, -1.15, 0.0]),  # 90-deg SO(2) rotated orbit
        np.array([-1.2, 0.0, 0.0, -0.85]),
        np.array([0.8, 0.6, -0.7, 0.9]),
    ]
    ics = base_ics[:n_trajectories]
    while len(ics) < n_trajectories:
        ang = rng.uniform(0, 2 * np.pi)
        r0 = rng.uniform(0.9, 1.4)
        v0 = rng.uniform(0.85, 1.15)
        ics.append(np.array([r0 * np.cos(ang), r0 * np.sin(ang), -v0 * np.sin(ang), v0 * np.cos(ang)]))

    t_list, trajs = _integrate_multi(rhs, ics, (0.0, t_end), dt)
    causal_adj = np.array(
        [
            [0, 0, 1, 1],  # x -> dvx, dvy
            [0, 0, 1, 1],  # y -> dvx, dvy
            [1, 0, 0, 0],  # vx -> dx
            [0, 1, 0, 0],  # vy -> dy
        ],
        dtype=int,
    )
    return DynamicalDataset(
        name="kepler",
        domain="Astrophysics (Central-Force Orbital Mechanics)",
        var_names=["x", "y", "vx", "vy"],
        t_list=t_list,
        trajectories=[tr.copy() for tr in trajs],
        clean_trajectories=trajs,
        ground_truth_equations=[
            "vx",
            "vy",
            "-x/(x**2 + y**2)**(3/2)",
            "-y/(x**2 + y**2)**(3/2)",
        ],
        ground_truth_causal_adj=causal_adj,
        ground_truth_invariants=[
            "x*vy - y*vx",
            "1/2*(vx**2 + vy**2) - 1/sqrt(x**2 + y**2)",
        ],
        ground_truth_symmetries=["SO(2) rotational invariance", "Z2_inversion: (x,y,vx,vy)->(-x,-y,-vx,-vy)"],
        true_divergence="0.000000",
        units={"x": (0, 1, 0), "y": (0, 1, 0), "vx": (0, 1, -1), "vy": (0, 1, -1)},
        rhs_fn=rhs,
        state_bounds=[(-2.5, 2.5), (-2.5, 2.5), (-2.0, 2.0), (-2.0, 2.0)],
        metadata={"mu": mu},
    )


def gen_michaelis_menten(
    n_trajectories: int = 4,
    t_end: float = 12.0,
    dt: float = 0.05,
    Vmax: float = 1.5,
    Km: float = 0.5,
    seed: int = 42,
) -> DynamicalDataset:
    """Michaelis-Menten biochemical enzyme kinetics (rational law + stoichiometry S+P=const)."""
    rng = np.random.default_rng(seed)

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        S, P = state
        rate = Vmax * S / (Km + S)
        return np.array([-rate, rate])

    base_ics = [
        np.array([3.0, 0.0]),
        np.array([5.0, 0.5]),
        np.array([1.5, 0.2]),
        np.array([4.0, 1.0]),
    ]
    ics = base_ics[:n_trajectories]
    while len(ics) < n_trajectories:
        ics.append(np.array([rng.uniform(1.0, 6.0), rng.uniform(0.0, 1.5)]))

    t_list, trajs = _integrate_multi(rhs, ics, (0.0, t_end), dt)
    causal_adj = np.array(
        [
            [1, 1],  # S -> dS, dP
            [0, 0],  # P -> none
        ],
        dtype=int,
    )
    return DynamicalDataset(
        name="michaelis_menten",
        domain="Biochemistry (Rational Enzyme Kinetics)",
        var_names=["S", "P"],
        t_list=t_list,
        trajectories=[tr.copy() for tr in trajs],
        clean_trajectories=trajs,
        ground_truth_equations=[
            f"-{Vmax}*S/({Km} + S)",
            f"{Vmax}*S/({Km} + S)",
        ],
        ground_truth_causal_adj=causal_adj,
        ground_truth_invariants=["S + P"],
        ground_truth_symmetries=["Stoichiometric mass conservation S + P = const"],
        rhs_fn=rhs,
        state_bounds=[(0.05, 6.0), (0.0, 6.0)],
        metadata={"Vmax": Vmax, "Km": Km, "rational": True},
    )


def gen_burgers_pde(
    nx: int = 64,
    nt: int = 101,
    L: float = 8.0,
    t_end: float = 4.0,
    nu: float = 0.15,
) -> PDEDataset:
    """1D Viscous Burgers PDE: u_t = -u * u_x + nu * u_xx solved spectrally on periodic domain."""
    x = np.linspace(-L / 2, L / 2, nx, endpoint=False)
    dx = x[1] - x[0]
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    t_eval = np.linspace(0.0, t_end, nt)

    # Smooth multi-modal initial profile
    u0 = np.exp(-((x + 1.0) ** 2)) + 0.5 * np.exp(-((x - 1.5) ** 2) / 1.5)

    def rhs(_t: float, u_vec: np.ndarray) -> np.ndarray:
        u_hat = np.fft.fft(u_vec)
        ux = np.fft.ifft(1j * k * u_hat).real
        uxx = np.fft.ifft(-(k**2) * u_hat).real
        return -u_vec * ux + nu * uxx

    sol = solve_ivp(rhs, (0.0, t_end), u0, t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-11)
    u_clean = sol.y.T
    return PDEDataset(
        name="burgers_pde",
        domain="Fluid Mechanics (Nonlinear Shock-Diffusion PDE)",
        var_name="u",
        x=x,
        t=t_eval,
        u=u_clean.copy(),
        u_clean=u_clean,
        ground_truth_equation=f"-1.0*u*u_x + {nu}*u_xx",
        ground_truth_symmetries=["Spatial translation invariance", "Mass conservation int u dx = const"],
        metadata={"nu": nu, "dx": dx, "dt": t_eval[1] - t_eval[0]},
    )


def gen_advection_diffusion_pde(
    nx: int = 64,
    nt: int = 101,
    L: float = 8.0,
    t_end: float = 4.0,
    c: float = 0.8,
    D: float = 0.12,
) -> PDEDataset:
    """1D Advection-Diffusion PDE: u_t = -c * u_x + D * u_xx solved spectrally."""
    x = np.linspace(-L / 2, L / 2, nx, endpoint=False)
    dx = x[1] - x[0]
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    t_eval = np.linspace(0.0, t_end, nt)
    u0 = np.exp(-(x**2) / 0.8) + 0.4 * np.cos(2.0 * np.pi * x / L)

    def rhs(_t: float, u_vec: np.ndarray) -> np.ndarray:
        u_hat = np.fft.fft(u_vec)
        ux = np.fft.ifft(1j * k * u_hat).real
        uxx = np.fft.ifft(-(k**2) * u_hat).real
        return -c * ux + D * uxx

    sol = solve_ivp(rhs, (0.0, t_end), u0, t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-11)
    u_clean = sol.y.T
    return PDEDataset(
        name="advection_diffusion_pde",
        domain="Environmental Science (Transport PDE)",
        var_name="u",
        x=x,
        t=t_eval,
        u=u_clean.copy(),
        u_clean=u_clean,
        ground_truth_equation=f"-{c}*u_x + {D}*u_xx",
        ground_truth_symmetries=["Spatial translation invariance", "Linearity / scaling invariance", "Mass conservation int u dx = const"],
        metadata={"c": c, "D": D, "dx": dx, "dt": t_eval[1] - t_eval[0]},
    )


def gen_duffing(
    n_trajectories: int = 4,
    t_end: float = 16.0,
    dt: float = 0.02,
    delta: float = 0.2,
    alpha: float = -1.0,
    beta: float = 1.0,
    seed: int = 42,
) -> DynamicalDataset:
    """Double-well Duffing oscillator: dx/dt = v, dv/dt = -delta*v - alpha*x - beta*x^3.

    Cubic self-power nonlinearity (-x^3) exhibits strong Errors-in-Variables bias
    E[(x+eps)^3] = x^3 + 3*sigma^2*x, directly isolating Hermite Cumulant EIV Debiasing!
    Exact Z_2 parity symmetry: (x, v) -> (-x, -v).
    """
    rng = np.random.default_rng(seed)

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        x, v = state
        return np.array([v, -delta * v - alpha * x - beta * (x**3)])

    base_ics = [
        np.array([1.6, 0.5]),
        np.array([-1.6, -0.5]),  # Z_2 parity partner
        np.array([0.5, 1.5]),
        np.array([-0.5, -1.5]),
    ]
    ics = base_ics[:n_trajectories]
    while len(ics) < n_trajectories:
        ics.append(rng.uniform([-2.0, -1.8], [2.0, 1.8]))

    t_list, trajs = _integrate_multi(rhs, ics, (0.0, t_end), dt)
    causal_adj = np.array(
        [
            [0, 1],  # x -> dv/dt
            [1, 1 if delta != 0.0 else 0],  # v -> dx/dt, dv/dt
        ],
        dtype=int,
    )
    return DynamicalDataset(
        name="duffing",
        domain="Nonlinear Mechanics (Cubic Double-Well Oscillator)",
        var_names=["x", "v"],
        t_list=t_list,
        trajectories=[tr.copy() for tr in trajs],
        clean_trajectories=trajs,
        ground_truth_equations=[
            "v",
            "-1/5*v + x - x**3",
        ],
        ground_truth_causal_adj=causal_adj,
        ground_truth_invariants=[],
        ground_truth_symmetries=["Z2_parity: (x, v) -> (-x, -v)"],
        true_divergence=f"{-delta:.6f}",
        rhs_fn=rhs,
        state_bounds=[(-2.5, 2.5), (-2.5, 2.5)],
        metadata={"delta": delta, "alpha": alpha, "beta": beta, "cubic_eiv": True},
    )


def gen_van_der_pol(
    n_trajectories: int = 4,
    t_end: float = 18.0,
    dt: float = 0.02,
    mu: float = 1.5,
    seed: int = 42,
) -> DynamicalDataset:
    """Van der Pol relaxation limit-cycle oscillator: dx/dt = y, dy/dt = mu*(1 - x^2)*y - x.

    Self-power quadratic-damping term (-mu*x^2*y) has E[(x+e)^2 y] = x^2*y + sigma^2*y,
    creating systematic linear-damping EIV bias eliminated by Hermite Cumulant Debiasing!
    Exact Z_2 parity symmetry: (x, y) -> (-x, -y).
    """
    rng = np.random.default_rng(seed)

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        x, y = state
        return np.array([y, mu * (1.0 - x * x) * y - x])

    base_ics = [
        np.array([2.0, 0.0]),
        np.array([-2.0, 0.0]),
        np.array([0.4, 1.8]),
        np.array([-0.4, -1.8]),
    ]
    ics = base_ics[:n_trajectories]
    while len(ics) < n_trajectories:
        ics.append(rng.uniform([-2.5, -2.5], [2.5, 2.5]))

    t_list, trajs = _integrate_multi(rhs, ics, (0.0, t_end), dt)
    causal_adj = np.array([[0, 1], [1, 1]], dtype=int)
    return DynamicalDataset(
        name="van_der_pol",
        domain="Nonlinear Dynamics (Limit-Cycle Oscillator)",
        var_names=["x", "y"],
        t_list=t_list,
        trajectories=[tr.copy() for tr in trajs],
        clean_trajectories=trajs,
        ground_truth_equations=[
            "y",
            "3/2*y - 3/2*x**2*y - x",
        ],
        ground_truth_causal_adj=causal_adj,
        ground_truth_invariants=[],
        ground_truth_symmetries=["Z2_parity: (x, y) -> (-x, -y)"],
        true_divergence=f"{mu}*(1 - x**2)",
        rhs_fn=rhs,
        state_bounds=[(-3.0, 3.0), (-3.5, 3.5)],
        metadata={"mu": mu, "cubic_eiv": True},
    )


def gen_lorenz96(
    dim: int = 8,
    n_trajectories: int = 4,
    t_end: float = 12.0,
    dt: float = 0.02,
    forcing: float = 8.0,
    seed: int = 42,
) -> DynamicalDataset:
    """High-dimensional Lorenz-96 atmospheric ring: dx_i/dt = (x_{i+1} - x_{i-2})*x_{i-1} - x_i + F.

    Tests sparse causal + symbolic scaling in d >= 8 dimensions with cyclic shift symmetry.
    """
    rng = np.random.default_rng(seed)

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        dx = np.empty(dim, dtype=float)
        for i in range(dim):
            dx[i] = (state[(i + 1) % dim] - state[(i - 2) % dim]) * state[(i - 1) % dim] - state[i] + forcing
        return dx

    ics = []
    for k in range(n_trajectories):
        x0 = np.full(dim, forcing, dtype=float) + rng.normal(0.0, 1.5, size=dim)
        x0[k % dim] += 0.8
        ics.append(x0)

    t_list, trajs = _integrate_multi(rhs, ics, (0.0, t_end), dt)
    vnames = [f"x{i+1}" for i in range(dim)]
    gt_eqs = []
    causal_adj = np.zeros((dim, dim), dtype=int)
    for i in range(dim):
        ip1 = (i + 1) % dim
        im1 = (i - 1) % dim
        im2 = (i - 2) % dim
        gt_eqs.append(f"({vnames[ip1]} - {vnames[im2]})*{vnames[im1]} - {vnames[i]} + {int(forcing)}")
        for j in (ip1, im1, im2, i):
            causal_adj[j, i] = 1

    return DynamicalDataset(
        name=f"lorenz96_{dim}d",
        domain=f"High-Dimensional Atmospheric Chaos (d={dim})",
        var_names=vnames,
        t_list=t_list,
        trajectories=[tr.copy() for tr in trajs],
        clean_trajectories=trajs,
        ground_truth_equations=gt_eqs,
        ground_truth_causal_adj=causal_adj,
        ground_truth_invariants=[],
        ground_truth_symmetries=[f"Cyclic shift Z_{dim}: x_i -> x_{{i+1}}"],
        true_divergence=f"-{dim}",
        rhs_fn=rhs,
        state_bounds=[(-10.0, 15.0)] * dim,
        metadata={"dim": dim, "forcing": forcing},
    )


def generate_synthetic_dataset(name: str, seed: int = 42, **kwargs: Any) -> DynamicalDataset | PDEDataset:
    """Factory for all synthetic ground-truth datasets."""
    key = name.lower().strip()
    registry: dict[str, Callable[..., DynamicalDataset | PDEDataset]] = {
        "lorenz": lambda: gen_lorenz(seed=seed, **kwargs),
        "lorenz96": lambda: gen_lorenz96(seed=seed, **kwargs),
        "lorenz96_8d": lambda: gen_lorenz96(dim=8, seed=seed, **kwargs),
        "lotka_volterra": lambda: gen_lotka_volterra(seed=seed, **kwargs),
        "sir": lambda: gen_sir(seed=seed, **kwargs),
        "sir_regime_drift": lambda: gen_sir_regime_drift(**kwargs),
        "pendulum": lambda: gen_pendulum(seed=seed, **kwargs),
        "damped_pendulum": lambda: gen_pendulum(damping=0.25, seed=seed, **kwargs),
        "kepler": lambda: gen_kepler_2body(seed=seed, **kwargs),
        "michaelis_menten": lambda: gen_michaelis_menten(seed=seed, **kwargs),
        "duffing": lambda: gen_duffing(seed=seed, **kwargs),
        "van_der_pol": lambda: gen_van_der_pol(seed=seed, **kwargs),
        "burgers_pde": lambda: gen_burgers_pde(**kwargs),
        "advection_diffusion_pde": lambda: gen_advection_diffusion_pde(**kwargs),
    }
    if key not in registry:
        raise ValueError(f"Unknown synthetic dataset '{name}'. Available: {sorted(registry.keys())}")
    return registry[key]()
