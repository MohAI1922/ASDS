"""Counterfactual Do-Calculus Intervention Simulator for Discovered Symbolic ODEs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp


@dataclass
class CounterfactualResult:
    """Outcome of a counterfactual intervention experiment."""

    t: np.ndarray
    baseline_trajectory: np.ndarray
    intervened_trajectory: np.ndarray
    delta_trajectory: np.ndarray
    ate_per_var: dict[str, float]
    peak_shift_per_var: dict[str, float]
    description: str


class CounterfactualSimulator:
    """Execute do(X_j = c), parameter-scaling, and IC interventions on discovered SymPy ODEs."""

    def __init__(self, var_names: list[str], equations: list[sp.Expr]) -> None:
        self.var_names = var_names
        self.equations = equations
        self.symbols = [sp.Symbol(v) for v in var_names]

    def _compile_rhs(
        self,
        exprs: list[sp.Expr],
        clamped: dict[int, float] | None = None,
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        funcs = [sp.lambdify(self.symbols, eq, modules="numpy") for eq in exprs]
        clamped = clamped or {}

        def rhs(_t: float, state: np.ndarray) -> np.ndarray:
            s = state.copy()
            for idx, val in clamped.items():
                s[idx] = val
            out = np.zeros(len(funcs), dtype=float)
            for i, fn in enumerate(funcs):
                if i in clamped:
                    out[i] = 0.0
                else:
                    val = float(fn(*s))
                    out[i] = float(np.clip(val, -1e6, 1e6)) if np.isfinite(val) else 0.0
            return out

        return rhs

    def simulate(
        self,
        x0: np.ndarray,
        t_eval: np.ndarray,
        clamped_vars: dict[str, float] | None = None,
        eq_overrides: dict[str, sp.Expr] | None = None,
        scale_equation: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Integrate discovered ODE system under optional interventions."""
        exprs = list(self.equations)
        if eq_overrides:
            for v, new_eq in eq_overrides.items():
                if v in self.var_names:
                    exprs[self.var_names.index(v)] = new_eq
        if scale_equation:
            for v, factor in scale_equation.items():
                if v in self.var_names:
                    idx = self.var_names.index(v)
                    exprs[idx] = sp.Float(factor) * exprs[idx]

        clamped_idx: dict[int, float] = {}
        x0_run = np.asarray(x0, dtype=float).copy()
        if clamped_vars:
            for v, val in clamped_vars.items():
                if v in self.var_names:
                    idx = self.var_names.index(v)
                    clamped_idx[idx] = float(val)
                    x0_run[idx] = float(val)

        rhs = self._compile_rhs(exprs, clamped=clamped_idx)
        sol = solve_ivp(
            rhs,
            (float(t_eval[0]), float(t_eval[-1])),
            x0_run,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )
        if not sol.success or sol.y.shape[1] != len(t_eval):
            out = np.full((len(t_eval), len(self.var_names)), np.nan)
            out[: sol.y.shape[1]] = sol.y.T
            return out
        return sol.y.T

    def run_intervention(
        self,
        x0: np.ndarray,
        t_eval: np.ndarray,
        clamped_vars: dict[str, float] | None = None,
        scale_equation: dict[str, float] | None = None,
        x0_intervened: np.ndarray | None = None,
        description: str = "Counterfactual intervention",
    ) -> CounterfactualResult:
        """Compare baseline trajectory vs intervened trajectory and compute ATE metrics."""
        base = self.simulate(x0, t_eval)
        x0_int = x0_intervened if x0_intervened is not None else x0
        inter = self.simulate(x0_int, t_eval, clamped_vars=clamped_vars, scale_equation=scale_equation)
        delta = inter - base
        ate = {v: float(np.nanmean(delta[:, i])) for i, v in enumerate(self.var_names)}
        peak_shift = {
            v: float(np.nanmax(inter[:, i]) - np.nanmax(base[:, i]))
            for i, v in enumerate(self.var_names)
        }
        return CounterfactualResult(
            t=t_eval,
            baseline_trajectory=base,
            intervened_trajectory=inter,
            delta_trajectory=delta,
            ate_per_var=ate,
            peak_shift_per_var=peak_shift,
            description=description,
        )
