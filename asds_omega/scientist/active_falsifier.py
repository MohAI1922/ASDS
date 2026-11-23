"""Popperian Active Falsification Engine (PAFE) — Optimal Experimental Design.

When multiple rival Pareto-optimal symbolic theories H = {f^(1), ..., f^(K)} explain
existing observations similarly well, PAFE autonomously selects the next initial
condition x_0^* that MAXIMIZES expected pairwise trajectory disagreement across
the rival ensemble, executes the oracle experiment, updates Bayesian model posteriors,
and falsifies wrong competing theories in minimal experimental queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from asds_omega.causal.counterfactual import CounterfactualSimulator
from asds_omega.data.synthetic import DynamicalDataset
from asds_omega.preprocessing.ms_avp import MSAVPProjector
from asds_omega.symbolic.vesp import CandidateEquation


@dataclass
class ActiveExperimentStep:
    """Single closed-loop Popperian falsification iteration record."""

    step: int
    selected_x0: list[float]
    max_disagreement: float
    posterior_entropy_before: float
    posterior_entropy_after: float
    winning_hypothesis_idx: int
    hypotheses_posteriors: list[float]


@dataclass
class ActiveFalsificationResult:
    """Outcome of a multi-step Popperian Active Falsification campaign."""

    n_hypotheses_initial: int
    steps: list[ActiveExperimentStep]
    final_posteriors: list[float]
    best_hypothesis_equations: list[str]
    entropy_reduction: float
    random_baseline_entropy_reduction: float
    hypotheses_descriptions: list[list[str]] = field(default_factory=list)


def _shannon_entropy(p: np.ndarray) -> float:
    p_c = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p_c * np.log2(p_c)))


class PopperianActiveFalsifier:
    """Optimal experimental design engine maximizing epistemic trajectory disagreement."""

    def __init__(self, seed: int = 42, n_candidates: int = 24) -> None:
        self.rng = np.random.default_rng(seed)
        self.n_candidates = n_candidates

    def _assemble_rival_systems(
        self,
        pareto_per_var: list[list[CandidateEquation]],
        max_systems: int = 5,
    ) -> list[list[sp.Expr]]:
        """Assemble distinct competing full-system ODE hypotheses from per-variable Pareto fronts."""
        d = len(pareto_per_var)
        best_base = [front[0].expr for front in pareto_per_var]
        systems: list[list[sp.Expr]] = [best_base]
        seen = {tuple(str(e) for e in best_base)}

        # Create competing system hypotheses by swapping individual equations with Pareto rivals
        for rank in range(1, 4):
            for i in range(d):
                if rank < len(pareto_per_var[i]):
                    alt = list(best_base)
                    alt[i] = pareto_per_var[i][rank].expr
                    key = tuple(str(e) for e in alt)
                    if key not in seen:
                        seen.add(key)
                        systems.append(alt)
                        if len(systems) >= max_systems:
                            return systems
        return systems

    def _sample_candidate_ics(
        self,
        dataset: DynamicalDataset,
        linear_conservation: np.ndarray | None,
        n: int,
    ) -> list[np.ndarray]:
        """Sample physically valid candidate initial conditions respecting state bounds & invariants."""
        d = dataset.n_vars
        if dataset.state_bounds is not None:
            lo = np.array([b[0] for b in dataset.state_bounds], dtype=float)
            hi = np.array([b[1] for b in dataset.state_bounds], dtype=float)
        else:
            all_X = np.vstack(dataset.trajectories)
            lo = np.percentile(all_X, 5, axis=0)
            hi = np.percentile(all_X, 95, axis=0)

        ics: list[np.ndarray] = []
        # Detect simplex-like conservation (e.g. S + I + R = 1)
        is_simplex = (
            linear_conservation is not None
            and linear_conservation.shape[0] == 1
            and np.allclose(linear_conservation[0], np.ones(d), atol=0.15)
            and float(np.min(lo)) >= -1e-6
        )
        for _ in range(n):
            if is_simplex:
                raw = self.rng.dirichlet(np.ones(d))
                ics.append(raw)
            else:
                ics.append(self.rng.uniform(lo, hi))
        return ics

    def _evaluate_system_weak_loss(
        self,
        eqs: list[sp.Expr],
        var_names: list[str],
        t_new: np.ndarray,
        X_new: np.ndarray,
    ) -> float:
        """Compute normalized weak-form MSE of a candidate ODE system on new experimental data."""
        sub = MSAVPProjector().fit_transform([t_new], [X_new], var_names)
        syms = [sp.Symbol(v) for v in var_names]
        losses = []
        for i, eq in enumerate(eqs):
            y = sub.weak_velocities[:, i]
            y_var = float(np.var(y)) + 1e-8
            try:
                fn = sp.lambdify(syms, eq, modules="numpy")
                pred = sub.project_callable(
                    lambda X, _f=fn: np.full(X.shape[0], float(_f(*[X[:, j] for j in range(X.shape[1])])))
                    if np.ndim(_f(*[X[:, j] for j in range(X.shape[1])])) == 0
                    else np.asarray(_f(*[X[:, j] for j in range(X.shape[1])]), dtype=float)
                )
                if np.all(np.isfinite(pred)):
                    losses.append(float(np.mean((y - pred) ** 2) / y_var))
                else:
                    losses.append(10.0)
            except Exception:
                losses.append(10.0)
        return float(np.mean(losses))

    def run_campaign(
        self,
        dataset: DynamicalDataset,
        pareto_per_var: list[list[CandidateEquation]],
        linear_conservation: np.ndarray | None = None,
        budget: int = 3,
        noise_std: float = 0.05,
    ) -> ActiveFalsificationResult:
        """Execute closed-loop Popperian falsification vs random-query baseline."""
        hypotheses = self._assemble_rival_systems(pareto_per_var)
        n_h = len(hypotheses)
        if n_h <= 1 or dataset.rhs_fn is None:
            return ActiveFalsificationResult(
                n_hypotheses_initial=n_h,
                steps=[],
                final_posteriors=[1.0],
                best_hypothesis_equations=[str(e) for e in hypotheses[0]],
                entropy_reduction=0.0,
                random_baseline_entropy_reduction=0.0,
                hypotheses_descriptions=[[str(e) for e in h] for h in hypotheses],
            )

        sims = [CounterfactualSimulator(dataset.var_names, eqs) for eqs in hypotheses]
        dt = float(np.median(np.diff(dataset.t_list[0])))
        t_horizon = min(6.0, float(dataset.t_list[0][-1] - dataset.t_list[0][0]) * 0.5)
        t_probe = np.arange(0.0, t_horizon + 1e-12, dt)

        # Uniform prior over surviving Pareto hypotheses
        log_post = np.zeros(n_h, dtype=float)
        log_post_rand = np.zeros(n_h, dtype=float)
        init_entropy = _shannon_entropy(np.ones(n_h) / n_h)

        steps: list[ActiveExperimentStep] = []
        for step in range(1, budget + 1):
            p_before = np.exp(log_post - np.max(log_post))
            p_before /= np.sum(p_before)
            ent_before = _shannon_entropy(p_before)

            cand_ics = self._sample_candidate_ics(dataset, linear_conservation, self.n_candidates)
            best_x0 = cand_ics[0]
            best_disagree = -1.0

            # Query-by-Committee: maximize posterior-weighted pairwise trajectory disagreement
            t_fast = t_probe[:: max(1, len(t_probe) // 40)]
            for x0 in cand_ics:
                trajs = []
                valid_all = True
                for sim in sims:
                    tr = sim.simulate(x0, t_fast)
                    if not np.all(np.isfinite(tr)) or float(np.max(np.abs(tr))) > 1e4:
                        valid_all = False
                        break
                    trajs.append(tr)
                if not valid_all:
                    continue
                scale = float(np.mean([np.std(tr) for tr in trajs])) + 1e-6
                disagree = 0.0
                for a in range(n_h):
                    for b in range(a + 1, n_h):
                        w_ab = float(p_before[a] * p_before[b]) + 0.02
                        diff = float(np.sqrt(np.mean((trajs[a] - trajs[b]) ** 2))) / scale
                        disagree += w_ab * min(diff, 10.0)
                if disagree > best_disagree:
                    best_disagree = disagree
                    best_x0 = x0

            # Execute maximally falsifying oracle experiment
            t_new, X_new = dataset.simulate_oracle(
                best_x0, t_span=(0.0, t_horizon), dt=dt, noise_std=noise_std, rng=self.rng
            )
            losses = np.array(
                [self._evaluate_system_weak_loss(h, dataset.var_names, t_new, X_new) for h in hypotheses],
                dtype=float,
            )
            # Bayesian weak-form likelihood update
            beta_temp = 18.0
            log_post = log_post - beta_temp * (losses - np.min(losses))
            p_after = np.exp(log_post - np.max(log_post))
            p_after /= np.sum(p_after)
            ent_after = _shannon_entropy(p_after)

            # Fair paired random-query baseline
            rand_x0 = cand_ics[int(self.rng.integers(0, len(cand_ics)))]
            t_r, X_r = dataset.simulate_oracle(
                rand_x0, t_span=(0.0, t_horizon), dt=dt, noise_std=noise_std, rng=self.rng
            )
            losses_r = np.array(
                [self._evaluate_system_weak_loss(h, dataset.var_names, t_r, X_r) for h in hypotheses],
                dtype=float,
            )
            log_post_rand = log_post_rand - beta_temp * (losses_r - np.min(losses_r))

            steps.append(
                ActiveExperimentStep(
                    step=step,
                    selected_x0=[round(float(v), 4) for v in best_x0],
                    max_disagreement=float(max(best_disagree, 0.0)),
                    posterior_entropy_before=ent_before,
                    posterior_entropy_after=ent_after,
                    winning_hypothesis_idx=int(np.argmax(p_after)),
                    hypotheses_posteriors=[round(float(v), 4) for v in p_after],
                )
            )

        p_final = np.exp(log_post - np.max(log_post))
        p_final /= np.sum(p_final)
        p_rand = np.exp(log_post_rand - np.max(log_post_rand))
        p_rand /= np.sum(p_rand)
        win_idx = int(np.argmax(p_final))

        return ActiveFalsificationResult(
            n_hypotheses_initial=n_h,
            steps=steps,
            final_posteriors=[round(float(v), 4) for v in p_final],
            best_hypothesis_equations=[str(e) for e in hypotheses[win_idx]],
            entropy_reduction=float(init_entropy - _shannon_entropy(p_final)),
            random_baseline_entropy_reduction=float(init_entropy - _shannon_entropy(p_rand)),
            hypotheses_descriptions=[[str(e) for e in h] for h in hypotheses],
        )
