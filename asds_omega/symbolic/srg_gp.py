"""Stage B of EVG-HE: Semantic Residual-Guided Multi-Objective Weak-Form NSGA-II GP + VarPro.

Key Algorithmic Innovations:
1. Weak-Form Fitness Evaluation: Every candidate expression tree T_theta(x) is evaluated
   directly inside the MS-AVP weak-form integral <phi_k, T_theta(x)>, making genetic
   programming immune to noisy derivative targets.
2. Semantic Residual-Guided Mutation: Computes the weak-form residual vector r = v - v_hat
   of a parent tree, ranks atomic features by correlation |corr(r, g_m)|, and grafts the
   most residual-aligned subtree onto the tree — dramatically accelerating convergence over
   blind random subtree swaps.
3. Variable Projection (VarPro): Solves outer linear scaling/bias analytically in closed
   form while optimizing inner nonlinear constants via bounded L-BFGS-B.
4. Reynolds-Projected, Causally-Masked Typed Grammar + NSGA-II Pareto Front over
   (weak_mse, mdl_complexity, symmetry_penalty).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy.optimize import minimize

from asds_omega.config import EVGHEConfig
from asds_omega.preprocessing.ms_avp import WeakFormSubstrate
from asds_omega.symmetry.reynolds import ReynoldsEquivariantProjector
from asds_omega.symbolic.rationalizer import (
    compute_mdl_score,
    expression_complexity,
    rationalize_expression,
)
from asds_omega.symbolic.vesp import CandidateEquation, VESPEngine


@dataclass
class _Node:
    """Compact symbolic AST node supporting vectorized NumPy evaluation & SymPy export."""

    op: str  # "var", "const", "+", "-", "*", "/", "sin", "cos"
    value: int | float | None = None
    left: _Node | None = None
    right: _Node | None = None

    def copy(self) -> _Node:
        return copy.deepcopy(self)

    def depth(self) -> int:
        if self.op in ("var", "const"):
            return 1
        ld = self.left.depth() if self.left else 0
        rd = self.right.depth() if self.right else 0
        return 1 + max(ld, rd)

    def size(self) -> int:
        if self.op in ("var", "const"):
            return 1
        return 1 + (self.left.size() if self.left else 0) + (self.right.size() if self.right else 0)

    def collect_const_nodes(self) -> list[_Node]:
        out: list[_Node] = []
        if self.op == "const":
            out.append(self)
        if self.left:
            out.extend(self.left.collect_const_nodes())
        if self.right:
            out.extend(self.right.collect_const_nodes())
        return out

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        if self.op == "var":
            return X[:, int(self.value)]
        if self.op == "const":
            return np.full(X.shape[0], float(self.value), dtype=float)
        if self.op == "sin":
            return np.sin(np.clip(self.left.evaluate(X), -50.0, 50.0))
        if self.op == "cos":
            return np.cos(np.clip(self.left.evaluate(X), -50.0, 50.0))
        lv = self.left.evaluate(X)
        rv = self.right.evaluate(X)
        if self.op == "+":
            return lv + rv
        if self.op == "-":
            return lv - rv
        if self.op == "*":
            return np.clip(lv * rv, -1e8, 1e8)
        if self.op == "/":
            denom = np.where(np.abs(rv) < 0.08, np.where(rv >= 0, 0.08, -0.08), rv)
            return np.clip(lv / denom, -1e8, 1e8)
        raise ValueError(f"Unknown op {self.op}")

    def to_sympy(self, syms: list[sp.Symbol]) -> sp.Expr:
        if self.op == "var":
            return syms[int(self.value)]
        if self.op == "const":
            return sp.Float(round(float(self.value), 5))
        if self.op == "sin":
            return sp.sin(self.left.to_sympy(syms))
        if self.op == "cos":
            return sp.cos(self.left.to_sympy(syms))
        le = self.left.to_sympy(syms)
        re = self.right.to_sympy(syms)
        if self.op == "+":
            return le + re
        if self.op == "-":
            return le - re
        if self.op == "*":
            return le * re
        if self.op == "/":
            return le / re
        raise ValueError(f"Unknown op {self.op}")


@dataclass
class GPIndividual:
    """Evaluated GP individual with multi-objective fitness values."""

    tree: _Node
    scale: float
    bias: float
    expr: sp.Expr
    weak_mse: float
    weak_r2: float
    complexity: int
    symmetry_penalty: float
    mdl_score: float
    weak_pred: np.ndarray


def _non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """Standard NSGA-II non-dominated sorting (minimization on all columns)."""
    n = len(objectives)
    dom_count = np.zeros(n, dtype=int)
    dom_sets: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if np.all(objectives[p] <= objectives[q]) and np.any(objectives[p] < objectives[q]):
                dom_sets[p].append(q)
            elif np.all(objectives[q] <= objectives[p]) and np.any(objectives[q] < objectives[p]):
                dom_count[p] += 1
        if dom_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in dom_sets[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return [f for f in fronts if f]


class SemanticWeakFormGP:
    """Semantic Residual-Guided Multi-Objective Weak-Form GP with VarPro Optimization."""

    def __init__(self, config: EVGHEConfig | None = None, seed: int = 42) -> None:
        self.config = config or EVGHEConfig()
        self.rng = np.random.default_rng(seed)

    def _random_tree(self, max_depth: int, allowed_vars: list[int], allow_trig: bool) -> _Node:
        if max_depth <= 1 or self.rng.random() < 0.35:
            if self.rng.random() < 0.78:
                return _Node("var", int(self.rng.choice(allowed_vars)))
            return _Node("const", float(self.rng.uniform(-2.5, 2.5)))
        unary_ops = ["sin", "cos"] if allow_trig else []
        bin_ops = ["+", "-", "*", "/"]
        if unary_ops and self.rng.random() < 0.22:
            op = str(self.rng.choice(unary_ops))
            return _Node(op, left=self._random_tree(max_depth - 1, allowed_vars, allow_trig=False))
        op = str(self.rng.choice(bin_ops))
        return _Node(
            op,
            left=self._random_tree(max_depth - 1, allowed_vars, allow_trig),
            right=self._random_tree(max_depth - 1, allowed_vars, allow_trig),
        )

    def _eval_and_varpro(
        self,
        tree: _Node,
        substrate: WeakFormSubstrate,
        target_idx: int,
        reynolds: ReynoldsEquivariantProjector | None,
        optimize_inner: bool = False,
    ) -> GPIndividual:
        """Evaluate tree in weak form with closed-form outer affine VarPro + optional L-BFGS inner tuning."""
        y = substrate.weak_velocities[:, target_idx]
        y_var = float(np.var(y)) + 1e-12
        K = substrate.n_subdomains
        k_eff = VESPEngine._effective_samples(substrate)
        syms = [sp.Symbol(v) for v in substrate.var_names]

        def _project_tree(nd: _Node) -> np.ndarray:
            return substrate.project_callable(nd.evaluate)

        def _affine_fit(raw_w: np.ndarray) -> tuple[float, float, np.ndarray, float]:
            if not np.all(np.isfinite(raw_w)) or float(np.std(raw_w)) < 1e-10:
                return 0.0, float(np.mean(y)), np.full(K, float(np.mean(y))), y_var
            # Test zero-bias scaling first (parsimony) vs affine scaling
            a0 = float(np.dot(raw_w, y) / (np.dot(raw_w, raw_w) + 1e-12))
            p0 = a0 * raw_w
            mse0 = float(np.mean((y - p0) ** 2))

            A = np.column_stack([raw_w, np.ones(K, dtype=float)])
            ab, *_ = np.linalg.lstsq(A, y, rcond=None)
            p1 = A @ ab
            mse1 = float(np.mean((y - p1) ** 2))
            if mse1 < mse0 * 0.92:
                return float(ab[0]), float(ab[1]), p1, mse1
            return a0, 0.0, p0, mse0

        # Optional L-BFGS-B optimization of inner constants (Variable Projection!)
        t_use = tree.copy()
        consts = t_use.collect_const_nodes()
        if optimize_inner and 1 <= len(consts) <= 4:
            c0 = np.array([float(nd.value) for nd in consts], dtype=float)

            def _obj(vec: np.ndarray) -> float:
                for nd, v in zip(consts, vec):
                    nd.value = float(v)
                rw = _project_tree(t_use)
                _, _, _, m = _affine_fit(rw)
                return m

            res = minimize(_obj, c0, method="L-BFGS-B", bounds=[(-15.0, 15.0)] * len(c0), options={"maxiter": 15})
            for nd, v in zip(consts, res.x):
                nd.value = float(v)

        raw_w = _project_tree(t_use)
        scale, bias, pred, mse = _affine_fit(raw_w)
        r2 = float(1.0 - mse / y_var)

        sym_raw = t_use.to_sympy(syms)
        expr = sp.Float(round(scale, 5)) * sym_raw + (sp.Float(round(bias, 5)) if abs(bias) > 1e-4 else sp.Integer(0))
        try:
            expr = sp.simplify(expr)
        except Exception:
            pass
        if expr.has(sp.zoo, sp.oo, -sp.oo, sp.nan, sp.I):
            expr = sp.Float(round(float(np.mean(y)), 5))
            mse = y_var
            r2 = 0.0
        else:
            # Nonsingular denominator guard: reject symbolic divisions whose denominator approaches 0 on data
            _, denom_expr = expr.as_numer_denom()
            if len(denom_expr.free_symbols) > 0:
                try:
                    d_fn = sp.lambdify(syms, denom_expr, modules="numpy")
                    all_X = np.vstack(substrate.trajectories)
                    d_vals = np.asarray(d_fn(*[all_X[:, j] for j in range(all_X.shape[1])]), dtype=float)
                    if not np.all(np.isfinite(d_vals)) or float(np.min(np.abs(d_vals))) < 0.08:
                        expr = sp.Float(round(float(np.mean(y)), 5))
                        mse = y_var
                        r2 = 0.0
                except Exception:
                    expr = sp.Float(round(float(np.mean(y)), 5))
                    mse = y_var
                    r2 = 0.0

        # Symmetry penalty via Reynolds projector
        sym_pen = 0.0
        if reynolds is not None and not reynolds.is_trivial:
            try:
                proj_expr = reynolds.project_expression(expr, target_idx)
                if sp.simplify(proj_expr - expr) != 0:
                    sym_pen = 0.25
            except Exception:
                sym_pen = 0.25

        comp = expression_complexity(expr)
        mdl = compute_mdl_score(mse, k_eff, expr, y_var, symmetry_penalty=sym_pen)
        return GPIndividual(
            tree=t_use,
            scale=scale,
            bias=bias,
            expr=expr,
            weak_mse=mse,
            weak_r2=r2,
            complexity=comp,
            symmetry_penalty=sym_pen,
            mdl_score=mdl,
            weak_pred=pred,
        )

    def _semantic_mutate(
        self,
        parent: GPIndividual,
        atomic_nodes: list[tuple[_Node, np.ndarray]],
        y_target: np.ndarray,
        allowed_vars: list[int],
        allow_trig: bool,
    ) -> _Node:
        """Semantic Residual-Guided Mutation: grafts the atomic term most correlated with weak residual."""
        if self.rng.random() < self.config.gp_semantic_mutation_rate and atomic_nodes:
            residual = y_target - parent.weak_pred
            res_norm = float(np.linalg.norm(residual)) + 1e-12
            corrs = []
            for _, g_col in atomic_nodes:
                c = abs(float(np.dot(residual, g_col))) / (res_norm * (float(np.linalg.norm(g_col)) + 1e-12))
                corrs.append(c)
            corrs_arr = np.asarray(corrs, dtype=float)
            # Softmax sampling over top residual-correlated atomic subtrees
            logits = 8.0 * (corrs_arr - np.max(corrs_arr))
            probs = np.exp(logits)
            probs /= np.sum(probs)
            chosen_node = atomic_nodes[int(self.rng.choice(len(atomic_nodes), p=probs))][0].copy()
            # Combine with parent via addition or multiplication
            op = "+" if self.rng.random() < 0.78 else "*"
            # Apply parent's learned scale inside the combined tree so residual alignment holds
            scaled_parent = _Node("*", left=_Node("const", parent.scale), right=parent.tree.copy())
            cand = _Node(op, left=scaled_parent, right=chosen_node)
            if cand.depth() <= self.config.gp_max_depth + 1:
                return cand

        # Fallback: standard structural point/subtree mutation
        child = parent.tree.copy()
        if self.rng.random() < 0.5:
            return self._random_tree(min(3, self.config.gp_max_depth), allowed_vars, allow_trig)
        if child.left is not None and self.rng.random() < 0.5:
            child.left = self._random_tree(max(1, self.config.gp_max_depth - 2), allowed_vars, allow_trig)
        elif child.right is not None:
            child.right = self._random_tree(max(1, self.config.gp_max_depth - 2), allowed_vars, allow_trig)
        else:
            return self._random_tree(2, allowed_vars, allow_trig)
        return child

    def _crossover(self, p1: _Node, p2: _Node) -> _Node:
        c1 = p1.copy()
        c2 = p2.copy()
        if c1.left is not None and c2.right is not None:
            c1.left = c2.right
        elif c1.right is not None and c2.left is not None:
            c1.right = c2.left
        return c1 if c1.depth() <= self.config.gp_max_depth + 1 else p1.copy()

    def evolve_target(
        self,
        substrate: WeakFormSubstrate,
        target_idx: int,
        allowed_vars: list[int],
        reynolds: ReynoldsEquivariantProjector | None = None,
        allow_trig: bool = True,
        generations: int | None = None,
    ) -> list[CandidateEquation]:
        """Run Semantic Residual-Guided Weak-Form NSGA-II GP for dx_{target_idx}/dt."""
        y = substrate.weak_velocities[:, target_idx]
        y_var = float(np.var(y)) + 1e-12
        K = substrate.n_subdomains
        k_eff = VESPEngine._effective_samples(substrate)
        pop_size = self.config.gp_population_size
        n_gen = generations if generations is not None else self.config.gp_generations

        # Build atomic semantic library for residual-guided grafting
        atomic_nodes: list[tuple[_Node, np.ndarray]] = []
        for v in allowed_vars:
            nd = _Node("var", v)
            atomic_nodes.append((nd, substrate.project_callable(nd.evaluate)))
            if allow_trig:
                for trig_op in ("sin", "cos"):
                    tnd = _Node(trig_op, left=_Node("var", v))
                    atomic_nodes.append((tnd, substrate.project_callable(tnd.evaluate)))
        for idx_a in range(len(allowed_vars)):
            for idx_b in range(idx_a, len(allowed_vars)):
                pnd = _Node("*", left=_Node("var", allowed_vars[idx_a]), right=_Node("var", allowed_vars[idx_b]))
                atomic_nodes.append((pnd, substrate.project_callable(pnd.evaluate)))

        # Initialize population (seeded with atomic nodes + random trees)
        trees: list[_Node] = [nd.copy() for nd, _ in atomic_nodes[: pop_size // 2]]
        while len(trees) < pop_size:
            trees.append(self._random_tree(self.config.gp_max_depth, allowed_vars, allow_trig))

        pop = [self._eval_and_varpro(t, substrate, target_idx, reynolds, optimize_inner=False) for t in trees]

        for gen in range(n_gen):
            offspring: list[GPIndividual] = []
            # Sort by MDL for tournament parent selection
            pop.sort(key=lambda ind: ind.mdl_score)
            elites = pop[: max(4, pop_size // 4)]

            for _ in range(pop_size // 2):
                p1 = elites[int(self.rng.integers(0, len(elites)))]
                if self.rng.random() < 0.72:
                    child_tree = self._semantic_mutate(p1, atomic_nodes, y, allowed_vars, allow_trig)
                else:
                    p2 = elites[int(self.rng.integers(0, len(elites)))]
                    child_tree = self._crossover(p1.tree, p2.tree)
                tune_inner = (gen >= n_gen - 4)
                offspring.append(
                    self._eval_and_varpro(child_tree, substrate, target_idx, reynolds, optimize_inner=tune_inner)
                )

            combined = pop + offspring
            objs = np.array(
                [[ind.weak_mse, float(ind.complexity), ind.symmetry_penalty] for ind in combined],
                dtype=float,
            )
            fronts = _non_dominated_sort(objs)
            next_pop: list[GPIndividual] = []
            for f in fronts:
                f_sorted = sorted(f, key=lambda idx: combined[idx].mdl_score)
                for idx in f_sorted:
                    if len(next_pop) < pop_size:
                        next_pop.append(combined[idx])
            pop = next_pop

        # VarPro L-BFGS polish + MDL rationalization on top-4 Pareto candidates
        pop.sort(key=lambda ind: ind.mdl_score)
        out_cands: list[CandidateEquation] = []
        seen_str: set[str] = set()
        syms = [sp.Symbol(v) for v in substrate.var_names]

        for ind in pop[:5]:
            polished = self._eval_and_varpro(ind.tree, substrate, target_idx, reynolds, optimize_inner=True)
            expr, mse = polished.expr, polished.weak_mse
            if self.config.enable_rationalizer:
                def _eval_ex(ex: sp.Expr) -> float:
                    try:
                        fn = sp.lambdify(syms, ex, modules="numpy")
                        pred = substrate.project_callable(
                            lambda X: np.full(X.shape[0], float(fn(*[X[:, j] for j in range(X.shape[1])])))
                            if np.ndim(fn(*[X[:, j] for j in range(X.shape[1])])) == 0
                            else np.asarray(fn(*[X[:, j] for j in range(X.shape[1])]), dtype=float)
                        )
                        return float(np.mean((y - pred) ** 2)) if np.all(np.isfinite(pred)) else 1e9
                    except Exception:
                        return 1e9

                expr_r, mse_r = rationalize_expression(
                    expr, _eval_ex, k_eff, y_var, max_denom=self.config.max_rational_denom
                )
                if mse_r < 1e8:
                    expr, mse = expr_r, mse_r

            s_key = str(expr)
            if s_key in seen_str:
                continue
            seen_str.add(s_key)
            r2 = float(1.0 - mse / y_var)
            out_cands.append(
                CandidateEquation(
                    target_var=substrate.var_names[target_idx],
                    expr=expr,
                    expr_str=s_key,
                    weak_mse=mse,
                    weak_r2=r2,
                    mdl_score=compute_mdl_score(mse, k_eff, expr, y_var, symmetry_penalty=polished.symmetry_penalty),
                    complexity=expression_complexity(expr),
                    symmetry_residual=polished.symmetry_penalty,
                    source="srg_gp_varpro",
                )
            )
        out_cands.sort(key=lambda c: c.mdl_score)
        return out_cands
