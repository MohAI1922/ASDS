"""Machine-Checkable Lean 4 Proof Certificate Exporter for Discovered Scientific Laws.

Compiles discovered symbolic ODEs, exact rational coefficients, conservation identities
(sum_i (dH/dx_i) f_i(x) = 0), and discrete Z_2 equivariances into syntactically valid
Lean 4 theorem files using standard Mathlib algebraic tactics (`ring`, `field_simp`).
"""

from __future__ import annotations

from pathlib import Path

import sympy as sp

from asds_omega.symmetry.reynolds import SymmetryDiscoveryResult
from asds_omega.verification.sympy_prover import FormalProofCertificate


def _sympy_to_lean(expr: sp.Expr) -> str:
    """Convert a SymPy algebraic expression into Lean 4 Real arithmetic syntax."""
    if expr.is_Integer:
        return f"({int(expr)} : ℝ)" if int(expr) < 0 else str(int(expr))
    if expr.is_Rational:
        return f"({int(expr.p)} : ℝ) / ({int(expr.q)} : ℝ)"
    if expr.is_Float:
        return f"({float(expr):.6f} : ℝ)"
    if isinstance(expr, sp.Symbol):
        return str(expr)
    if isinstance(expr, sp.Add):
        return "(" + " + ".join(_sympy_to_lean(a) for a in expr.args) + ")"
    if isinstance(expr, sp.Mul):
        return "(" + " * ".join(_sympy_to_lean(a) for a in expr.args) + ")"
    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        if exp.is_Integer and int(exp) >= 0:
            return f"({_sympy_to_lean(base)} ^ {int(exp)})"
        if exp == sp.Rational(-1, 1):
            return f"(1 / {_sympy_to_lean(base)})"
        return f"(Real.rpow ({_sympy_to_lean(base)}) ({_sympy_to_lean(exp)}))"
    if expr.func == sp.sin:
        return f"(Real.sin {_sympy_to_lean(expr.args[0])})"
    if expr.func == sp.cos:
        return f"(Real.cos {_sympy_to_lean(expr.args[0])})"
    if expr.func == sp.log:
        return f"(Real.log {_sympy_to_lean(expr.args[0])})"
    if expr.func == sp.exp:
        return f"(Real.exp {_sympy_to_lean(expr.args[0])})"
    return f"({expr})"


def export_lean4_certificate(
    system_name: str,
    var_names: list[str],
    equations: list[sp.Expr],
    symmetry_result: SymmetryDiscoveryResult,
    proof_cert: FormalProofCertificate,
    output_path: str | Path | None = None,
) -> str:
    """Generate a Lean 4 formal certificate file verifying discovered laws, invariants, and symmetries."""
    safe_ns = "".join(ch if ch.isalnum() else "_" for ch in system_name.title())
    args_sig = " ".join(f"({v} : ℝ)" for v in var_names)
    args_call = " ".join(var_names)
    syms = [sp.Symbol(v) for v in var_names]

    lines = [
        f"/-!",
        f"  ASDS-OMEGA v2.0 (VECSD) — Automated Lean 4 Formal Verification Certificate",
        f"  System: {system_name}",
        f"  Variables: {', '.join(var_names)}",
        f"  Symbolic Divergence: div(f) = {proof_cert.symbolic_divergence}",
        f"-/",
        "",
        "import Mathlib.Data.Real.Basic",
        "import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic",
        "import Mathlib.Tactic.Ring",
        "",
        f"namespace ASDSOmega.{safe_ns}",
        "",
        "-- Discovered Governing Vector Field f(x)",
    ]

    for v, eq in zip(var_names, equations):
        lines.append(f"noncomputable def f_{v} {args_sig} : ℝ :=")
        lines.append(f"  {_sympy_to_lean(eq)}")
        lines.append("")

    # Export Verified Conservation Law Theorems (strictly proved exact — zero `sorry` stubs!)
    for idx, (inv, pcheck) in enumerate(zip(symmetry_result.invariants, proof_cert.conservation_proofs)):
        if not pcheck.proved_exact or inv.expr.has(sp.log, sp.sqrt):
            continue
        H = inv.expr
        grad_terms = []
        for v, s in zip(var_names, syms):
            dH_dv = sp.diff(H, s)
            if dH_dv != 0:
                grad_terms.append(f"({_sympy_to_lean(dH_dv)}) * (f_{v} {args_call})")
        lhs = " + ".join(grad_terms) if grad_terms else "0"
        tactic = "unfold " + " ".join(f"f_{v}" for v in var_names) + "; ring"
        lines.append(f"/-- Conservation Law {idx + 1}: d/dt ({H}) = 0 -/")
        lines.append(f"theorem conservation_law_{idx + 1} {args_sig} :")
        lines.append(f"    {lhs} = 0 := by")
        lines.append(f"  {tactic}")
        lines.append("")

    # Export Discrete Equivariance Theorems
    for idx, (dsym, pcheck) in enumerate(zip(symmetry_result.discrete_symmetries, proof_cert.discrete_symmetry_proofs)):
        trans_args = " ".join(f"(-{v})" if sg == -1 else v for v, sg in zip(var_names, dsym.signs))
        for i, (v, sg) in enumerate(zip(var_names, dsym.signs)):
            rhs_term = f"-(f_{v} {args_call})" if sg == -1 else f"(f_{v} {args_call})"
            has_trig = equations[i].has(sp.sin, sp.cos)
            if pcheck.proved_exact and not has_trig and not equations[i].has(sp.Pow):
                tactic = f"unfold f_{v}; ring"
            elif pcheck.proved_exact and has_trig:
                tactic = f"unfold f_{v}; simp [Real.sin_neg, Real.cos_neg]; ring"
            elif pcheck.proved_exact:
                tactic = f"unfold f_{v}; ring"
            else:
                tactic = "sorry"
            lines.append(f"/-- Discrete Z2 Equivariance ({dsym.description}) for d({v})/dt -/")
            lines.append(f"theorem z2_equivariance_{idx + 1}_{v} {args_sig} :")
            lines.append(f"    f_{v} {trans_args} = {rhs_term} := by")
            lines.append(f"  {tactic}")
            lines.append("")

    lines.append(f"end ASDSOmega.{safe_ns}")
    lean_src = "\n".join(lines) + "\n"

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(lean_src, encoding="utf-8")

    return lean_src
