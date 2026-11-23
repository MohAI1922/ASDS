/-!
  ASDS-OMEGA v2.0 (VECSD) — Automated Lean 4 Formal Verification Certificate
  System: van_der_pol
  Variables: x, y
  Symbolic Divergence: div(f) = 3/2 - 3*x**2/2
-/

import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Tactic.Ring

namespace ASDSOmega.Van_Der_Pol

-- Discovered Governing Vector Field f(x)
noncomputable def f_x (x : ℝ) (y : ℝ) : ℝ :=
  y

noncomputable def f_y (x : ℝ) (y : ℝ) : ℝ :=
  (((-1 : ℝ) * x) + ((3 : ℝ) / (2 : ℝ) * y) + ((-3 : ℝ) / (2 : ℝ) * y * (x ^ 2)))

/-- Discrete Z2 Equivariance (Z2_reflection: (x, y) -> (-x, -y)) for d(x)/dt -/
theorem z2_equivariance_1_x (x : ℝ) (y : ℝ) :
    f_x (-x) (-y) = -(f_x x y) := by
  unfold f_x; ring

/-- Discrete Z2 Equivariance (Z2_reflection: (x, y) -> (-x, -y)) for d(y)/dt -/
theorem z2_equivariance_1_y (x : ℝ) (y : ℝ) :
    f_y (-x) (-y) = -(f_y x y) := by
  unfold f_y; ring

end ASDSOmega.Van_Der_Pol
