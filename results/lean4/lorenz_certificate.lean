/-!
  ASDS-OMEGA v2.0 (VECSD) — Automated Lean 4 Formal Verification Certificate
  System: lorenz
  Variables: x, y, z
  Symbolic Divergence: div(f) = -41/3
-/

import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Tactic.Ring

namespace ASDSOmega.Lorenz

-- Discovered Governing Vector Field f(x)
noncomputable def f_x (x : ℝ) (y : ℝ) (z : ℝ) : ℝ :=
  (((-10 : ℝ) * x) + (10 * y))

noncomputable def f_y (x : ℝ) (y : ℝ) (z : ℝ) : ℝ :=
  (((-1 : ℝ) * y) + (28 * x) + ((-1 : ℝ) * x * z))

noncomputable def f_z (x : ℝ) (y : ℝ) (z : ℝ) : ℝ :=
  (((-8 : ℝ) / (3 : ℝ) * z) + (x * y))

/-- Discrete Z2 Equivariance (Z2_reflection: (x, y, z) -> (-x, -y, z)) for d(x)/dt -/
theorem z2_equivariance_1_x (x : ℝ) (y : ℝ) (z : ℝ) :
    f_x (-x) (-y) z = -(f_x x y z) := by
  unfold f_x; ring

/-- Discrete Z2 Equivariance (Z2_reflection: (x, y, z) -> (-x, -y, z)) for d(y)/dt -/
theorem z2_equivariance_1_y (x : ℝ) (y : ℝ) (z : ℝ) :
    f_y (-x) (-y) z = -(f_y x y z) := by
  unfold f_y; ring

/-- Discrete Z2 Equivariance (Z2_reflection: (x, y, z) -> (-x, -y, z)) for d(z)/dt -/
theorem z2_equivariance_1_z (x : ℝ) (y : ℝ) (z : ℝ) :
    f_z (-x) (-y) z = (f_z x y z) := by
  unfold f_z; ring

end ASDSOmega.Lorenz
