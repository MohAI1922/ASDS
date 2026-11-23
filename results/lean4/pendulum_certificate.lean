/-!
  ASDS-OMEGA v2.0 (VECSD) — Automated Lean 4 Formal Verification Certificate
  System: pendulum
  Variables: theta, omega
  Symbolic Divergence: div(f) = 0
-/

import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Tactic.Ring

namespace ASDSOmega.Pendulum

-- Discovered Governing Vector Field f(x)
noncomputable def f_theta (theta : ℝ) (omega : ℝ) : ℝ :=
  omega

noncomputable def f_omega (theta : ℝ) (omega : ℝ) : ℝ :=
  ((-2 : ℝ) * (Real.sin theta))

/-- Discrete Z2 Equivariance (Z2_reflection: (theta, omega) -> (-theta, -omega)) for d(theta)/dt -/
theorem z2_equivariance_1_theta (theta : ℝ) (omega : ℝ) :
    f_theta (-theta) (-omega) = -(f_theta theta omega) := by
  unfold f_theta; ring

/-- Discrete Z2 Equivariance (Z2_reflection: (theta, omega) -> (-theta, -omega)) for d(omega)/dt -/
theorem z2_equivariance_1_omega (theta : ℝ) (omega : ℝ) :
    f_omega (-theta) (-omega) = -(f_omega theta omega) := by
  unfold f_omega; simp [Real.sin_neg, Real.cos_neg]; ring

end ASDSOmega.Pendulum
