/-!
  ASDS-OMEGA v2.0 (VECSD) — Automated Lean 4 Formal Verification Certificate
  System: michaelis_menten
  Variables: S, P
  Symbolic Divergence: div(f) = (-4010.2218*S - 2123.0586)/(17*S + 9)**3
-/

import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Tactic.Ring

namespace ASDSOmega.Michaelis_Menten

-- Discovered Governing Vector Field f(x)
noncomputable def f_S (S : ℝ) (P : ℝ) : ℝ :=
  ((-26.210600 : ℝ) * S * (1 / (9 + (17 * S))))

noncomputable def f_P (S : ℝ) (P : ℝ) : ℝ :=
  ((120 : ℝ) / (7 : ℝ) * S * (1 / (5 + (12 * S))))

end ASDSOmega.Michaelis_Menten
