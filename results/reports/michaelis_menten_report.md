# ASDS-OMEGA v2.0 (VECSD) — Autonomous Scientific Discovery Report

- **System:** `michaelis_menten` (Biochemistry (Rational Enzyme Kinetics))
- **Provenance Config Hash:** `0a2414d2f55e0c5f` | **Seed:** `42` | **Runtime:** `2.594s`
- **Estimated Channel Noise (Wavelet-MAD):** `[0.03352, 0.03435]`

## 1. Discovered Governing Equations

| State Derivative | Discovered Symbolic Law | Weak $R^2$ | Complexity | MDL | Source |
|---|---|---:|---:|---:|---|
| `d(S)/dt` | `-26.2106*S/(17*S + 9)` | 0.9875 | 7 | -29.71 | `vesp_implicit_rational` |
| `d(P)/dt` | `120*S/(7*(12*S + 5))` | 0.9774 | 6 | -24.84 | `vesp_implicit_rational` |

## 2. Direct Differential Causal Graph (`VIFC`)

- **Differential SHD:** `2` | **Precision:** `0.500` | **Recall:** `1.000` | **F1:** `0.667`
- **Inferred Edges:** `S -> d(S)/dt` (0.95), `S -> d(P)/dt` (0.96), `P -> d(S)/dt` (0.47), `P -> d(P)/dt` (0.48)

## 3. Model-Free Symmetries, Conservation Laws & Dynamical Regime (`W-LIND`)

- **Regime Classification:** `dissipative_chaotic` (Divergence $\nabla\cdot f =$ `(-4010.2218*S - 2123.0586)/(17*S + 9)**3`, $\hat{\lambda}_1 =$ `1.520`)
- **Closed-Loop Search Space Pruned:** `0.0%`
- **Discrete Symmetries:** `[]`
- **Conserved Quantities / First Integrals:** `[{'expr': 'P + S', 'family': 'linear_conservation', 'weak_residual': 0.0594, 'cancellation_ratio': 0.0368}]`
- **Continuous Lie Generators:** `[{'type': 'translation', 'components': ['0', '1.00000000000000'], 'residual': 0.0}, {'type': 'affine_lie', 'components': ['0', 'P + S'], 'residual': 0.0009}, {'type': 'affine_lie', 'components': ['-S', '-0.1489*P + 0.161*S'], 'residual': 0.027}]`

## 4. Formal Verification & Derived Scientific Insights (`ASDS-Verify`)

- **Formal Proof Pass Rate:** `25.0%`
- State-dependent phase-space divergence: div(f) = (-4010.2218*S - 2123.0586)/(17*S + 9)**3.
