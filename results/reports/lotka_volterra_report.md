# ASDS-OMEGA v2.0 (VECSD) — Autonomous Scientific Discovery Report

- **System:** `lotka_volterra` (Ecology (Predator-Prey Dynamics))
- **Provenance Config Hash:** `0a2414d2f55e0c5f` | **Seed:** `42` | **Runtime:** `3.131s`
- **Estimated Channel Noise (Wavelet-MAD):** `[0.01801, 0.06096]`

## 1. Discovered Governing Equations

| State Derivative | Discovered Symbolic Law | Weak $R^2$ | Complexity | MDL | Source |
|---|---|---:|---:|---:|---|
| `d(x)/dt` | `x*(11 - 4*y)/10` | 0.9910 | 7 | -191.51 | `vesp_sparse` |
| `d(y)/dt` | `y*(4*x - 1)/10` | 0.9781 | 7 | -153.22 | `vesp_sparse` |

## 2. Direct Differential Causal Graph (`VIFC`)

- **Differential SHD:** `0` | **Precision:** `1.000` | **Recall:** `1.000` | **F1:** `1.000`
- **Inferred Edges:** `x -> d(x)/dt` (0.66), `x -> d(y)/dt` (0.90), `y -> d(x)/dt` (0.85), `y -> d(y)/dt` (0.60)

## 3. Model-Free Symmetries, Conservation Laws & Dynamical Regime (`W-LIND`)

- **Regime Classification:** `dissipative_chaotic` (Divergence $\nabla\cdot f =$ `2*x/5 - 2*y/5 + 1`, $\hat{\lambda}_1 =$ `0.588`)
- **Closed-Loop Search Space Pruned:** `0.0%`
- **Discrete Symmetries:** `[]`
- **Conserved Quantities / First Integrals:** `[]`
- **Continuous Lie Generators:** `[{'type': 'affine_lie', 'components': ['0.0829*x - 0.0529', '0.1797*x + 0.1546*y - 1.0'], 'residual': 0.0156}, {'type': 'affine_lie', 'components': ['x - 0.2612', '-x + 0.5767*y'], 'residual': 0.0365}]`

## 4. Formal Verification & Derived Scientific Insights (`ASDS-Verify`)

- **Formal Proof Pass Rate:** `0.0%`
- State-dependent phase-space divergence: div(f) = 2*x/5 - 2*y/5 + 1.
