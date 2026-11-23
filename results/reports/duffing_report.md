# ASDS-OMEGA v2.0 (VECSD) — Autonomous Scientific Discovery Report

- **System:** `duffing` (Nonlinear Mechanics (Cubic Double-Well Oscillator))
- **Provenance Config Hash:** `0a2414d2f55e0c5f` | **Seed:** `42` | **Runtime:** `1.483s`
- **Estimated Channel Noise (Wavelet-MAD):** `[0.03714, 0.02591]`

## 1. Discovered Governing Equations

| State Derivative | Discovered Symbolic Law | Weak $R^2$ | Complexity | MDL | Source |
|---|---|---:|---:|---:|---|
| `d(x)/dt` | `v` | 0.9923 | 2 | -168.89 | `vesp_sparse` |
| `d(v)/dt` | `-v/5 - x**3 + x` | 0.9969 | 7 | -185.74 | `vesp_tier3_escape` |

## 2. Direct Differential Causal Graph (`VIFC`)

- **Differential SHD:** `1` | **Precision:** `1.000` | **Recall:** `0.667` | **F1:** `0.800`
- **Inferred Edges:** `x -> d(v)/dt` (0.98), `v -> d(x)/dt` (0.95)

## 3. Model-Free Symmetries, Conservation Laws & Dynamical Regime (`W-LIND`)

- **Regime Classification:** `dissipative_chaotic` (Divergence $\nabla\cdot f =$ `-1/5`, $\hat{\lambda}_1 =$ `0.597`)
- **Closed-Loop Search Space Pruned:** `78.6%`
- **Discrete Symmetries:** `[{'description': 'Z2_reflection: (x, v) -> (-x, -v)', 'signs': [-1, -1], 'error': 0.0007}]`
- **Conserved Quantities / First Integrals:** `[]`
- **Continuous Lie Generators:** `[]`

## 4. Formal Verification & Derived Scientific Insights (`ASDS-Verify`)

- **Formal Proof Pass Rate:** `100.0%`
- Constant phase-space volume contraction rate div(f) = -1/5 (-0.2000 < 0): trajectories collapse onto a lower-dimensional attractor.
- Formally verified discrete equivariance: Z2_reflection: (x, v) -> (-x, -v).
