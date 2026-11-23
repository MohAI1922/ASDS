# ASDS-OMEGA v2.0 (VECSD) — Autonomous Scientific Discovery Report

- **System:** `van_der_pol` (Nonlinear Dynamics (Limit-Cycle Oscillator))
- **Provenance Config Hash:** `0a2414d2f55e0c5f` | **Seed:** `42` | **Runtime:** `1.050s`
- **Estimated Channel Noise (Wavelet-MAD):** `[0.06909, 0.07272]`

## 1. Discovered Governing Equations

| State Derivative | Discovered Symbolic Law | Weak $R^2$ | Complexity | MDL | Source |
|---|---|---:|---:|---:|---|
| `d(x)/dt` | `y` | 0.9967 | 2 | -208.55 | `vesp_sparse` |
| `d(y)/dt` | `-3*x**2*y/2 - x + 3*y/2` | 0.9973 | 11 | -193.84 | `vesp_tier3` |

## 2. Direct Differential Causal Graph (`VIFC`)

- **Differential SHD:** `0` | **Precision:** `1.000` | **Recall:** `1.000` | **F1:** `1.000`
- **Inferred Edges:** `x -> d(y)/dt` (0.97), `y -> d(x)/dt` (0.95), `y -> d(y)/dt` (0.97)

## 3. Model-Free Symmetries, Conservation Laws & Dynamical Regime (`W-LIND`)

- **Regime Classification:** `dissipative_chaotic` (Divergence $\nabla\cdot f =$ `3/2 - 3*x**2/2`, $\hat{\lambda}_1 =$ `1.753`)
- **Closed-Loop Search Space Pruned:** `60.7%`
- **Discrete Symmetries:** `[{'description': 'Z2_reflection: (x, y) -> (-x, -y)', 'signs': [-1, -1], 'error': 0.0015}]`
- **Conserved Quantities / First Integrals:** `[{'expr': '0.3569*x**2 + cos(x)', 'family': 'hamiltonian_trigonometric', 'weak_residual': 0.0591, 'cancellation_ratio': 0.0653}]`
- **Continuous Lie Generators:** `[]`

## 4. Formal Verification & Derived Scientific Insights (`ASDS-Verify`)

- **Formal Proof Pass Rate:** `50.0%`
- State-dependent phase-space divergence: div(f) = 3/2 - 3*x**2/2.
- Formally verified discrete equivariance: Z2_reflection: (x, y) -> (-x, -y).
