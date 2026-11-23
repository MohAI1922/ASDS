# ASDS-OMEGA v2.0 (VECSD) — Autonomous Scientific Discovery Report

- **System:** `lorenz` (Physics (Chaotic Convection))
- **Provenance Config Hash:** `0a2414d2f55e0c5f` | **Seed:** `42` | **Runtime:** `5.955s`
- **Estimated Channel Noise (Wavelet-MAD):** `[0.33619, 0.39502, 0.40235]`

## 1. Discovered Governing Equations

| State Derivative | Discovered Symbolic Law | Weak $R^2$ | Complexity | MDL | Source |
|---|---|---:|---:|---:|---|
| `d(x)/dt` | `-10*x + 10*y` | 0.9980 | 6 | -262.04 | `vesp_sparse` |
| `d(y)/dt` | `-x*z + 28*x - y` | 0.9992 | 8 | -297.86 | `vesp_sparse` |
| `d(z)/dt` | `x*y - 8*z/3` | 0.9993 | 8 | -310.81 | `vesp_sparse` |

## 2. Direct Differential Causal Graph (`VIFC`)

- **Differential SHD:** `0` | **Precision:** `1.000` | **Recall:** `1.000` | **F1:** `1.000`
- **Inferred Edges:** `x -> d(x)/dt` (0.95), `x -> d(y)/dt` (0.96), `x -> d(z)/dt` (0.82), `y -> d(x)/dt` (0.95), `y -> d(y)/dt` (0.77), `y -> d(z)/dt` (0.90), `z -> d(y)/dt` (0.80), `z -> d(z)/dt` (0.87)

## 3. Model-Free Symmetries, Conservation Laws & Dynamical Regime (`W-LIND`)

- **Regime Classification:** `dissipative_chaotic` (Divergence $\nabla\cdot f =$ `-41/3`, $\hat{\lambda}_1 =$ `1.997`)
- **Closed-Loop Search Space Pruned:** `56.7%`
- **Discrete Symmetries:** `[{'description': 'Z2_reflection: (x, y, z) -> (-x, -y, z)', 'signs': [-1, -1, 1], 'error': 0.0017}]`
- **Conserved Quantities / First Integrals:** `[{'expr': '-0.7813*x**2 + x*y - 0.3687*y**2 + 0.0472*z**2', 'family': 'polynomial_first_integral', 'weak_residual': 0.0525, 'cancellation_ratio': 0.0568}]`
- **Continuous Lie Generators:** `[{'type': 'translation', 'components': ['-0.447500000000000', '-0.955800000000000', '1.00000000000000'], 'residual': 0.0266}, {'type': 'translation', 'components': ['-0.416000000000000', '-0.847900000000000', '-0.0605*y - 1.0'], 'residual': 0.0268}, {'type': 'translation', 'components': ['-1.00000000000000', '0.479200000000000', '0'], 'residual': 0.0213}]`

## 4. Formal Verification & Derived Scientific Insights (`ASDS-Verify`)

- **Formal Proof Pass Rate:** `20.0%`
- Constant phase-space volume contraction rate div(f) = -41/3 (-13.6667 < 0): trajectories collapse onto a lower-dimensional attractor.
- Formally verified discrete equivariance: Z2_reflection: (x, y, z) -> (-x, -y, z).
