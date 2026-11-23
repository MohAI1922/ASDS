# ASDS-OMEGA v2.0 (VECSD) — Autonomous Scientific Discovery Report

- **System:** `pendulum` (Classical Mechanics (Nonlinear Oscillator))
- **Provenance Config Hash:** `0a2414d2f55e0c5f` | **Seed:** `42` | **Runtime:** `0.907s`
- **Estimated Channel Noise (Wavelet-MAD):** `[0.05502, 0.06489]`

## 1. Discovered Governing Equations

| State Derivative | Discovered Symbolic Law | Weak $R^2$ | Complexity | MDL | Source |
|---|---|---:|---:|---:|---|
| `d(theta)/dt` | `omega` | 0.9972 | 2 | -202.93 | `vesp_sparse` |
| `d(omega)/dt` | `-2*sin(theta)` | 0.9972 | 5 | -198.29 | `vesp_sparse` |

## 2. Direct Differential Causal Graph (`VIFC`)

- **Differential SHD:** `0` | **Precision:** `1.000` | **Recall:** `1.000` | **F1:** `1.000`
- **Inferred Edges:** `theta -> d(omega)/dt` (0.95), `omega -> d(theta)/dt` (0.95)

## 3. Model-Free Symmetries, Conservation Laws & Dynamical Regime (`W-LIND`)

- **Regime Classification:** `conservative_volume_preserving` (Divergence $\nabla\cdot f =$ `0`, $\hat{\lambda}_1 =$ `1.262`)
- **Closed-Loop Search Space Pruned:** `79.2%`
- **Discrete Symmetries:** `[{'description': 'Z2_reflection: (theta, omega) -> (-theta, -omega)', 'signs': [-1, -1], 'error': 0.0041}]`
- **Conserved Quantities / First Integrals:** `[]`
- **Continuous Lie Generators:** `[{'type': 'SO(2)_rotation', 'components': ['-omega', 'theta'], 'residual': 0.1475}]`

## 4. Formal Verification & Derived Scientific Insights (`ASDS-Verify`)

- **Formal Proof Pass Rate:** `50.0%`
- Phase-space volume is strictly conserved (Liouville divergence div(f) = 0; Hamiltonian/symplectic flow).
- Formally verified discrete equivariance: Z2_reflection: (theta, omega) -> (-theta, -omega).
