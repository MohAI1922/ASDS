# ASDS-OMEGA v2.0 (VECSD) — Autonomous Scientific Discovery Report

- **System:** `sir` (Epidemiology (Compartmental Disease Spread))
- **Provenance Config Hash:** `0a2414d2f55e0c5f` | **Seed:** `42` | **Runtime:** `14.810s`
- **Estimated Channel Noise (Wavelet-MAD):** `[0.01108, 0.00741, 0.01563]`

## 1. Discovered Governing Equations

| State Derivative | Discovered Symbolic Law | Weak $R^2$ | Complexity | MDL | Source |
|---|---|---:|---:|---:|---|
| `d(S)/dt` | `-I*S/2` | 0.9579 | 6 | -98.75 | `vesp_manifold_reduced` |
| `d(I)/dt` | `I*(5*S - 1)/10` | 0.9575 | 7 | -91.37 | `vesp_manifold_reduced` |
| `d(R)/dt` | `I/10` | 0.7841 | 3 | -44.15 | `vesp_sparse` |

## 2. Direct Differential Causal Graph (`VIFC`)

- **Differential SHD:** `3` | **Precision:** `0.667` | **Recall:** `0.800` | **F1:** `0.727`
- **Inferred Edges:** `S -> d(S)/dt` (0.88), `S -> d(I)/dt` (0.93), `I -> d(S)/dt` (0.50), `I -> d(R)/dt` (0.85), `R -> d(S)/dt` (0.84), `R -> d(I)/dt` (0.43)

## 3. Model-Free Symmetries, Conservation Laws & Dynamical Regime (`W-LIND`)

- **Regime Classification:** `dissipative_relaxational` (Divergence $\nabla\cdot f =$ `-I/2 + S/2 - 1/10`, $\hat{\lambda}_1 =$ `0.094`)
- **Closed-Loop Search Space Pruned:** `43.3%`
- **Discrete Symmetries:** `[]`
- **Conserved Quantities / First Integrals:** `[{'expr': 'I + R + S', 'family': 'linear_conservation', 'weak_residual': 0.1306, 'cancellation_ratio': 0.0337}]`
- **Continuous Lie Generators:** `[{'type': 'translation', 'components': ['0', '0', '1.00000000000000'], 'residual': 0.0}, {'type': 'affine_lie', 'components': ['0', '0', 'I + R + S'], 'residual': 0.0}, {'type': 'affine_lie', 'components': ['-0.7473*I - 0.7533*R - 0.76*S + 0.7525', 'I + R + S - 0.9979', '0'], 'residual': 0.0012}]`

## 4. Formal Verification & Derived Scientific Insights (`ASDS-Verify`)

- **Formal Proof Pass Rate:** `75.0%`
- State-dependent phase-space divergence: div(f) = -I/2 + S/2 - 1/10.
- Formally verified exact conservation law: d/dt (I + R + S) = 0.
- Derived epidemiological basic reproduction number R0 = beta / gamma = 5.000 (herd-immunity threshold = 80.0%).
