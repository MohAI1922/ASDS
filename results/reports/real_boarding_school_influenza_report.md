# ASDS-OMEGA v2.0 (VECSD) — Autonomous Scientific Discovery Report

- **System:** `boarding_school_influenza` (Real-World Epidemiology (1978 BMJ Influenza Outbreak))
- **Provenance Config Hash:** `2395ed6274a16db1` | **Seed:** `42` | **Runtime:** `12.261s`
- **Estimated Channel Noise (Wavelet-MAD):** `[0.0, 0.00055, 0.0]`

## 1. Discovered Governing Equations

| State Derivative | Discovered Symbolic Law | Weak $R^2$ | Complexity | MDL | Source |
|---|---|---:|---:|---:|---|
| `d(S)/dt` | `-29*I*S/17` | 0.9349 | 7 | -11.32 | `vesp_sparse` |
| `d(I)/dt` | `-9*I**3 + 34*I/15 + 23*R**2*S/5 + 16*R**2/7 - 59*R/24` | 0.9948 | 21 | 5.05 | `vesp_tier3_escape` |
| `d(R)/dt` | `4*I/9` | 0.9984 | 4 | -48.58 | `vesp_sparse` |

## 2. Direct Differential Causal Graph (`VIFC`)

- **Differential SHD:** `2` | **Precision:** `0.800` | **Recall:** `0.800` | **F1:** `0.800`
- **Inferred Edges:** `S -> d(S)/dt` (0.80), `S -> d(I)/dt` (0.91), `I -> d(S)/dt` (0.56), `I -> d(R)/dt` (0.95), `R -> d(S)/dt` (0.56)

## 3. Model-Free Symmetries, Conservation Laws & Dynamical Regime (`W-LIND`)

- **Regime Classification:** `non_conservative_general` (Divergence $\nabla\cdot f =$ `-27*I**2 - 29*I/17 + 34/15`, $\hat{\lambda}_1 =$ `0.000`)
- **Closed-Loop Search Space Pruned:** `53.3%`
- **Discrete Symmetries:** `[]`
- **Conserved Quantities / First Integrals:** `[{'expr': 'I + R + S', 'family': 'linear_conservation', 'weak_residual': 0.0, 'cancellation_ratio': 0.0}, {'expr': '-0.5608*I*R - 0.5579*R*S + 0.5468*sin(R) + cos(R)', 'family': 'hamiltonian_trigonometric', 'weak_residual': 0.0016, 'cancellation_ratio': 0.0006}]`
- **Continuous Lie Generators:** `[{'type': 'affine_lie', 'components': ['0.1682*S', 'I + 0.8493*R + 0.876*S - 0.8816', '0.1741*I + 0.2186*R + 0.1932*S - 0.2005'], 'residual': 0.0048}, {'type': 'affine_lie', 'components': ['0.855*I + 0.882*R + S - 0.8837', '0.0689*I - 0.0449', '-0.4281*I - 0.4689*R - 0.4614*S + 0.455'], 'residual': 0.009}, {'type': 'affine_lie', 'components': ['0.466*I + 0.5117*R + 0.5554*S - 0.5022', '-0.2163*I - 0.265*R - 0.249*S + 0.2474', 'I + R + S - 0.9982'], 'residual': 0.0036}]`

## 4. Formal Verification & Derived Scientific Insights (`ASDS-Verify`)

- **Formal Proof Pass Rate:** `0.0%`
- State-dependent phase-space divergence: div(f) = -27*I**2 - 29*I/17 + 34/15.
- Derived epidemiological basic reproduction number R0 = beta / gamma = 3.838 (herd-immunity threshold = 73.9%).
