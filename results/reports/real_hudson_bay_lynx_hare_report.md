# ASDS-OMEGA v2.0 (VECSD) — Autonomous Scientific Discovery Report

- **System:** `hudson_bay_lynx_hare` (Real-World Ecology (Hudson Bay 1900-1920))
- **Provenance Config Hash:** `2395ed6274a16db1` | **Seed:** `42` | **Runtime:** `10.899s`
- **Estimated Channel Noise (Wavelet-MAD):** `[0.03494, 0.03085]`

## 1. Discovered Governing Equations

| State Derivative | Discovered Symbolic Law | Weak $R^2$ | Complexity | MDL | Source |
|---|---|---:|---:|---:|---|
| `d(hare)/dt` | `hare*(0.555469602749373 - 0.2572231818*lynx)` | 0.9447 | 6 | -5.07 | `srg_gp_varpro` |
| `d(lynx)/dt` | `0.03483*hare**2 + 0.26999*hare*lynx - hare/3 - lynx + 0.68169` | 0.9423 | 12 | 16.99 | `vesp_sparse` |

## 2. Direct Differential Causal Graph (`VIFC`)

- **Differential SHD:** `0` | **Precision:** `1.000` | **Recall:** `1.000` | **F1:** `1.000`
- **Inferred Edges:** `hare -> d(hare)/dt` (0.81), `hare -> d(lynx)/dt` (0.95), `lynx -> d(hare)/dt` (0.95), `lynx -> d(lynx)/dt` (0.85)

## 3. Model-Free Symmetries, Conservation Laws & Dynamical Regime (`W-LIND`)

- **Regime Classification:** `dissipative_chaotic` (Divergence $\nabla\cdot f =$ `0.26999*hare - 0.2572231818*lynx - 0.444530397250627`, $\hat{\lambda}_1 =$ `0.351`)
- **Closed-Loop Search Space Pruned:** `0.0%`
- **Discrete Symmetries:** `[]`
- **Conserved Quantities / First Integrals:** `[{'expr': '-0.2949*hare - 0.4337*lynx + log(hare) + log(lynx)', 'family': 'logarithmic_first_integral', 'weak_residual': 0.0435, 'cancellation_ratio': 0.0565}]`
- **Continuous Lie Generators:** `[{'type': 'scaling', 'components': ['0.2221*hare - 1.0', '0.2219*lynx - 0.498'], 'residual': 0.0211}, {'type': 'affine_lie', 'components': ['0.2644*lynx - 0.5354', '-0.1548*hare - 0.166*lynx + 1.0'], 'residual': 0.0386}]`

## 4. Formal Verification & Derived Scientific Insights (`ASDS-Verify`)

- **Formal Proof Pass Rate:** `0.0%`
- State-dependent phase-space divergence: div(f) = 0.26999*hare - 0.2572231818*lynx - 0.444530397250627.
