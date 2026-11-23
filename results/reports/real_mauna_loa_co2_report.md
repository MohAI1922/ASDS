# ASDS-OMEGA v2.0 (VECSD) — Autonomous Scientific Discovery Report

- **System:** `mauna_loa_co2` (Real-World Climate Science (NOAA Mauna Loa 1960-2023))
- **Provenance Config Hash:** `2395ed6274a16db1` | **Seed:** `42` | **Runtime:** `5.773s`
- **Estimated Channel Noise (Wavelet-MAD):** `[0.0]`

## 1. Discovered Governing Equations

| State Derivative | Discovered Symbolic Law | Weak $R^2$ | Complexity | MDL | Source |
|---|---|---:|---:|---:|---|
| `d(C)/dt` | `C/5` | 0.8376 | 3 | -8.81 | `vesp_sparse` |

## 2. Direct Differential Causal Graph (`VIFC`)

- **Differential SHD:** `0` | **Precision:** `1.000` | **Recall:** `1.000` | **F1:** `1.000`
- **Inferred Edges:** `C -> d(C)/dt` (0.40)

## 3. Model-Free Symmetries, Conservation Laws & Dynamical Regime (`W-LIND`)

- **Regime Classification:** `non_conservative_general` (Divergence $\nabla\cdot f =$ `1/5`, $\hat{\lambda}_1 =$ `0.552`)
- **Closed-Loop Search Space Pruned:** `0.0%`
- **Discrete Symmetries:** `[]`
- **Conserved Quantities / First Integrals:** `[]`
- **Continuous Lie Generators:** `[{'type': 'scaling', 'components': ['-C'], 'residual': 0.0}]`

## 4. Formal Verification & Derived Scientific Insights (`ASDS-Verify`)

- **Formal Proof Pass Rate:** `100.0%`
