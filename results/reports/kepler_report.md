# ASDS-OMEGA v2.0 (VECSD) — Autonomous Scientific Discovery Report

- **System:** `kepler` (Astrophysics (Central-Force Orbital Mechanics))
- **Provenance Config Hash:** `0a2414d2f55e0c5f` | **Seed:** `42` | **Runtime:** `7.734s`
- **Estimated Channel Noise (Wavelet-MAD):** `[0.04574, 0.04574, 0.02989, 0.03137]`

## 1. Discovered Governing Equations

| State Derivative | Discovered Symbolic Law | Weak $R^2$ | Complexity | MDL | Source |
|---|---|---:|---:|---:|---|
| `d(x)/dt` | `vx` | 0.9865 | 2 | -106.93 | `vesp_sparse` |
| `d(y)/dt` | `vy` | 0.9887 | 2 | -111.72 | `vesp_sparse` |
| `d(vx)/dt` | `-x/(x**2 + y**2)**(3/2)` | 0.9881 | 10 | -95.11 | `vesp_sparse` |
| `d(vy)/dt` | `-y/(x**2 + y**2)**(3/2)` | 0.9913 | 10 | -103.49 | `vesp_sparse` |

## 2. Direct Differential Causal Graph (`VIFC`)

- **Differential SHD:** `2` | **Precision:** `0.750` | **Recall:** `1.000` | **F1:** `0.857`
- **Inferred Edges:** `x -> d(vx)/dt` (0.47), `x -> d(vy)/dt` (0.81), `y -> d(vx)/dt` (0.47), `y -> d(vy)/dt` (0.52), `vx -> d(x)/dt` (0.95), `vx -> d(vy)/dt` (0.95), `vy -> d(y)/dt` (0.95), `vy -> d(vx)/dt` (0.83)

## 3. Model-Free Symmetries, Conservation Laws & Dynamical Regime (`W-LIND`)

- **Regime Classification:** `conservative_volume_preserving` (Divergence $\nabla\cdot f =$ `0`, $\hat{\lambda}_1 =$ `0.372`)
- **Closed-Loop Search Space Pruned:** `91.2%`
- **Discrete Symmetries:** `[{'description': 'Z2_reflection: (x, y, vx, vy) -> (x, -y, vx, -vy)', 'signs': [1, -1, 1, -1], 'error': 0.0237}, {'description': 'Z2_reflection: (x, y, vx, vy) -> (-x, -y, -vx, -vy)', 'signs': [-1, -1, -1, -1], 'error': 0.0293}, {'description': 'Z2_reflection: (x, y, vx, vy) -> (-x, y, -vx, vy)', 'signs': [-1, 1, -1, 1], 'error': 0.0298}]`
- **Conserved Quantities / First Integrals:** `[{'expr': 'vx*vy - 0.6644*vx*x + 0.7772*vy*y - 0.4453*x*y', 'family': 'polynomial_first_integral', 'weak_residual': 0.0247, 'cancellation_ratio': 0.0296}, {'expr': '0.7255*vx**2 + vx*y + 0.298*y**2', 'family': 'polynomial_first_integral', 'weak_residual': 0.0389, 'cancellation_ratio': 0.0409}, {'expr': '-vx*y + vy*x', 'family': 'central_force_energy', 'weak_residual': 0.0806, 'cancellation_ratio': 0.074}]`
- **Continuous Lie Generators:** `[{'type': 'SO(2)_rotation', 'components': ['y', '-x', 'vy', '-vx'], 'residual': 0.0}, {'type': 'affine_lie', 'components': ['0.9214*vx - 0.0605*vy + 0.6607*y + 0.0496', '-0.063*vx + vy - 0.7162*x - 0.1197', '0.1706*vx - 0.7272*vy + 0.455*x + 0.1288*y + 0.1959', '0.6497*vx - 0.2096*vy + 0.1457*x + 0.4073*y + 0.1633'], 'residual': 0.0596}]`

## 4. Formal Verification & Derived Scientific Insights (`ASDS-Verify`)

- **Formal Proof Pass Rate:** `62.5%`
- Phase-space volume is strictly conserved (Liouville divergence div(f) = 0; Hamiltonian/symplectic flow).
- Formally verified exact conservation law: d/dt (-vx*y + vy*x) = 0.
- Formally verified discrete equivariance: Z2_reflection: (x, y, vx, vy) -> (x, -y, vx, -vy).
- Formally verified discrete equivariance: Z2_reflection: (x, y, vx, vy) -> (-x, -y, -vx, -vy).
- Formally verified discrete equivariance: Z2_reflection: (x, y, vx, vy) -> (-x, y, -vx, vy).
