# ASDS-OMEGA v2.0 — Variational Equivariant Causal-Symbolic Discovery (VECSD)

**ASDS-OMEGA v2.0** is an open-source, publication-grade autonomous scientific discovery framework that infers interpretable governing differential equations (ODEs & 1+1D PDEs), direct differential causal graphs, discrete/continuous symmetries, and exact conservation laws directly from noisy multi-trajectory data—complete with Popperian active experimental falsification, online concept-drift structural regeneration, and formal **SymPy + Lean 4** verification certificates.

---

## Unifying Theoretical Architecture (VECSD)

Rather than chaining disconnected heuristics, all six layers operate inside a shared **variational operator-theoretic substrate**:

| Layer | Module | Novel Algorithmic Contribution |
|---|---|---|
| **1. Variational Substrate** | `asds_omega/preprocessing/ms_avp.py` | **Multi-Scale Adaptive Variational Projection (MS-AVP)** with Wavelet-MAD noise estimation, local curvature-optimal test-function widths $W_k^*$, **analytical Hermite/Wick Cumulant Errors-in-Variables (EIV) debiasing** ($\mathbb{E}[H_n(\tilde{x};\hat{\sigma}^2)] = x^n$), 2D spacetime PDE weak projection, and **Buckingham $\Pi$** dimensional analysis. |
| **2. Dynamical Causality** | `asds_omega/causal/vifc.py` | **Variational Interventional Flow Causality (VIFC)** discovering the direct differential causal graph $x_j(t) \to \dot{x}_i(t)$ (handling cyclic ODE feedback loops cleanly) via **WF-cHSIC + temporal block-permuted knockoffs + Manifold-Aware Faithfulness Guard**, **Multivariate Partial CCM**, and **Universal RBF Dyn-ICP**. |
| **3. Symmetry & Invariants** | `asds_omega/symmetry/reynolds.py` | **Model-Free Weak-Form Lie-Noether & Foliation Discovery (W-LIND)** solving $\langle \dot{\phi}_k, \psi(x)\rangle c = 0$ (zero derivatives, zero prior $f(x)$!) with Univariate Exclusion Theorem, discrete group equivariance, continuous Lie bracket SVD + closure, Liouville divergence/Lyapunov classification, and **exact Reynolds Equivariant Projection $\mathcal{P}_{\mathcal{G}}$**. |
| **4. Symbolic Evolution** | `asds_omega/symbolic/vesp.py` & `srg_gp.py` | **Equivariant Variational Grammar-Guided Hybrid Evolution (EVG-HE)** combining Reynolds-pruned, causally-masked tiered **V-ESP**, **Total-Least-Squares Implicit Weak-Form Rational Discovery**, coupled **KKT conservation enforcement**, **Semantic Residual-Guided Weak-Form NSGA-II GP + Variable Projection (VarPro)**, **$K_{\text{eff}}$-corrected MDL continued-fraction rationalizer**, and a **Contextual Thompson-Sampling Meta-Controller**. |
| **5. Autonomous Agency** | `asds_omega/scientist/` | **Popperian Active Falsification Engine (PAFE)** selecting maximally falsifying initial conditions via epistemic trajectory disagreement, plus **Dual-Channel CUSUM Surgical Structural Regeneration (SLR)** for online regime drift. |
| **6. Formal Certification** | `asds_omega/verification/` | **SymPy Analytical Prover** ($\nabla H \cdot \hat{f} \equiv 0$, $g\cdot\hat{f}(x) = \hat{f}(g\cdot x)$, $[V,\hat{f}] \equiv 0$, exact $\nabla\cdot\hat{f}$, equilibria, $R_0$, cycle periods) + **Machine-Checkable Lean 4 (`.lean`) Certificate Exporter** + automated LaTeX/Markdown/JSON discovery monographs. |

---

## Installation & Quick Start

```bash
# Install into isolated virtual environment
uv venv .venv --python 3.13
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"

# Run unit, mathematical-property, integration, and API tests
.venv/Scripts/pytest tests/ -v

# Run autonomous discovery on any synthetic or real-world dataset
.venv/Scripts/asds-omega discover --dataset lorenz --noise 0.05 --seed 42
.venv/Scripts/asds-omega discover --dataset hudson_bay_lynx_hare

# Execute full reproducible benchmark + 6-factor ablation suite + 300-DPI figures
.venv/Scripts/asds-omega benchmark --seed 42 --out results

# Launch as a standalone native desktop GUI window (Edge/Chrome --app or pywebview)
.venv/Scripts/asds-omega gui

# Or launch as a browser-accessible local/HPC Scientific Discovery Workbench
.venv/Scripts/asds-omega serve --host 127.0.0.1 --port 8000
```

---

## Verified 5-Seed Benchmark Summary (@ 5% Gaussian Noise, $N = 40$ Trials)

All metrics below are reproduced deterministically via `asds-omega benchmark --seed 42 --out results` (`results/benchmark_ledger.json`) across 5 independent random seeds with Student-$t$ 95% confidence intervals and one-sided paired Wilcoxon signed-rank tests:

| System | ASDS-OMEGA v2.0 | PySINDy v2.1 (Official) | Weak-SINDy | Std. SINDy | Pure GP | $3\sigma$ OOD RMSE (ASDS / MLP) |
|---|---:|---:|---:|---:|---:|---:|
| **Lorenz-63** | **0.997 ± 0.004** | 0.797 ± 0.074 | 0.976 ± 0.055 | 0.679 | 0.315 ± 0.160 | **0.0030** / 0.586 |
| **Lotka–Volterra** | 0.995 ± 0.005 | 0.944 ± 0.149 | 0.996 ± 0.003 | **0.997 ± 0.002** | 0.362 ± 0.228 | **0.0115** / 0.073 |
| **SIR Epidemic** | **1.000** | 0.041 ± 0.016 | 0.246 ± 0.107 | 0.540 ± 0.146 | 0.659 ± 0.074 | **0.0000** / 1.156 |
| **Nonlinear Pendulum** | **1.000** | 0.689 ± 0.414 | 0.999 | 0.999 | 0.999 | **0.0000** / 0.489 |
| **Kepler 2-Body** | **1.000** | 0.498 | 0.898 ± 0.170 | 0.848 ± 0.164 | 0.498 | **0.0000** / 0.932 |
| **Michaelis–Menten** | **0.985 ± 0.003** | 0.851 ± 0.005 | 0.864 ± 0.004 | 0.849 ± 0.004 | 0.694 ± 0.098 | **0.0176** / 0.190 |
| **Duffing Oscillator** | **0.998 ± 0.003** | 0.995 ± 0.003 | **0.998 ± 0.003** | 0.926 ± 0.183 | 0.572 ± 0.082 | **0.0006** / 0.508 |
| **Van der Pol** | **1.000** | 0.998 | 0.999 | 0.993 ± 0.001 | 0.663 | **0.0000** / 0.871 |
| **Overall Mean ($N=40$)** | **0.997 ± 0.002** | 0.727 ± 0.105 | 0.872 ± 0.080 | 0.854 ± 0.056 | 0.595 ± 0.070 | — |
| **Wilcoxon $p$-value** | — | **$< 10^{-5}$** | **$< 10^{-5}$** | **$< 10^{-5}$** | **$< 10^{-5}$** | — |

- **High-Noise Cubic EIV Debiasing (Van der Pol $-x^2 y$)**: At 15% and 20% noise, Hermite cumulant debiasing reduces coefficient error to **0.004 ± 0.002** and **0.006 ± 0.004** (vs. 0.006 / 0.011 un-debiased, 0.010 / 0.016 Weak-SINDy, and 0.017 / 0.072 PySINDy v2.1).
- **1+1D PDEs @ 5% Noise (5 seeds)**: Viscous Burgers ($u_t = -1.003\,u u_x + 0.147\,u_{xx}$, Weak $R^2 = 0.986 \pm 0.002$) and Advection–Diffusion ($u_t = -0.799\,u_x + 0.119\,u_{xx}$, Weak $R^2 = 0.993 \pm 0.001$).
- **6-Factor Ablation @ 10% Noise (5 systems × 5 seeds)**: Full VECSD achieves **0.936 ± 0.069** Symbolic Match / **0.042 ± 0.056** Coeff. Error, outperforming every ablated variant (e.g., removing Weak-Form drops to `0.755` / `0.222`; removing W-LIND drops to `0.862` / `0.104`; removing VIFC drops to `0.921` / `0.055`).

---

## Bundled Benchmarks, Manuscript & Formal Verification Artifacts

- **8 Synthetic Ground-Truth ODEs**: Lorenz-63, Lotka–Volterra, SIR (& SIR$\to$SIRS Regime Drift), Nonlinear Pendulum, Kepler 2-Body, Michaelis–Menten, Duffing Double-Well Oscillator, and Van der Pol Limit-Cycle Oscillator.
- **2 Spatiotemporal 1+1D PDEs**: Viscous Burgers PDE ($u_t = -u u_x + \nu u_{xx}$) and Advection–Diffusion PDE ($u_t = -c u_x + D u_{xx}$).
- **3 Real-World Experimental Datasets**: Hudson Bay Company Lynx–Hare fur records (1900–1920), 1978 English Boarding School Influenza outbreak (BMJ, $N=763$), and NOAA Mauna Loa Atmospheric $\text{CO}_2$ (1960–2023).
- **LaTeX Manuscript & Auto-Generated Tables**: `paper/main.tex` (compiled to `paper/main.pdf`), `paper/references.bib`, and `paper/generate_tables.py`.
- **Lean 4 Certificates**: `sorry`-free formal theorem certificates exported to `results/lean4/*_certificate.lean`.
- **Reproducibility Packaging**: `CITATION.cff`, `LICENSE` (MIT), `Dockerfile`, and `.github/workflows/ci.yml`.
