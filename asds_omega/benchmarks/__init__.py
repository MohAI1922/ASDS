"""Publication-grade benchmarking harness, real baselines, ablations, and high-DPI plotting."""

from asds_omega.benchmarks.baselines import (
    BaselineResult,
    run_neural_mlp_baseline,
    run_pure_gp_baseline,
    run_pysindy_official_baseline,
    run_standard_sindy_baseline,
    run_weak_sindy_baseline,
)

__all__ = [
    "BaselineResult",
    "run_standard_sindy_baseline",
    "run_weak_sindy_baseline",
    "run_pysindy_official_baseline",
    "run_pure_gp_baseline",
    "run_neural_mlp_baseline",
]
