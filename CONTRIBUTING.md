# Contributing to ASDS-OMEGA v2.0

We welcome contributions from researchers, physicists, applied mathematicians, biologists, and software engineers!

## Reporting Bugs & Seeking Support

- **Bug reports & unexpected symbolic failures:** Open a GitHub Issue including your dataset shape, noise settings, random seed, `PipelineConfig`, Python version, and minimal reproducible snippet.
- **Scientific questions & feature requests:** Open a GitHub Discussion or Issue labeled `enhancement` or `question`.

## Development Setup

```bash
git clone https://github.com/MohAI1922/ASDS.git
cd ASDS
uv venv .venv --python 3.13
uv pip install --python .venv/Scripts/python.exe -e ".[dev,benchmarks]"
```

On Linux/macOS, replace `.venv/Scripts/python.exe` with `.venv/bin/python`.

## Running the Automated Test Suite

All pull requests must pass the full unit, mathematical-property, integration, and API test suite:

```bash
pytest tests/ -v
```

To regenerate the full multi-seed publication benchmarks, 300-DPI figures, and LaTeX tables:

```bash
asds-omega benchmark --seed 42 --n-seeds 5 --out results
python paper/generate_tables.py
```

## Pull Request Guidelines

1. Keep every algorithmic addition deterministic (accept an explicit `seed` parameter and use `np.random.default_rng(seed)`).
2. Add at least one mathematical-property or regression test in `tests/test_asds_omega.py` (or API test in `tests/test_api.py`).
3. Ensure no new dependencies are added to core `pyproject.toml` without justification; place optional dependencies under `[project.optional-dependencies]`.
4. Follow the [Code of Conduct](CODE_OF_CONDUCT.md) in all interactions.
