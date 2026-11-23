"""Command-line interface for ASDS-OMEGA v2.0 (VECSD)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from asds_omega.benchmarks.plotting import generate_all_publication_figures
from asds_omega.benchmarks.runner import run_full_benchmark_suite
from asds_omega.config import PipelineConfig
from asds_omega.data import generate_synthetic_dataset, inject_noise, load_real_dataset
from asds_omega.scientist import ASDSOmegaPipeline
from asds_omega.verification import export_discovery_json, export_latex_report, export_markdown_report


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="asds-omega",
        description="ASDS-OMEGA v2.0 (VECSD): Variational Equivariant Causal-Symbolic Discovery",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("discover", help="Run autonomous scientific discovery on a dataset")
    p_run.add_argument("--dataset", required=True, help="lorenz|lotka_volterra|sir|pendulum|kepler|michaelis_menten|hudson_bay_lynx_hare|boarding_school_influenza|mauna_loa_co2")
    p_run.add_argument("--noise", type=float, default=0.05, help="Relative Gaussian noise level (synthetic only)")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--active", action="store_true", help="Enable Popperian active falsification loop")
    p_run.add_argument("--out", type=str, default="results/single_run")

    p_bench = sub.add_parser("benchmark", help="Run full multi-seed publication benchmark + ablation suite and generate figures")
    p_bench.add_argument("--seed", type=int, default=42)
    p_bench.add_argument("--n-seeds", type=int, default=5)
    p_bench.add_argument("--out", type=str, default="results")

    p_serve = sub.add_parser("serve", help="Launch browser-based FastAPI Scientific Discovery Workbench")
    p_serve.add_argument("--host", type=str, default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    p_gui = sub.add_parser("gui", help="Launch ASDS-OMEGA v2.0 as a standalone native desktop GUI window")
    p_gui.add_argument("--port", type=int, default=None, help="Optional fixed port (default: auto-select free port)")

    args = parser.parse_args()

    if args.command == "discover":
        real_names = {"hudson_bay_lynx_hare", "boarding_school_influenza", "mauna_loa_co2", "lynx_hare", "influenza", "mauna_loa", "co2"}
        if args.dataset.lower() in real_names:
            ds = load_real_dataset(args.dataset)
        else:
            ds = inject_noise(generate_synthetic_dataset(args.dataset, seed=args.seed), args.noise, seed=args.seed)
        cfg = PipelineConfig(seed=args.seed)
        res = ASDSOmegaPipeline(cfg).run(ds, run_active_falsification=args.active)
        summ = res.to_summary_dict()
        out_dir = Path(args.out)
        export_discovery_json(summ, out_dir / f"{ds.name}.json")
        export_markdown_report(summ, out_dir / f"{ds.name}_report.md")
        export_latex_report(summ, out_dir / f"{ds.name}_report.tex")
        if res.lean4_certificate:
            (out_dir / f"{ds.name}_certificate.lean").write_text(res.lean4_certificate, encoding="utf-8")
        print(json.dumps(summ, indent=2))

    elif args.command == "benchmark":
        run_full_benchmark_suite(output_dir=args.out, seed=args.seed, n_seeds=args.n_seeds)
        figs = generate_all_publication_figures(
            ledger_path=Path(args.out) / "benchmark_ledger.json",
            figures_dir=Path(args.out) / "figures",
        )
        print(f"Generated {len(figs)} publication figures in {Path(args.out) / 'figures'}")

    elif args.command == "serve":
        import uvicorn
        uvicorn.run("app.server:app", host=args.host, port=args.port, reload=False)

    elif args.command == "gui":
        from asds_omega.gui import launch_desktop_gui
        launch_desktop_gui(port=args.port)


if __name__ == "__main__":
    main()
