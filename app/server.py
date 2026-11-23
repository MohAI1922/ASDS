"""FastAPI Scientific Research Workbench Server for ASDS-OMEGA v2.0 (VECSD).

Serves both the local browser workspace (`asds-omega serve`) and the native
standalone desktop GUI window (`asds-omega gui`), exposing ODE/PDE discovery,
CSV ingestion, counterfactual do-interventions, online concept-drift monitoring,
reproducible script/report export, and verified benchmark ledger inspection.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from asds_omega.causal.counterfactual import CounterfactualSimulator
from asds_omega.config import EVGHEConfig, MSAVPConfig, PipelineConfig, VIFCConfig, WLINDConfig
from asds_omega.data import (
    DynamicalDataset,
    gen_sir_regime_drift,
    generate_synthetic_dataset,
    inject_noise,
    load_real_dataset,
)
from asds_omega.scientist import ASDSOmegaPipeline, SurgicalDriftRegenerator
from asds_omega.symbolic import discover_pde_vesp
from asds_omega.verification import export_latex_report, export_markdown_report

app = FastAPI(title="ASDS-OMEGA v2.0 (VECSD) Scientific Discovery Workbench", version="2.0.0")
STATIC_DIR = Path(__file__).parent / "static"
REPO_ROOT = Path(__file__).resolve().parent.parent

SYNTHETIC_LIST = [
    {"id": "lorenz", "label": "Lorenz-63 (Chaotic Physics, Z2)", "kind": "synthetic", "vars": 3},
    {"id": "lotka_volterra", "label": "Lotka-Volterra (Ecology, Log Integral)", "kind": "synthetic", "vars": 2},
    {"id": "sir", "label": "SIR Epidemic (Mass Conservation S+I+R=1)", "kind": "synthetic", "vars": 3},
    {"id": "pendulum", "label": "Nonlinear Pendulum (Hamiltonian, Z2)", "kind": "synthetic", "vars": 2},
    {"id": "kepler", "label": "Kepler 2-Body (SO(2) Rotation, L & E)", "kind": "synthetic", "vars": 4},
    {"id": "michaelis_menten", "label": "Michaelis-Menten (Rational Kinetics)", "kind": "synthetic", "vars": 2},
    {"id": "duffing", "label": "Duffing Double-Well (Cubic x^3 EIV)", "kind": "synthetic", "vars": 2},
    {"id": "van_der_pol", "label": "Van der Pol Limit Cycle (Cubic x^2*y EIV)", "kind": "synthetic", "vars": 2},
]
REAL_LIST = [
    {"id": "hudson_bay_lynx_hare", "label": "Hudson Bay Lynx-Hare (1900-1920)", "kind": "real", "vars": 2},
    {"id": "boarding_school_influenza", "label": "1978 English Boarding School Influenza (BMJ)", "kind": "real", "vars": 3},
    {"id": "mauna_loa_co2", "label": "NOAA Mauna Loa Atmospheric CO2 (1960-2023)", "kind": "real", "vars": 1},
]
PDE_LIST = [
    {"id": "burgers_pde", "label": "Viscous Burgers PDE (u_t = -u u_x + 0.15 u_xx)", "kind": "pde"},
    {"id": "advection_diffusion_pde", "label": "Advection-Diffusion PDE (u_t = -0.8 u_x + 0.12 u_xx)", "kind": "pde"},
]


class DiscoverRequest(BaseModel):
    dataset: str = "lorenz"
    noise_level: float = 0.05
    noise_type: str = "gaussian"
    seed: int = 42
    use_weak_form: bool = True
    cumulant_debiasing: bool = True
    use_vifc: bool = True
    use_wlind: bool = True
    use_rationalizer: bool = True
    run_active_falsification: bool = False


class PDEDiscoverRequest(BaseModel):
    dataset: str = "burgers_pde"
    noise_level: float = 0.05
    cumulant_debiasing: bool = True
    seed: int = 42


class CounterfactualRequest(BaseModel):
    var_names: list[str]
    equations: list[str]
    x0: list[float]
    t_end: float = 15.0
    n_points: int = 250
    scale_equation: dict[str, float] | None = None
    clamped_vars: dict[str, float] | None = None


class ExportRequest(BaseModel):
    summary: dict[str, Any]
    format: str = "markdown"  # markdown | latex | scipy | lean4


def _expr_to_latex(expr_str: str, var_names: list[str]) -> str:
    local_map = {v: sp.Symbol(v) for v in var_names}
    try:
        expr = sp.sympify(expr_str, locals=local_map)
        return sp.latex(expr)
    except Exception:
        return expr_str


def _enrich_latex(payload: dict[str, Any]) -> None:
    vnames = payload.get("var_names", [])
    for eq in payload.get("discovered_equations", []):
        eq["latex"] = _expr_to_latex(eq["expr_str"], vnames)
    for inv in payload.get("symmetry", {}).get("invariants", []):
        inv["latex"] = _expr_to_latex(inv["expr"], vnames)
    ver = payload.get("verification", {})
    if ver.get("symbolic_divergence") is not None:
        ver["symbolic_divergence_latex"] = _expr_to_latex(str(ver["symbolic_divergence"]), vnames)


def _build_trajectory_preview(
    ds: DynamicalDataset,
    eqs: list[sp.Expr],
    invariants: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    t0 = ds.t_list[0]
    X_obs = ds.trajectories[0]
    X_clean = ds.clean_trajectories[0]
    step = max(1, len(t0) // 260)
    t_sub = t0[::step]
    X_obs_sub = X_obs[::step]
    X_clean_sub = X_clean[::step]

    sim = CounterfactualSimulator(ds.var_names, eqs)
    X_disc = sim.simulate(X_clean_sub[0], t_sub)
    X_disc_clean = np.where(np.isfinite(X_disc), X_disc, np.nan)
    resid = np.where(np.isfinite(X_disc_clean), X_disc_clean - X_clean_sub, np.nan)

    # Evaluate discovered conservation laws H(t) - H(0) along observed & discovered trajectories
    inv_traces: list[dict[str, Any]] = []
    if invariants:
        local_map = {v: sp.Symbol(v) for v in ds.var_names}
        syms = [local_map[v] for v in ds.var_names]
        for inv in invariants[:2]:
            try:
                h_expr = sp.sympify(inv["expr"], locals=local_map)
                h_fn = sp.lambdify(syms, h_expr, modules="numpy")
                h_obs = np.array([float(h_fn(*row)) for row in X_clean_sub], dtype=float)
                h_disc = np.array(
                    [float(h_fn(*row)) if np.all(np.isfinite(row)) else np.nan for row in X_disc_clean],
                    dtype=float,
                )
                if np.isfinite(h_obs[0]):
                    h0 = h_obs[0]
                    scale = max(abs(h0), float(np.nanstd(h_obs)) + 1e-6, 1.0)
                    inv_traces.append({
                        "expr": inv["expr"],
                        "latex": _expr_to_latex(inv["expr"], ds.var_names),
                        "obs_drift": [None if not np.isfinite(v) else round(float((v - h0) / scale), 5) for v in h_obs],
                        "disc_drift": [None if not np.isfinite(v) else round(float((v - h0) / scale), 5) for v in h_disc],
                    })
            except Exception:
                continue

    return {
        "t": [round(float(v), 4) for v in t_sub],
        "observed": np.round(X_obs_sub, 4).tolist(),
        "clean": np.round(X_clean_sub, 4).tolist(),
        "discovered": [
            [None if not np.isfinite(v) else round(float(v), 4) for v in row]
            for row in X_disc_clean
        ],
        "residuals": [
            [None if not np.isfinite(v) else round(float(v), 4) for v in row]
            for row in resid
        ],
        "invariant_traces": inv_traces,
        "x0": [round(float(v), 4) for v in X_clean_sub[0]],
        "t_end": float(t_sub[-1]),
    }


def _generate_scipy_script(summary: dict[str, Any]) -> str:
    vnames = summary.get("var_names", [])
    eqs = summary.get("discovered_equations", [])
    tr = summary.get("trajectories", {})
    x0 = tr.get("x0", [1.0] * len(vnames))
    t_end = tr.get("t_end", 15.0)
    unpack = ", ".join(vnames)
    body_lines = [f"    d_{eq['target_var']} = {eq['expr_str']}" for eq in eqs]
    ret_list = ", ".join(f"d_{eq['target_var']}" for eq in eqs)
    return "\n".join([
        f'"""Auto-generated reproducible SciPy ODE simulator for {summary.get("dataset_name", "discovered_system")}."""',
        "import numpy as np",
        "from numpy import sin, cos, exp, log, sqrt",
        "from scipy.integrate import solve_ivp",
        "",
        f"def rhs(t, state):",
        f"    {unpack} = state",
        *body_lines,
        f"    return [{ret_list}]",
        "",
        "if __name__ == '__main__':",
        f"    x0 = {x0}",
        f"    t_eval = np.linspace(0.0, {t_end}, 500)",
        "    sol = solve_ivp(rhs, (0.0, t_eval[-1]), x0, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)",
        "    print('Simulated trajectory shape:', sol.y.T.shape)",
    ])


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return "<h1>ASDS-OMEGA v2.0 Workbench</h1><p>Static UI missing.</p>"
    return html_path.read_text(encoding="utf-8")


@app.get("/api/datasets")
def list_datasets() -> dict[str, Any]:
    return {"synthetic": SYNTHETIC_LIST, "real_world": REAL_LIST, "pdes": PDE_LIST}


@app.post("/api/discover")
def run_discovery(req: DiscoverRequest) -> dict[str, Any]:
    real_ids = {item["id"] for item in REAL_LIST}
    try:
        if req.dataset in real_ids:
            ds = load_real_dataset(req.dataset)
        else:
            ds = inject_noise(
                generate_synthetic_dataset(req.dataset, seed=req.seed),
                noise_level=req.noise_level,
                noise_type=req.noise_type,
                seed=req.seed,
            )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    cfg = PipelineConfig(
        seed=req.seed,
        ms_avp=MSAVPConfig(use_weak_form=req.use_weak_form, cumulant_debiasing=req.cumulant_debiasing),
        vifc=VIFCConfig(enabled=req.use_vifc),
        wlind=WLINDConfig(enabled=req.use_wlind),
        evghe=EVGHEConfig(enable_rationalizer=req.use_rationalizer),
    )
    res = ASDSOmegaPipeline(cfg).run(ds, run_active_falsification=req.run_active_falsification)
    payload = res.to_summary_dict()
    _enrich_latex(payload)
    payload["lean4_certificate"] = res.lean4_certificate
    payload["ground_truth_equations"] = [
        {"raw": gt, "latex": _expr_to_latex(gt, ds.var_names)}
        if not ("expected" in gt.lower() or "unknown" in gt.lower())
        else {"raw": gt, "latex": ""}
        for gt in ds.ground_truth_equations
    ]
    payload["trajectories"] = _build_trajectory_preview(
        ds, [c.expr for c in res.equations], payload.get("symmetry", {}).get("invariants", [])
    )
    return payload


@app.post("/api/discover_pde")
def run_pde_discovery(req: PDEDiscoverRequest) -> dict[str, Any]:
    try:
        pds = inject_noise(generate_synthetic_dataset(req.dataset, seed=req.seed), req.noise_level, seed=req.seed)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    cand = discover_pde_vesp(pds.x, pds.t, pds.u, cumulant_debiasing=req.cumulant_debiasing, seed=req.seed)
    # Downsample spacetime field for responsive WebGL heatmap
    sx = max(1, len(pds.x) // 64)
    st = max(1, len(pds.t) // 64)
    u_sub = pds.u[::st, ::sx]
    u_clean_sub = pds.u_clean[::st, ::sx]

    pde_syms = {k: sp.Symbol(k) for k in ["u", "u_x", "u_xx", "u_xxx"]}
    try:
        expr_tex = sp.latex(sp.sympify(cand.expr_str, locals=pde_syms))
        gt_tex = sp.latex(sp.sympify(pds.ground_truth_equation, locals=pde_syms))
    except Exception:
        expr_tex, gt_tex = cand.expr_str, pds.ground_truth_equation

    return {
        "dataset_name": pds.name,
        "domain": pds.domain,
        "ground_truth": pds.ground_truth_equation,
        "ground_truth_latex": gt_tex,
        "expr_str": cand.expr_str,
        "expr_latex": expr_tex,
        "weak_r2": round(float(cand.weak_r2), 5),
        "weak_mse": round(float(cand.weak_mse), 6),
        "coefficients": cand.coefficients,
        "x": [round(float(v), 4) for v in pds.x[::sx]],
        "t": [round(float(v), 4) for v in pds.t[::st]],
        "u_noisy": np.round(u_sub, 4).tolist(),
        "u_clean": np.round(u_clean_sub, 4).tolist(),
    }


@app.post("/api/counterfactual")
def run_counterfactual(req: CounterfactualRequest) -> dict[str, Any]:
    local_map = {v: sp.Symbol(v) for v in req.var_names}
    try:
        eqs = [sp.sympify(s, locals=local_map) for s in req.equations]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid symbolic expression: {exc}")
    t_eval = np.linspace(0.0, req.t_end, req.n_points)
    sim = CounterfactualSimulator(req.var_names, eqs)
    cf = sim.run_intervention(
        x0=np.asarray(req.x0, dtype=float),
        t_eval=t_eval,
        clamped_vars=req.clamped_vars,
        scale_equation=req.scale_equation,
    )
    return {
        "t": [round(float(v), 4) for v in cf.t],
        "baseline": [[None if not np.isfinite(v) else round(float(v), 4) for v in row] for row in cf.baseline_trajectory],
        "intervened": [[None if not np.isfinite(v) else round(float(v), 4) for v in row] for row in cf.intervened_trajectory],
        "ate_per_var": cf.ate_per_var,
        "peak_shift_per_var": cf.peak_shift_per_var,
    }


@app.post("/api/drift")
def run_drift_demo(noise_level: float = 0.02, seed: int = 42) -> dict[str, Any]:
    dds = inject_noise(gen_sir_regime_drift(), noise_level=noise_level, seed=seed)
    dres = SurgicalDriftRegenerator(seed=seed).run_stream(
        dds.t_list[0], dds.trajectories[0], dds.var_names, true_shift_time=20.0
    )
    vnames = dds.var_names
    return {
        "var_names": vnames,
        "drift_detected": dres.drift_detected,
        "detected_time": dres.detected_time,
        "true_shift_time": dres.true_shift_time,
        "detection_delay": dres.detection_delay,
        "drift_type": dres.drift_type,
        "pre_drift_equations": dres.pre_drift_equations,
        "pre_drift_latex": [_expr_to_latex(e, vnames) for e in dres.pre_drift_equations],
        "post_drift_equations": dres.post_drift_equations,
        "post_drift_latex": [_expr_to_latex(e, vnames) for e in dres.post_drift_equations],
        "pre_drift_r2_on_post": round(dres.pre_drift_r2_on_post, 4),
        "regenerated_r2_on_post": round(dres.regenerated_r2_on_post, 4),
        "conservation_preserved": dres.conservation_preserved_across_drift,
        "t": dres.time_series_t,
        "cusum": dres.cusum_trace,
        "residual": dres.residual_trace,
    }


@app.post("/api/discover_csv")
async def discover_csv(
    file: UploadFile = File(...),
    seed: int = Form(42),
    use_weak_form: bool = Form(True),
    cumulant_debiasing: bool = Form(True),
    use_vifc: bool = Form(True),
    use_wlind: bool = Form(True),
) -> dict[str, Any]:
    raw = (await file.read()).decode("utf-8")
    lines = [ln.strip() for ln in io.StringIO(raw) if ln.strip()]
    if len(lines) < 20:
        raise HTTPException(status_code=400, detail="CSV must contain header + at least 20 rows.")
    header = [h.strip() for h in lines[0].split(",")]
    if len(header) < 2:
        raise HTTPException(status_code=400, detail="CSV must have time column + at least 1 state variable.")
    data = np.loadtxt(lines[1:], delimiter=",")
    t_arr = data[:, 0]
    X_arr = data[:, 1:]
    var_names = header[1:]
    ds = DynamicalDataset(
        name=file.filename or "uploaded_csv",
        domain="Custom Uploaded Experimental Data",
        var_names=var_names,
        t_list=[t_arr],
        trajectories=[X_arr],
        clean_trajectories=[X_arr],
        ground_truth_equations=[f"unknown ({v})" for v in var_names],
        ground_truth_causal_adj=np.ones((len(var_names), len(var_names)), dtype=int),
    )
    cfg = PipelineConfig(
        seed=seed,
        ms_avp=MSAVPConfig(use_weak_form=use_weak_form, cumulant_debiasing=cumulant_debiasing),
        vifc=VIFCConfig(enabled=use_vifc),
        wlind=WLINDConfig(enabled=use_wlind),
    )
    res = ASDSOmegaPipeline(cfg).run(ds, run_active_falsification=False)
    payload = res.to_summary_dict()
    _enrich_latex(payload)
    payload["lean4_certificate"] = res.lean4_certificate
    payload["trajectories"] = _build_trajectory_preview(
        ds, [c.expr for c in res.equations], payload.get("symmetry", {}).get("invariants", [])
    )
    return payload


@app.post("/api/export", response_class=PlainTextResponse)
def export_artifact(req: ExportRequest) -> str:
    fmt = req.format.lower()
    if fmt == "markdown":
        return export_markdown_report(req.summary)
    if fmt == "latex":
        return export_latex_report(req.summary)
    if fmt == "scipy":
        return _generate_scipy_script(req.summary)
    if fmt == "lean4":
        return str(req.summary.get("lean4_certificate") or "-- No Lean 4 certificate generated.")
    raise HTTPException(status_code=400, detail=f"Unsupported export format: {req.format}")


@app.get("/api/benchmark_ledger")
def get_benchmark_ledger() -> dict[str, Any]:
    ledger_path = REPO_ROOT / "results" / "benchmark_ledger.json"
    if not ledger_path.exists():
        raise HTTPException(status_code=404, detail="No verified benchmark_ledger.json found. Run `asds-omega benchmark`.")
    raw = json.loads(ledger_path.read_text(encoding="utf-8"))
    exp1 = raw["experiments"]["exp1_multidomain_5pct"]
    exp2 = raw["experiments"]["exp2_noise_stress"]
    exp3 = raw["experiments"]["exp3_ablation_10pct"]
    return {
        "framework": raw.get("framework"),
        "n_seeds": raw.get("n_seeds"),
        "seeds": raw.get("seeds"),
        "exp1_odes": {k: v["multiseed"] for k, v in exp1["odes"].items()},
        "exp1_pdes": exp1["pdes"],
        "exp1_aggregate": exp1["aggregate_significance"],
        "exp2_noise_stress": exp2,
        "exp3_ablation": exp3,
    }
