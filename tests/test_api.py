"""Integration tests for the FastAPI Scientific Discovery Workbench endpoints & GUI helpers."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.server import app
from asds_omega.gui import _find_free_port

client = TestClient(app)


def test_studio_home_and_datasets():
    r_home = client.get("/")
    assert r_home.status_code == 200
    assert "ASDS-OMEGA v2.0" in r_home.text
    assert "katex" in r_home.text.lower()

    r_ds = client.get("/api/datasets")
    assert r_ds.status_code == 200
    payload = r_ds.json()
    assert len(payload["synthetic"]) == 8
    assert len(payload["real_world"]) == 3
    assert len(payload["pdes"]) == 2


def test_studio_discover_counterfactual_and_exports():
    r = client.post(
        "/api/discover",
        json={"dataset": "pendulum", "noise_level": 0.05, "seed": 42, "run_active_falsification": False},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["evidence"]["symbolic_match_score"] >= 0.98
    assert "lean4_certificate" in data and "namespace ASDSOmega." in data["lean4_certificate"]
    assert all("latex" in eq for eq in data["discovered_equations"])
    assert "residuals" in data["trajectories"]
    assert "invariant_traces" in data["trajectories"]

    r_cf = client.post(
        "/api/counterfactual",
        json={
            "var_names": data["var_names"],
            "equations": [e["expr_str"] for e in data["discovered_equations"]],
            "x0": data["trajectories"]["x0"],
            "t_end": 5.0,
            "n_points": 80,
            "scale_equation": {"omega": 0.5},
        },
    )
    assert r_cf.status_code == 200
    cf = r_cf.json()
    assert len(cf["t"]) == 80
    assert "omega" in cf["ate_per_var"]

    for fmt, needle in [
        ("scipy", "solve_ivp"),
        ("latex", "\\begin{align}"),
        ("markdown", "# ASDS-OMEGA"),
        ("lean4", "namespace ASDSOmega."),
    ]:
        rexp = client.post("/api/export", json={"summary": data, "format": fmt})
        assert rexp.status_code == 200
        assert needle in rexp.text


def test_studio_pde_and_benchmark_ledger_and_gui_port():
    r_pde = client.post(
        "/api/discover_pde",
        json={"dataset": "burgers_pde", "noise_level": 0.05, "cumulant_debiasing": True, "seed": 42},
    )
    assert r_pde.status_code == 200
    pde = r_pde.json()
    assert pde["weak_r2"] > 0.95
    assert "u_xx" in pde["expr_str"]
    assert "expr_latex" in pde

    r_led = client.get("/api/benchmark_ledger")
    assert r_led.status_code == 200
    led = r_led.json()
    assert led["n_seeds"] == 5
    assert len(led["exp1_odes"]) == 8

    port = _find_free_port()
    assert 1024 <= port <= 65535
