import json
import pytest
from dynforecast.experiments.tuning import tune, backtest
from dynforecast.experiments.runner import run_experiment, run_benchmark
from dynforecast.visualization.dashboard import generate_dashboard
from dynforecast.data.builtin import load_builtin


def test_tuning_reads_validation_before_one_selected_test(tmp_path):
    result = tune(
        {
            "dataset": {"n": 180},
            "history": 8,
            "horizon": 3,
            "model": {"name": "ridge"},
            "output": str(tmp_path),
            "tuning": {"trials": 2, "budget_seconds": 30, "space": {"model.alpha": [0.01, 1.0]}},
        }
    )
    assert result["status"] == "completed"
    trials = list((tmp_path / "tuning" / "trials").glob("*/manifest.json"))
    assert trials and all(
        json.loads(p.read_text())["config"]["evaluation"] == "validation" for p in trials
    )
    selected = list((tmp_path / "tuning" / "selected").glob("*/manifest.json"))
    assert len(selected) == 1
    assert json.loads(selected[0].read_text())["config"]["evaluation"] == "test"
    again = tune(
        {
            "dataset": {"n": 180},
            "history": 8,
            "horizon": 3,
            "model": {"name": "ridge"},
            "output": str(tmp_path),
            "tuning": {"trials": 2, "budget_seconds": 30, "space": {"model.alpha": [0.01, 1.0]}},
        }
    )
    assert again["test_run_id"] == result["test_run_id"]


def test_expanding_origin_backtest(tmp_path):
    outcome = backtest(
        {
            "dataset": {"n": 180},
            "history": 8,
            "horizon": 3,
            "output": str(tmp_path),
            "backtest": {"folds": [0.3, 0.5, 0.65]},
        }
    )
    assert outcome["status"] == "completed" and len(outcome["folds"]) == 3


def test_hard_budget_records_interruption(tmp_path):
    outcome = run_benchmark(
        {
            "dataset": {"n": 180},
            "model": {"name": "lstm", "epochs": 100000},
            "history": 8,
            "horizon": 3,
            "output": str(tmp_path),
            "budget_seconds": 0.3,
            "scheduler": {"isolated": True, "devices": ["cpu"]},
            "sweep": {"seed": [0, 1]},
        }
    )
    assert outcome["interrupted"] == 1 and outcome["unlaunched"] == 1
    assert list(tmp_path.glob("*/manifest.json"))


@pytest.mark.parametrize("name", ["co2", "sunspots", "macrodata"])
def test_real_measurement_sources_available_offline(name):
    series = load_builtin(name, max_rows=150)
    assert len(series.time) == 150
    assert series.metadata["provenance"] == "statsmodels public dataset"


def test_tracking_and_dashboard_artifacts(tmp_path):
    result = run_experiment(
        {
            "dataset": {"n": 160},
            "history": 8,
            "horizon": 3,
            "model": {"name": "mlp", "epochs": 2, "width": 8},
            "output": str(tmp_path),
            "tracking": {"tensorboard": True, "mlflow": True},
        }
    )
    assert result["status"] == "completed", result.get("error")
    folder = tmp_path / result["run_id"]
    assert (folder / "mlflow.json").exists()
    assert list((folder / "tensorboard").glob("events*"))
    page = generate_dashboard(tmp_path)
    assert "plotly.js" in page.read_text().lower()
    assert "History → forecast" in page.read_text()


@pytest.mark.parametrize("name", ["nf_nhits", "nf_patchtst"])
def test_canonical_library_adapters(name, tmp_path):
    result = run_experiment(
        {
            "dataset": {"n": 160},
            "history": 12,
            "horizon": 3,
            "inputs": [0],
            "model": {"name": name, "max_steps": 2, "width": 8},
            "stride": 12,
            "output": str(tmp_path),
        }
    )
    assert result["status"] == "completed", result.get("error")
