import time
import numpy as np
from dynforecast.experiments import run_experiment, run_benchmark
from dynforecast.visualization import generate_report
from dynforecast.data import load_csv, load_tsf


def test_end_to_end_resume_equal_targets_and_report(tmp_path):
    cfg = {
        "dataset": {"name": "oscillator", "n": 200},
        "history": 12,
        "horizon": 4,
        "inputs": [1, 0],
        "targets": [0],
        "output": str(tmp_path),
    }
    first = run_experiment(cfg)
    assert first["status"] == "completed", first.get("error")
    folder = tmp_path / first["run_id"]
    stamp = (folder / "metrics.json").stat().st_mtime_ns
    assert run_experiment(cfg)["run_id"] == first["run_id"]
    assert (folder / "metrics.json").stat().st_mtime_ns == stamp
    values = np.load(folder / "predictions.npz")
    from dynforecast.simulations import simulate

    series = simulate(n=200)
    np.testing.assert_allclose(values["truth"][0, :, 0], series.observations[160:164, 0])
    assert (generate_report(tmp_path) / "summary.html").exists()


def test_partial_dynamics_is_explicit_and_failures_recorded(tmp_path):
    outcome = run_experiment(
        {
            "dataset": {"n": 160},
            "history": 8,
            "horizon": 3,
            "inputs": [0],
            "model": {"name": "sindy"},
            "output": str(tmp_path),
        }
    )
    assert outcome["status"] == "failed"
    assert "dynamics_assumption" in outcome["error"]
    assert (tmp_path / outcome["run_id"] / "error.log").exists()


def test_training_fraction_changes_only_training_window(tmp_path):
    base = {"dataset": {"n": 200}, "history": 8, "horizon": 3, "output": str(tmp_path)}
    full = run_experiment(base)
    small = run_experiment({**base, "train_fraction": 0.5})
    one = np.load(tmp_path / full["run_id"] / "predictions.npz")
    two = np.load(tmp_path / small["run_id"] / "predictions.npz")
    np.testing.assert_array_equal(one["origins"], two["origins"])
    np.testing.assert_array_equal(one["truth"], two["truth"])
    assert small["metrics"]["n_training_observations"] == 60


def test_budget_and_resume_checkpoint(tmp_path):
    cfg = {
        "dataset": {"n": 160},
        "model": {"name": "mlp", "epochs": 4},
        "history": 8,
        "horizon": 3,
        "output": str(tmp_path),
    }
    outcome = run_experiment(cfg, deadline=time.monotonic() - 1)
    assert outcome["status"] == "interrupted"
    assert run_experiment(cfg)["status"] == "completed"
    result = run_benchmark({**cfg, "budget_seconds": 0.000001, "sweep": {"seed": [0, 1]}})
    assert result["processed"] < 2


def test_csv_and_monash_loader(tmp_path):
    csv = tmp_path / "sensor.csv"
    csv.write_text("date,temp,pressure\n2020-01-01,1,3\n2020-01-02,2,4\n2020-01-03,3,5\n")
    data = load_csv(csv)
    assert data.names == ["temp", "pressure"]
    assert data.time[1] == 86400
    tsf = tmp_path / "sample.tsf"
    tsf.write_text(
        "@attribute series_name string\n@frequency daily\n@data\ns1:1,2,?,4\ns2:5,6,7,8\n"
    )
    collection = load_tsf(tsf)
    assert len(collection) == 2
    assert np.isnan(collection[0].observations[2, 0])
    csv.write_text("sensor\n1\n-9999\n2\n")
    sensor = load_csv(csv, missing_values={"sensor": -9999})
    assert np.isnan(sensor.observations[1, 0])


def test_ingestion_failure_has_a_manifest(tmp_path):
    outcome = run_experiment(
        {
            "dataset": {"name": "csv", "path": str(tmp_path / "absent.csv")},
            "output": str(tmp_path / "runs"),
        }
    )
    assert outcome["status"] == "failed" and outcome["stage"] == "ingestion"
    assert (tmp_path / "runs" / outcome["run_id"] / "manifest.json").exists()


def test_nonzero_target_subset_uses_correct_local_coordinates(tmp_path):
    outcome = run_experiment(
        {
            "dataset": {"name": "lorenz", "n": 200, "dt": 0.01},
            "inputs": [2],
            "targets": [2],
            "history": 8,
            "horizon": 3,
            "output": str(tmp_path),
        }
    )
    assert outcome["status"] == "completed", outcome.get("error")
    from dynforecast.simulations import simulate

    data = simulate("lorenz", n=200, dt=0.01)
    saved = np.load(tmp_path / outcome["run_id"] / "predictions.npz")
    np.testing.assert_allclose(saved["truth"][0, :, 0], data.observations[160:163, 2])
