import json
import numpy as np
import pytest
from dynforecast.models import make_model
from dynforecast.simulations import simulate
from dynforecast.training import seed_everything
from dynforecast.data.pipeline import training_windows, transition_pairs
from dynforecast.experiments import run_experiment


@pytest.mark.parametrize(
    "name",
    [
        "nhits",
        "patchtst",
        "kalman",
        "xgboost",
        "pysindy",
        "forecast_residual",
        "adaptive_hybrid",
        "average_hybrid",
        "physics_guided",
        "sindy_hybrid",
        "trainable_mechanistic",
        "latent_ode",
    ],
)
def test_extended_models_end_to_end(name, tmp_path):
    seed_everything(5)
    result = run_experiment(
        {
            "dataset": {"n": 180, "dt": 0.03},
            "model": {"name": name, "epochs": 2, "width": 8, "trees": 3},
            "history": 12,
            "horizon": 4,
            "targets": [0, 1],
            "output": str(tmp_path),
        }
    )
    assert result["status"] == "completed", result.get("error")
    with np.load(tmp_path / result["run_id"] / "predictions.npz") as arrays:
        assert np.isfinite(arrays["predictions"]).all()


def test_trajectory_windows_never_join_independent_states():
    parts = [np.zeros((20, 1)), np.ones((20, 1)) * 100]
    x, y = training_windows(parts, 4, 3, [0])
    assert all(np.unique(window).size == 1 for window in x)
    assert all(np.unique(window).size == 1 for window in y)
    one, two = transition_pairs(parts, 3)
    assert np.all(two[:, :, 0] == one[:, [0]])


@pytest.mark.parametrize("name", ["persistence", "var", "lstm", "sindy", "hybrid"])
def test_unseen_trajectory_split(name, tmp_path):
    result = run_experiment(
        {
            "dataset": {"n": 100, "trajectories": 5},
            "model": {"name": name, "epochs": 2, "width": 8},
            "history": 8,
            "horizon": 3,
            "targets": [0, 1],
            "output": str(tmp_path),
        }
    )
    assert result["status"] == "completed", result.get("error")
    pre = json.loads((tmp_path / result["run_id"] / "preprocessing.json").read_text())
    assert set(pre["train_trajectory_ids"]).isdisjoint(pre["test_trajectory_ids"])
    assert result["metrics"]["n_training_trajectories"] == 3


def test_pysindy_agrees_with_transparent_implementation():
    data = simulate(n=600, dt=0.01).observations
    reference = make_model("pysindy", {"threshold": 0.02})
    native = make_model("sindy", {"threshold": 0.02})
    for model in (reference, native):
        model.fit(data, data, [0, 1], 8, 4, 0.01)
    np.testing.assert_allclose(native.coefficients, reference.coefficients, atol=0.002)


def test_noise_derivative_smoothing_reduces_error():
    data = simulate(n=600, dt=0.01, noise=0.02).observations
    finite = make_model("sindy", {"degree": 1, "threshold": 0.02})
    smooth = make_model(
        "sindy", {"degree": 1, "threshold": 0.02, "derivative": "savgol", "smoothing_window": 21}
    )
    for model in (finite, smooth):
        model.fit(data, data, [0, 1], 8, 4, 0.01)
    assert (
        smooth.diagnostics["training_derivative_rmse"]
        < finite.diagnostics["training_derivative_rmse"]
    )


def test_irregular_drift_and_spatial_simulations():
    irregular = simulate(n=100, irregular=0.2)
    assert not np.allclose(np.diff(irregular.time), 0.05)
    normal = simulate(n=200)
    drift = simulate(n=200, drift={"at": 5.0, "params": {"stiffness": 2.0}})
    np.testing.assert_allclose(normal.latent[:80], drift.latent[:80], atol=1e-6)
    assert not np.allclose(normal.latent[-30:], drift.latent[-30:])
    field = simulate("reaction_diffusion", n=100, grid=8)
    assert field.observations.shape == (100, 8)
    assert np.isfinite(field.derivatives).all()
