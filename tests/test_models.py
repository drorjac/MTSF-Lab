import numpy as np
import pytest
import torch
from dynforecast.models import make_model
from dynforecast.models.dynamical import SINDy
from dynforecast.simulations import simulate
from dynforecast.training import seed_everything
from dynforecast.data import windows


@pytest.mark.parametrize(
    "name",
    [
        "persistence",
        "seasonal_naive",
        "ar",
        "var",
        "ridge",
        "random_forest",
        "mlp",
        "lstm",
        "gru",
        "tcn",
        "dlinear",
        "transformer",
        "arima",
        "sindy",
        "neural_ode",
        "hybrid",
        "mechanistic",
    ],
)
def test_all_models_shape_and_finite_predictions(name):
    seed_everything(7)
    data = simulate("oscillator", n=160, dt=0.03).observations
    cfg = {"epochs": 2, "width": 8, "period": 8, "lags": 4, "trees": 4}
    model = make_model(name, cfg)
    model.fit(data[:100], data[100:140], [0], 12, 4, 0.03)
    prediction = model.predict(data[140:152][None])
    assert prediction.shape == (1, 4, 1)
    assert np.isfinite(prediction).all()


def test_sindy_recovers_oscillator_coefficients_on_unseen_initial_state():
    data = simulate("oscillator", n=1000, dt=0.01).observations
    model = SINDy(degree=2, threshold=0.02).fit(data, data, [0, 1], 8, 30, 0.01)
    names = model.library.names
    assert abs(model.coefficients[names.index("x1"), 0] - 1) < 0.005
    assert abs(model.coefficients[names.index("x0"), 1] + 1) < 0.005
    assert abs(model.coefficients[names.index("x1"), 1] + 0.15) < 0.005
    unseen = simulate("oscillator", n=60, dt=0.01, initial=[-0.7, 0.8]).observations
    pred = model.predict(unseen[:8][None])[0]
    np.testing.assert_allclose(pred, unseen[8:38], atol=0.005)


def test_neural_sanity_outperforms_persistence_on_linear_trend():
    seed_everything(0)
    data = np.arange(180, dtype=float)[:, None] / 100
    model = make_model("mlp", {"epochs": 120, "width": 16, "lr": 0.01, "patience": 30})
    model.fit(data[:100], data[90:140], [0], 8, 3, 1.0)
    x, truth, _ = windows(data, 8, 3, [0], start=140)
    predictions = model.predict(x)
    naive = np.repeat(x[:, -1:, :], 3, axis=1)
    assert np.mean((predictions - truth) ** 2) < np.mean((naive - truth) ** 2)


def test_tcn_is_causal_within_history():
    from dynforecast.models.neural import TemporalNetwork

    seed_everything(1)
    model = TemporalNetwork("tcn", 1, 10, 2, 1, 8)
    one, two = torch.randn(1, 1, 10), torch.randn(1, 1, 10)
    two[:, :, :6] = one[:, :, :6]
    torch.testing.assert_close(model.body(one)[:, :, :6], model.body(two)[:, :, :6])


@pytest.mark.parametrize("name", ["mlp", "lstm", "tcn"])
def test_neural_models_learn_periodic_process(name):
    seed_everything(8)
    time = np.arange(400) * 0.2
    data = np.stack([np.sin(time), np.cos(time)], axis=-1)
    model = make_model(name, {"epochs": 60, "width": 16, "lr": 0.005, "patience": 15})
    model.fit(data[:240], data[228:320], [0], 12, 6, 0.2)
    x, y, _ = windows(data, 12, 6, [0], start=320)
    predictions = model.predict(x)
    naive = np.repeat(x[:, -1:, [0]], 6, axis=1)
    assert np.mean((predictions - y) ** 2) < np.mean((naive - y) ** 2)


@pytest.mark.parametrize("name", ["neural_ode", "hybrid"])
def test_learned_flow_generalizes_to_unseen_initial_condition(name):
    seed_everything(8)
    series = simulate("oscillator", n=300, dt=0.05, params={"damping": 0.0})
    model = make_model(name, {"epochs": 50, "width": 16, "lr": 0.01, "patience": 15})
    model.fit(series.observations[:200], series.observations[200:260], [0, 1], 8, 8, 0.05)
    unseen = simulate("oscillator", n=30, dt=0.05, params={"damping": 0.0}, initial=[-0.7, 0.3])
    prediction = model.predict(unseen.observations[:8][None])[0]
    naive = np.repeat(unseen.observations[7:8], 8, axis=0)
    assert np.mean((prediction - unseen.observations[8:16]) ** 2) < np.mean(
        (naive - unseen.observations[8:16]) ** 2
    )
