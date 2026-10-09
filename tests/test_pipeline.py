import numpy as np
import pytest
from dynforecast.data import Standardizer, windows, chronological_bounds, Series


def test_training_only_statistics_and_causal_imputation():
    scaler = Standardizer().fit(np.array([[1.0, 10.0], [3.0, 20.0]]))
    np.testing.assert_allclose(scaler.mean, [2.0, 15.0])
    values = np.array([[np.nan, 30.0], [1000.0, np.nan], [np.nan, 9999.0]])
    filled = scaler.impute(values)
    np.testing.assert_allclose(filled, [[2.0, 30.0], [1000.0, 30.0], [1000.0, 9999.0]])
    np.testing.assert_allclose(scaler.mean, [2.0, 15.0])


def test_windows_do_not_cross_target_split():
    values = np.arange(40)[:, None]
    x, y, origins = windows(values, 5, 3, [0], start=20, stop=30)
    assert origins[0] == 20 and origins[-1] == 27
    assert x[0, -1, 0] == 19 and y[0, 0, 0] == 20
    assert y.max() == 29
    with pytest.raises(ValueError):
        windows(values, 5, 10, [0], start=39)


def test_contract_rejects_unordered_and_empty_features():
    with pytest.raises(ValueError):
        Series(np.array([0, 2, 1]), np.ones((3, 1)), ["x"])
    with pytest.raises(ValueError):
        Standardizer().fit(np.full((10, 1), np.nan))
    with pytest.raises(ValueError):
        chronological_bounds(100, 0.8, 0.3)
