from abc import ABC, abstractmethod
import numpy as np


class Forecaster(ABC):
    """Common interface for all forecasters.

    ``fit`` sees only observed series (training and validation splits). ``predict``
    maps histories of shape ``[batch, history, inputs]`` to forecasts of shape
    ``[batch, horizon, targets]``. ``rollout`` records how the forecast is produced
    (direct, recursive, or continuous integration) and ``parameters`` the model size.
    """

    rollout = "direct"
    parameters = 0
    diagnostics = None

    @abstractmethod
    def fit(self, train, validation, targets, history, horizon, dt, **kwargs): ...

    @abstractmethod
    def predict(self, histories): ...


def check_predictions(predictions, expected):
    """Return predictions as a float array, failing on wrong shape or nonfinite values."""
    predictions = np.asarray(predictions, float)
    if predictions.shape != expected:
        raise ValueError(f"Prediction shape {predictions.shape}; expected {expected}")
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Forecast diverged or contains nonfinite values")
    return predictions
