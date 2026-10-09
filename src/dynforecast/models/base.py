from abc import ABC, abstractmethod
import numpy as np


class Forecaster(ABC):
    """Fit only observations. predict accepts [batch, history, inputs] and returns [batch,H,targets]."""

    rollout = "direct"
    parameters = 0
    diagnostics = None

    @abstractmethod
    def fit(self, train, validation, targets, history, horizon, dt, **kwargs): ...

    @abstractmethod
    def predict(self, histories): ...


def check_predictions(predictions, expected):
    predictions = np.asarray(predictions, float)
    if predictions.shape != expected:
        raise ValueError(f"Prediction shape {predictions.shape}; expected {expected}")
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Forecast diverged or contains nonfinite values")
    return predictions
