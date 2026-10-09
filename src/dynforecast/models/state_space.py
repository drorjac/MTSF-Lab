"""Observed linear Gaussian dynamics with a Kalman filter; no hidden simulator states."""

import numpy as np
from dynforecast.data.pipeline import transition_pairs
from .base import Forecaster


class KalmanForecaster(Forecaster):
    """Linear Gaussian state-space model fitted by ridge regression on transitions.

    Filters each history with a Kalman filter, then propagates the mean."""

    rollout = "recursive"

    def __init__(self, observation_variance=0.01, ridge=0.001):
        if observation_variance <= 0:
            raise ValueError("Observation variance must be positive")
        self.observation_variance, self.ridge = observation_variance, ridge

    def fit(self, train, validation, targets, history, horizon, dt, **kwargs):
        x, y = transition_pairs(train)
        design = np.column_stack([x, np.ones(len(x))])
        regularizer = np.eye(design.shape[1]) * self.ridge
        regularizer[-1, -1] = 0
        coef = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
        self.transition, self.bias = coef[:-1].T, coef[-1]
        residual = y - design @ coef
        features = x.shape[1]
        self.q = np.atleast_2d(np.cov(residual, rowvar=False)) + np.eye(features) * 1e-8
        self.r = np.eye(features) * self.observation_variance
        self.targets, self.horizon = targets, horizon
        self.parameters = coef.size + self.q.size
        self.diagnostics = {
            "transition": self.transition.tolist(),
            "process_covariance": self.q.tolist(),
            "observation_variance": self.observation_variance,
            "observation_operator": "identity on observed coordinates",
        }
        return self

    def predict(self, histories):
        output, eye = [], np.eye(len(self.bias))
        for history in histories:
            state, covariance = history[0].copy(), eye.copy()
            for observed in history[1:]:
                state = self.transition @ state + self.bias
                covariance = self.transition @ covariance @ self.transition.T + self.q
                gain = np.linalg.solve((covariance + self.r).T, covariance.T).T
                state += gain @ (observed - state)
                # Joseph form protects positive semidefiniteness under roundoff.
                covariance = (eye - gain) @ covariance @ (eye - gain).T + gain @ self.r @ gain.T
            future = []
            for _ in range(self.horizon):
                state = self.transition @ state + self.bias
                future.append(state[self.targets].copy())
            output.append(future)
        return np.asarray(output)
