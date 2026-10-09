import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from dynforecast.data.pipeline import training_windows, trajectories
from .base import Forecaster


class Persistence(Forecaster):
    def fit(self, train, validation, targets, history, horizon, dt, **kwargs):
        self.targets, self.horizon = targets, horizon
        return self

    def predict(self, histories):
        return np.repeat(histories[:, -1:, self.targets], self.horizon, axis=1)


class SeasonalNaive(Persistence):
    def __init__(self, period=24):
        if period < 1:
            raise ValueError("Seasonal period must be positive")
        self.period = period

    def predict(self, histories):
        if histories.shape[1] < self.period:
            raise ValueError("History is shorter than the seasonal period")
        indices = -self.period + np.arange(self.horizon) % self.period
        return histories[:, indices][:, :, self.targets]


class Regression(Forecaster):
    def __init__(self, kind="ridge", alpha=1.0, trees=100, seed=0):
        self.estimator = (
            Ridge(alpha=alpha)
            if kind == "ridge"
            else RandomForestRegressor(n_estimators=trees, random_state=seed, n_jobs=2)
        )

    def fit(self, train, validation, targets, history, horizon, dt, **kwargs):
        self.horizon, self.targets = horizon, targets
        x, y = training_windows(train, history, horizon, targets)
        self.estimator.fit(x.reshape(len(x), -1), y.reshape(len(y), -1))
        if hasattr(self.estimator, "coef_"):
            self.parameters = int(self.estimator.coef_.size + self.estimator.intercept_.size)
        else:
            self.parameters = sum(
                tree.tree_.value.size + np.sum(tree.tree_.children_left >= 0)
                for tree in self.estimator.estimators_
            )
            self.diagnostics = {
                "capacity_definition": "leaf/node prediction scalars plus split thresholds"
            }
        return self

    def predict(self, histories):
        return self.estimator.predict(histories.reshape(len(histories), -1)).reshape(
            len(histories), self.horizon, len(self.targets)
        )


class AutoRegression(Forecaster):
    rollout = "recursive"

    def __init__(self, multivariate=False, lags=8, alpha=0.001):
        self.multivariate, self.lags, self.alpha = multivariate, lags, alpha

    def fit(self, train, validation, targets, history, horizon, dt, **kwargs):
        if self.lags > history:
            raise ValueError("AR/VAR lags cannot exceed input history")
        self.targets, self.horizon = targets, horizon
        self.inputs = list(range(trajectories(train)[0].shape[1])) if self.multivariate else targets
        values = [v[:, self.inputs] for v in trajectories(train)]
        x, y = training_windows(values, self.lags, 1, list(range(len(self.inputs))))
        self.estimator = Ridge(alpha=self.alpha).fit(x.reshape(len(x), -1), y[:, 0])
        self.parameters = self.estimator.coef_.size + self.estimator.intercept_.size
        return self

    def predict(self, histories):
        values = histories[:, -self.lags :, self.inputs].copy()
        output = []
        for _ in range(self.horizon):
            nxt = self.estimator.predict(values.reshape(len(values), -1))
            output.append(nxt)
            values = np.concatenate([values[:, 1:], nxt[:, None]], axis=1)
        indices = [self.inputs.index(t) for t in self.targets]
        return np.stack(output, axis=1)[:, :, indices]


class ARIMAForecaster(Forecaster):
    rollout = "recursive"

    def __init__(self, order=(2, 0, 0)):
        self.order = tuple(order)

    def fit(self, train, validation, targets, history, horizon, dt, **kwargs):
        if isinstance(train, (list, tuple)):
            raise ValueError(
                "ARIMA requires a single continuous training series; use VAR for independent trajectories"
            )
        self.targets, self.horizon = targets, horizon
        self.fits = [
            ARIMA(train[:, t], order=self.order).fit(method_kwargs={"maxiter": 500})
            for t in targets
        ]
        if any(not f.mle_retvals.get("converged", True) for f in self.fits):
            raise RuntimeError("ARIMA optimization failed to converge")
        self.parameters = sum(len(f.params) for f in self.fits)
        return self

    def predict(self, histories):
        # Frozen training parameters; local filtering uses only each available history.
        return np.stack(
            [
                np.stack(
                    [
                        fit.apply(h[:, t], refit=False).forecast(self.horizon)
                        for fit, t in zip(self.fits, self.targets)
                    ],
                    axis=-1,
                )
                for h in histories
            ]
        )
