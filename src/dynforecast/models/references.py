"""Optional, explicit adapters for correctness references and established baselines."""

import numpy as np
from dynforecast.data.pipeline import training_windows, trajectories
from .base import Forecaster
from .dynamical import SINDy, PolynomialLibrary


class XGBoostForecaster(Forecaster):
    def __init__(self, trees=100, depth=4, seed=0):
        self.trees, self.depth, self.seed = trees, depth, seed

    def fit(self, train, validation, targets, history, horizon, dt, **kwargs):
        from xgboost import XGBRegressor
        from sklearn.multioutput import MultiOutputRegressor

        x, y = training_windows(train, history, horizon, targets)
        self.model = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=self.trees,
                max_depth=self.depth,
                random_state=self.seed,
                n_jobs=2,
                objective="reg:squarederror",
            )
        )
        self.model.fit(x.reshape(len(x), -1), y.reshape(len(y), -1))
        self.targets, self.horizon = targets, horizon
        self.parameters = sum(
            len(est.get_booster().trees_to_dataframe()) for est in self.model.estimators_
        )
        self.diagnostics = {"capacity_definition": "total learned tree nodes"}
        return self

    def predict(self, histories):
        return self.model.predict(histories.reshape(len(histories), -1)).reshape(
            len(histories), self.horizon, len(self.targets)
        )


class PySINDyForecaster(SINDy):
    def fit(self, train, validation, targets, history, horizon, dt, **kwargs):
        import pysindy as ps

        parts = trajectories(train)
        self.reference = ps.SINDy(
            feature_library=ps.PolynomialLibrary(degree=self.degree),
            optimizer=ps.STLSQ(threshold=self.threshold, alpha=0.0),
            differentiation_method=ps.FiniteDifference(order=2),
        )
        self.reference.fit(parts, t=dt, multiple_trajectories=True)
        self.targets, self.horizon, self.dt = targets, horizon, dt
        self.library = PolynomialLibrary(parts[0].shape[1], self.degree)
        self.coefficients = self.reference.coefficients().T
        self.parameters = int(np.count_nonzero(self.coefficients))
        self.diagnostics = {
            "terms": self.library.names,
            "coefficients": self.coefficients.tolist(),
            "equations": self.reference.equations(),
            "implementation": "PySINDy STLSQ",
        }
        return self


class NeuralForecastAdapter(Forecaster):
    """Canonical Nixtla baselines. Fixed optimization steps; no test-driven checkpoint choice.

    Fits independent univariate targets. Multivariate experiments should use native
    nhits/patchtst, whose cross-channel contract is explicit.
    """

    def __init__(self, kind="nhits", max_steps=50, width=32, seed=0):
        self.kind, self.max_steps, self.width, self.seed = kind, max_steps, width, seed

    def fit(self, train, validation, targets, history, horizon, dt, **kwargs):
        import pandas as pd
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS, PatchTST

        if isinstance(train, (list, tuple)) or train.shape[1] != 1 or targets != [0]:
            raise ValueError(
                "Canonical NeuralForecast adapters support one univariate trajectory; native models support multivariate input"
            )
        frame = pd.DataFrame({"unique_id": "target", "ds": np.arange(len(train)), "y": train[:, 0]})
        common = dict(
            h=horizon,
            input_size=history,
            max_steps=self.max_steps,
            random_seed=self.seed,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            val_check_steps=max(1, self.max_steps),
            scaler_type="identity",
        )
        model = (
            NHITS(mlp_units=[[self.width, self.width]] * 3, **common)
            if self.kind == "nhits"
            else PatchTST(
                hidden_size=self.width,
                n_heads=2,
                encoder_layers=2,
                patch_len=min(8, history),
                stride=max(1, min(8, history) // 2),
                **common,
            )
        )
        self.reference = NeuralForecast(models=[model], freq=1)
        self.reference.fit(frame, val_size=0)
        self.horizon, self.history = horizon, history
        self.parameters = sum(p.numel() for p in self.reference.models[0].parameters())
        self.diagnostics = {
            "implementation": "NeuralForecast",
            "max_steps": self.max_steps,
            "validation_selection": "fixed optimization budget; no checkpoint selection",
            "training_loss": "NeuralForecast default MAE; native neural models use MSE",
        }
        return self

    def predict(self, histories):
        import pandas as pd

        result = []
        for history in histories:
            frame = pd.DataFrame(
                {"unique_id": "target", "ds": np.arange(len(history)), "y": history[:, 0]}
            )
            forecast = self.reference.predict(df=frame)
            column = self.reference.models[0].__class__.__name__
            result.append(forecast[column].to_numpy()[:, None])
        return np.stack(result)
