from .statistical import Persistence, SeasonalNaive, Regression, AutoRegression, ARIMAForecaster
from .neural import NeuralForecaster
from .dynamical import SINDy, NeuralODE
from .mechanistic import MechanisticOscillator
from .hybrid import HybridForecaster
from .state_space import KalmanForecaster
from .references import XGBoostForecaster, PySINDyForecaster, NeuralForecastAdapter
from .latent import LatentODEForecaster

MODEL_NAMES = (
    "persistence",
    "seasonal_naive",
    "ar",
    "var",
    "arima",
    "ridge",
    "random_forest",
    "mlp",
    "lstm",
    "gru",
    "tcn",
    "dlinear",
    "transformer",
    "sindy",
    "neural_ode",
    "hybrid",
    "mechanistic",
)
MODEL_NAMES += (
    "nhits",
    "patchtst",
    "kalman",
    "xgboost",
    "pysindy",
    "nf_nhits",
    "nf_patchtst",
    "forecast_residual",
    "adaptive_hybrid",
    "average_hybrid",
    "physics_guided",
    "sindy_hybrid",
    "trainable_mechanistic",
)
MODEL_NAMES += ("latent_ode",)


def make_model(name, config, seed=0):
    neural = {
        k: config[k] for k in ("epochs", "batch_size", "lr", "patience", "device") if k in config
    }
    width = config.get("width", 32)
    if name == "latent_ode":
        return LatentODEForecaster(width, config.get("latent_size", 4), **neural)
    if name == "persistence":
        return Persistence()
    if name == "seasonal_naive":
        return SeasonalNaive(config.get("period", 24))
    if name in ("ar", "var"):
        return AutoRegression(name == "var", config.get("lags", 8))
    if name == "arima":
        return ARIMAForecaster(config.get("order", [2, 0, 0]))
    if name in ("ridge", "random_forest"):
        return Regression(name, config.get("alpha", 1.0), config.get("trees", 100), seed)
    if name in ("mlp", "lstm", "gru", "tcn", "dlinear", "transformer", "nhits", "patchtst"):
        return NeuralForecaster(name, width, **neural)
    if name == "sindy":
        return SINDy(
            config.get("degree", 2),
            config.get("threshold", 0.05),
            derivative=config.get("derivative", "finite"),
            smoothing_window=config.get("smoothing_window", 9),
        )
    if name == "pysindy":
        return PySINDyForecaster(config.get("degree", 2), config.get("threshold", 0.05))
    if name == "kalman":
        return KalmanForecaster(config.get("observation_variance", 0.01))
    if name == "xgboost":
        return XGBoostForecaster(config.get("trees", 100), config.get("depth", 4), seed)
    if name in ("nf_nhits", "nf_patchtst"):
        return NeuralForecastAdapter(name[3:], config.get("max_steps", 50), width, seed)
    if name in ("forecast_residual", "adaptive_hybrid", "average_hybrid", "physics_guided"):
        return HybridForecaster(
            name,
            width,
            prior=config.get("prior"),
            dynamics_weight=config.get("dynamics_weight", 0.01),
            short_weight=config.get("short_weight", 1.0),
            long_weight=config.get("long_weight", 1.0),
            **neural,
        )
    if name == "mechanistic":
        return MechanisticOscillator(**config.get("prior", {"stiffness": 1.0, "damping": 0.1}))
    if name in ("neural_ode", "hybrid", "sindy_hybrid", "trainable_mechanistic"):
        prior = (
            config.get("prior", {"stiffness": 1.0, "damping": 0.1})
            if name in ("hybrid", "trainable_mechanistic")
            else None
        )
        return NeuralODE(
            width,
            prior,
            train_prior=config.get("train_prior", name == "trainable_mechanistic"),
            residual=name != "trainable_mechanistic",
            sparse_prior=name == "sindy_hybrid",
            flow_steps=config.get("flow_steps", 1),
            **neural,
        )
    raise ValueError(f"Unknown model {name}; choose from {MODEL_NAMES}")
