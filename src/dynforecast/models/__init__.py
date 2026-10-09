"""Model registry.

Every forecaster is registered under a short name together with two flags the
experiment runner needs before fitting:

* ``dynamical``: the model learns an autonomous evolution law, so it needs
  regularly sampled, unforced observations.
* ``uses_prior``: the model relies on the damped-oscillator prior and only
  applies to the two-state oscillator observations.

Use :func:`make_model` to build a model from a config dict.
"""

from dataclasses import dataclass
from typing import Callable

from .base import Forecaster
from .dynamical import SINDy, NeuralODE
from .hybrid import HybridForecaster
from .latent import LatentODEForecaster
from .mechanistic import MechanisticOscillator
from .neural import NeuralForecaster
from .references import XGBoostForecaster, PySINDyForecaster, NeuralForecastAdapter
from .state_space import KalmanForecaster
from .statistical import Persistence, SeasonalNaive, Regression, AutoRegression, ARIMAForecaster

DEFAULT_PRIOR = {"stiffness": 1.0, "damping": 0.1}
NEURAL_KEYS = ("epochs", "batch_size", "lr", "patience", "device")


@dataclass(frozen=True)
class ModelSpec:
    build: Callable[[str, dict, int], Forecaster]
    dynamical: bool = False
    uses_prior: bool = False


def _neural_options(config):
    return {k: config[k] for k in NEURAL_KEYS if k in config}


def _temporal(name, config, seed):
    return NeuralForecaster(name, config.get("width", 32), **_neural_options(config))


def _sindy(name, config, seed):
    return SINDy(
        config.get("degree", 2),
        config.get("threshold", 0.05),
        derivative=config.get("derivative", "finite"),
        smoothing_window=config.get("smoothing_window", 9),
    )


def _hybrid(name, config, seed):
    return HybridForecaster(
        name,
        config.get("width", 32),
        prior=config.get("prior"),
        dynamics_weight=config.get("dynamics_weight", 0.01),
        short_weight=config.get("short_weight", 1.0),
        long_weight=config.get("long_weight", 1.0),
        **_neural_options(config),
    )


def _neural_ode(name, config, seed):
    """Neural ODE family: plain, residual on the oscillator prior, sparse prior, or
    the oscillator alone with trainable parameters."""
    with_prior = name in ("hybrid", "trainable_mechanistic")
    trainable = name == "trainable_mechanistic"
    return NeuralODE(
        config.get("width", 32),
        config.get("prior", DEFAULT_PRIOR) if with_prior else None,
        train_prior=config.get("train_prior", trainable),
        residual=not trainable,
        sparse_prior=name == "sindy_hybrid",
        flow_steps=config.get("flow_steps", 1),
        **_neural_options(config),
    )


REGISTRY: dict[str, ModelSpec] = {
    # Baselines and statistical models
    "persistence": ModelSpec(lambda n, c, s: Persistence()),
    "seasonal_naive": ModelSpec(lambda n, c, s: SeasonalNaive(c.get("period", 24))),
    "ar": ModelSpec(lambda n, c, s: AutoRegression(False, c.get("lags", 8))),
    "var": ModelSpec(lambda n, c, s: AutoRegression(True, c.get("lags", 8))),
    "arima": ModelSpec(lambda n, c, s: ARIMAForecaster(c.get("order", [2, 0, 0]))),
    "kalman": ModelSpec(lambda n, c, s: KalmanForecaster(c.get("observation_variance", 0.01))),
    # Classical machine learning on lagged windows
    "ridge": ModelSpec(lambda n, c, s: Regression(n, c.get("alpha", 1.0), c.get("trees", 100), s)),
    "random_forest": ModelSpec(
        lambda n, c, s: Regression(n, c.get("alpha", 1.0), c.get("trees", 100), s)
    ),
    "xgboost": ModelSpec(
        lambda n, c, s: XGBoostForecaster(c.get("trees", 100), c.get("depth", 4), s)
    ),
    # Temporal neural networks
    **{
        name: ModelSpec(_temporal)
        for name in ("mlp", "lstm", "gru", "tcn", "dlinear", "transformer", "nhits", "patchtst")
    },
    # Learned evolution laws
    "sindy": ModelSpec(_sindy, dynamical=True),
    "pysindy": ModelSpec(
        lambda n, c, s: PySINDyForecaster(c.get("degree", 2), c.get("threshold", 0.05)),
        dynamical=True,
    ),
    "neural_ode": ModelSpec(_neural_ode, dynamical=True),
    "latent_ode": ModelSpec(
        lambda n, c, s: LatentODEForecaster(
            c.get("width", 32), c.get("latent_size", 4), **_neural_options(c)
        ),
        dynamical=True,
    ),
    # Mechanistic and hybrid models
    "mechanistic": ModelSpec(
        lambda n, c, s: MechanisticOscillator(**c.get("prior", DEFAULT_PRIOR)),
        dynamical=True,
        uses_prior=True,
    ),
    "trainable_mechanistic": ModelSpec(_neural_ode, dynamical=True, uses_prior=True),
    "hybrid": ModelSpec(_neural_ode, dynamical=True, uses_prior=True),
    "sindy_hybrid": ModelSpec(_neural_ode, dynamical=True),
    **{
        name: ModelSpec(_hybrid, uses_prior=True)
        for name in ("forecast_residual", "adaptive_hybrid", "average_hybrid", "physics_guided")
    },
    # Nixtla NeuralForecast reference implementations
    **{
        name: ModelSpec(
            lambda n, c, s: NeuralForecastAdapter(
                n.removeprefix("nf_"), c.get("max_steps", 50), c.get("width", 32), s
            )
        )
        for name in ("nf_nhits", "nf_patchtst")
    },
}

MODEL_NAMES = tuple(REGISTRY)
DYNAMICAL_MODELS = frozenset(name for name, spec in REGISTRY.items() if spec.dynamical)
PRIOR_MODELS = frozenset(name for name, spec in REGISTRY.items() if spec.uses_prior)


def make_model(name: str, config: dict, seed: int = 0) -> Forecaster:
    """Build the registered model ``name`` from its ``config`` section."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown model {name}; choose from {MODEL_NAMES}")
    return REGISTRY[name].build(name, config, seed)
