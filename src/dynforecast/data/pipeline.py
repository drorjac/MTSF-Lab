"""Temporal contracts. Only training observations determine preprocessing state."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Series:
    """Observed time series with optional simulator-only ground truth.

    ``observations`` is ``[time, features]``. ``latent`` and ``derivatives`` are
    set only for simulated systems and are used for evaluation, never fitting.
    ``inputs`` holds exogenous forcing, if any.
    """

    time: np.ndarray
    observations: np.ndarray
    names: list[str]
    metadata: dict = field(default_factory=dict)
    latent: np.ndarray | None = None
    derivatives: np.ndarray | None = None
    inputs: np.ndarray | None = None

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=float)
        self.observations = np.asarray(self.observations, dtype=float)
        if self.observations.ndim == 1:
            self.observations = self.observations[:, None]
        if self.observations.shape != (len(self.time), len(self.names)):
            raise ValueError("Observations must have shape (time, observed features)")
        if len(self.time) < 3 or not np.all(np.diff(self.time) > 0):
            raise ValueError("At least three strictly increasing timestamps are required")
        if np.isinf(self.observations).any():
            raise ValueError("Infinite observations are invalid; use NaN for missing data")


def chronological_bounds(n, train=0.6, validation=0.2):
    if not 0 < train < 1 or not 0 < validation < 1 or train + validation >= 1:
        raise ValueError("Train, validation and test fractions must be positive")
    return int(n * train), int(n * (train + validation))


class Standardizer:
    """Per-feature z-scoring fitted on training data, ignoring missing values."""

    def fit(self, training):
        if np.any(np.all(np.isnan(training), axis=0)):
            raise ValueError("A training feature contains no observed values")
        self.mean = np.nanmean(training, axis=0)
        self.scale = np.nanstd(training, axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        return self

    def impute(self, values):
        """Forward fill; initial gaps use training means, never future observations."""
        result = np.asarray(values, float).copy()
        last = self.mean.copy()
        for i, row in enumerate(result):
            row = np.where(np.isnan(row), last, row)
            result[i] = row
            last = row
        return result

    def transform(self, values):
        return (self.impute(values) - self.mean) / self.scale

    def inverse(self, values, targets):
        return values * self.scale[targets] + self.mean[targets]


def windows(values, history, horizon, targets, start=None, stop=None, stride=1):
    """Origins index the first target. Targets stay entirely inside [start, stop)."""
    if min(history, horizon, stride) < 1:
        raise ValueError("History, horizon and stride must be positive")
    start = max(history, history if start is None else start)
    stop = len(values) if stop is None else stop
    origins = np.arange(start, stop - horizon + 1, stride, dtype=int)
    if not len(origins):
        raise ValueError("Split too short for the requested history and horizon")
    x = np.stack([values[o - history : o] for o in origins])
    y = np.stack([values[o : o + horizon, targets] for o in origins])
    return x, y, origins


def trajectories(values):
    """Normalize an array or an independent-trajectory collection without joining boundaries."""
    return list(values) if isinstance(values, (list, tuple)) else [values]


def training_windows(values, history, horizon, targets):
    parts = [windows(v, history, horizon, targets) for v in trajectories(values)]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def transition_pairs(values, steps=1):
    parts = trajectories(values)
    if any(len(v) <= steps for v in parts):
        raise ValueError("A trajectory is too short for transition learning")
    x = np.concatenate([v[:-steps] for v in parts])
    y = np.concatenate(
        [np.stack([v[i : i + steps] for i in range(1, len(v) - steps + 1)]) for v in parts]
    )
    return x, y[:, 0] if steps == 1 else y


def corrupt(values, seed, noise=0.0, missing=0.0, outliers=0.0, bias=0.0, outage=0.0):
    if noise < 0 or not 0 <= missing < 1 or not 0 <= outliers <= 1:
        raise ValueError("Invalid corruption settings")
    rng = np.random.default_rng(seed)
    x = np.asarray(values, float).copy()
    x += rng.normal(0, noise, x.shape) + bias
    mask = rng.random(x.shape) < outliers
    x[mask] += rng.normal(0, max(noise, 1.0) * 10, mask.sum())
    x[rng.random(x.shape) < missing] = np.nan
    if not 0 <= outage <= 1:
        raise ValueError("Sensor outage fraction must be in [0,1]")
    if outage:
        length = max(1, int(len(x) * outage))
        start = int(rng.integers(0, len(x) - length + 1))
        x[start : start + length] = np.nan
    return x
