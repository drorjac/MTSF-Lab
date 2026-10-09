import numpy as np
from scipy.integrate import solve_ivp
from .base import Forecaster
from dynforecast.data.pipeline import trajectories


class MechanisticOscillator(Forecaster):
    """Declared oscillator prior. It does not read simulator parameters or derivatives."""

    rollout = "continuous"

    def __init__(self, stiffness=1.0, damping=0.1):
        self.stiffness, self.damping = stiffness, damping

    def fit(
        self, train, validation, targets, history, horizon, dt, mean=None, scale=None, **kwargs
    ):
        if trajectories(train)[0].shape[1] != 2:
            raise ValueError("Mechanistic oscillator requires observed position and velocity")
        self.targets, self.horizon, self.dt = targets, horizon, dt
        self.mean = np.zeros(2) if mean is None else mean
        self.scale = np.ones(2) if scale is None else scale
        self.diagnostics = {
            "prior": {"stiffness": self.stiffness, "damping": self.damping},
            "oracle": False,
            "coordinate_contract": "position, velocity",
        }
        return self

    def predict(self, histories):
        times = np.arange(1, self.horizon + 1) * self.dt
        predictions = []
        evaluations = 0
        for history in histories:
            initial = history[-1] * self.scale + self.mean
            sol = solve_ivp(
                lambda t, x: [x[1], -self.stiffness * x[0] - self.damping * x[1]],
                (0.0, times[-1]),
                initial,
                t_eval=times,
                rtol=1e-7,
                atol=1e-9,
            )
            if not sol.success:
                raise FloatingPointError(sol.message)
            predictions.append(((sol.y.T - self.mean) / self.scale)[:, self.targets])
            evaluations += sol.nfev
        self.diagnostics["solver_evaluations"] = evaluations
        return np.stack(predictions)
