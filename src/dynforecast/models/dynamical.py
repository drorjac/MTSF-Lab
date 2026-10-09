"""Observation-space autonomous dynamics: no simulator derivatives or equations are read."""

from itertools import combinations_with_replacement
import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch import nn
from torchdiffeq import odeint
from dynforecast.training import train_network
from dynforecast.data.pipeline import trajectories, transition_pairs
from .base import Forecaster


class PolynomialLibrary:
    def __init__(self, dimensions, degree=2):
        self.terms = [()] + [
            c
            for d in range(1, degree + 1)
            for c in combinations_with_replacement(range(dimensions), d)
        ]

    def transform(self, x):
        return np.stack(
            [np.prod(x[..., list(t)], axis=-1) if t else np.ones(x.shape[:-1]) for t in self.terms],
            axis=-1,
        )

    @property
    def names(self):
        return ["*".join(f"x{i}" for i in t) if t else "1" for t in self.terms]


class SINDy(Forecaster):
    rollout = "continuous"

    def __init__(
        self, degree=2, threshold=0.05, iterations=10, derivative="finite", smoothing_window=9
    ):
        self.degree, self.threshold, self.iterations = degree, threshold, iterations
        self.derivative, self.smoothing_window = derivative, smoothing_window

    def fit(self, train, validation, targets, history, horizon, dt, **kwargs):
        self.targets, self.horizon, self.dt = targets, horizon, dt
        parts = trajectories(train)
        self.library = PolynomialLibrary(parts[0].shape[1], self.degree)
        values, derivatives = [], []
        for part in parts:
            if self.derivative == "savgol":
                from scipy.signal import savgol_filter

                width = min(self.smoothing_window, len(part) // 2 * 2 - 1)
                if width < 5 or width % 2 == 0:
                    raise ValueError("Savitzky-Golay window must be odd and at least five")
                smooth = savgol_filter(part, width, 3, axis=0)
                derivative = savgol_filter(part, width, 3, deriv=1, delta=dt, axis=0)
                edge = width // 2
                values.append(smooth[edge:-edge])
                derivatives.append(derivative[edge:-edge])
            elif self.derivative == "finite":
                values.append(part[1:-1])
                derivatives.append(np.gradient(part, dt, axis=0, edge_order=2)[1:-1])
            else:
                raise ValueError("Derivative method must be finite or savgol")
        theta = self.library.transform(np.concatenate(values))
        derivative = np.concatenate(derivatives)
        coef = np.linalg.lstsq(theta, derivative, rcond=None)[0]
        for _ in range(self.iterations):
            coef[np.abs(coef) < self.threshold] = 0.0
            for j in range(parts[0].shape[1]):
                active = coef[:, j] != 0
                if active.any():
                    coef[active, j] = np.linalg.lstsq(
                        theta[:, active], derivative[:, j], rcond=None
                    )[0]
        self.coefficients = coef
        self.parameters = int(np.count_nonzero(coef))
        self.diagnostics = {
            "terms": self.library.names,
            "coefficients": coef.tolist(),
            "coordinate_system": "training-standardized observed coordinates",
            "derivative_estimator": self.derivative,
            "training_derivative_rmse": float(np.sqrt(np.mean((theta @ coef - derivative) ** 2))),
        }
        return self

    def predict(self, histories):
        times = np.arange(1, self.horizon + 1) * self.dt
        predictions, evaluations = [], 0
        for h in histories:
            sol = solve_ivp(
                lambda t, x: self.library.transform(x) @ self.coefficients,
                (0.0, times[-1]),
                h[-1],
                t_eval=times,
                rtol=1e-6,
                atol=1e-8,
            )
            if not sol.success or sol.y.shape[1] != self.horizon:
                raise FloatingPointError(f"SINDy rollout failed: {sol.message}")
            predictions.append(sol.y.T[:, self.targets])
            evaluations += sol.nfev
        self.diagnostics["solver_evaluations"] = evaluations
        return np.stack(predictions)


class VectorNetwork(nn.Module):
    def __init__(
        self,
        dimensions,
        width,
        prior=None,
        mean=None,
        scale=None,
        train_prior=False,
        residual=True,
        polynomial=None,
    ):
        super().__init__()
        self.net = (
            nn.Sequential(nn.Linear(dimensions, width), nn.Tanh(), nn.Linear(width, dimensions))
            if residual
            else nn.Identity()
        )
        self.prior = prior
        self.residual = residual
        self.train_prior = train_prior
        if train_prior:
            self.log_prior = nn.Parameter(
                torch.log(
                    torch.tensor(
                        [max(prior["stiffness"], 1e-6), max(prior["damping"], 1e-6)],
                        dtype=torch.float32,
                    )
                )
            )
        self.polynomial = polynomial
        if polynomial is not None:
            self.register_buffer(
                "coefficients", torch.tensor(polynomial.coefficients, dtype=torch.float32)
            )
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.evaluations = 0

    def forward(self, x):
        result = self.net(x) if self.residual else torch.zeros_like(x)
        if self.polynomial is not None:
            theta = torch.stack(
                [
                    torch.prod(x[..., list(term)], dim=-1) if term else torch.ones_like(x[..., 0])
                    for term in self.polynomial.library.terms
                ],
                dim=-1,
            )
            result = result + theta @ self.coefficients
        if self.prior is not None:
            # Prior declared by experiment, never obtained from simulator truth.
            physical = x * self.scale + self.mean
            stiffness, damping = (
                self.log_prior.exp()
                if self.train_prior
                else (self.prior["stiffness"], self.prior["damping"])
            )
            base = torch.stack(
                (
                    physical[..., 1],
                    -stiffness * physical[..., 0] - damping * physical[..., 1],
                ),
                dim=-1,
            )
            result = result + base / self.scale
        return result

    def rhs(self, time, x):
        self.evaluations += 1
        return self(x)


class NeuralODE(Forecaster):
    rollout = "continuous"

    def __init__(
        self,
        width=32,
        prior=None,
        train_prior=False,
        residual=True,
        sparse_prior=False,
        flow_steps=1,
        **training,
    ):
        self.width, self.prior, self.training = width, prior, training
        self.train_prior, self.residual, self.sparse_prior = train_prior, residual, sparse_prior
        self.flow_steps = flow_steps

    def fit(
        self, train, validation, targets, history, horizon, dt, mean=None, scale=None, **kwargs
    ):
        features = trajectories(train)[0].shape[1]
        if self.prior is not None and features != 2:
            raise ValueError("Oscillator residual prior requires exactly two measured states")
        self.targets, self.horizon, self.dt = targets, horizon, dt
        polynomial = (
            SINDy(degree=3, threshold=0.02).fit(train, validation, targets, history, horizon, dt)
            if self.sparse_prior
            else None
        )
        self.network = VectorNetwork(
            features,
            self.width,
            self.prior,
            np.zeros(features) if mean is None else mean,
            np.ones(features) if scale is None else scale,
            self.train_prior,
            self.residual,
            polynomial,
        )
        # One-step flow matching differentiates through RK4 integration, not latent truth.
        field = self.network

        class Flow(nn.Module):
            def __init__(self):
                super().__init__()
                self.field = field

            def forward(self, x):
                time = torch.arange(self_steps + 1, device=x.device, dtype=x.dtype) * dt
                output = odeint(self.field.rhs, x, time, method="rk4")[1:].transpose(0, 1)
                return output[:, 0] if self_steps == 1 else output

        self_steps = self.flow_steps
        x, y = transition_pairs(train, self_steps)
        vx, vy = transition_pairs(validation, self_steps)

        self.diagnostics = train_network(
            Flow(),
            x,
            y,
            vx,
            vy,
            **self.training,
            **kwargs,
        )
        self.diagnostics["objective"] = "observation-only one-step flow matching"
        self.diagnostics["prior"] = self.prior
        self.diagnostics["flow_steps"] = self_steps
        if self.train_prior:
            self.diagnostics["estimated_prior"] = dict(
                zip(("stiffness", "damping"), self.network.log_prior.exp().detach().cpu().tolist())
            )
        if polynomial is not None:
            self.diagnostics["sparse_prior"] = polynomial.diagnostics
        self.parameters = sum(p.numel() for p in self.network.parameters())
        return self

    def predict(self, histories):
        device = next(self.network.parameters()).device
        initial = torch.as_tensor(histories[:, -1], dtype=torch.float32, device=device)
        times = torch.arange(self.horizon + 1, dtype=torch.float32, device=device) * self.dt
        self.network.evaluations = 0
        with torch.no_grad():
            states = odeint(
                self.network.rhs, initial, times, method="rk4", options={"step_size": self.dt / 2}
            )
        self.diagnostics["solver_evaluations"] = self.network.evaluations
        return states[1:].transpose(0, 1)[:, :, self.targets].cpu().numpy()
