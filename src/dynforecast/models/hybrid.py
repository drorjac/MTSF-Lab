import numpy as np
import torch
from torch import nn
from dynforecast.data.pipeline import training_windows, trajectories
from dynforecast.training import train_network
from .base import Forecaster
from .neural import TemporalNetwork


class OscillatorRollout(nn.Module):
    """RK4 rollout of a damped oscillator in standardized coordinates."""

    def __init__(self, prior, mean, scale, horizon, dt):
        super().__init__()
        self.prior, self.horizon, self.dt = prior, horizon, dt
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def field(self, x):
        physical = x * self.scale + self.mean
        return (
            torch.stack(
                (
                    physical[..., 1],
                    -self.prior["stiffness"] * physical[..., 0]
                    - self.prior["damping"] * physical[..., 1],
                ),
                dim=-1,
            )
            / self.scale
        )

    def forward(self, history):
        state, output = history[:, -1], []
        for _ in range(self.horizon):
            step = self.dt
            k1 = self.field(state)
            k2 = self.field(state + step * k1 / 2)
            k3 = self.field(state + step * k2 / 2)
            k4 = self.field(state + step * k3)
            state = state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            output.append(state)
        return torch.stack(output, dim=1)


class HybridNetwork(nn.Module):
    """Combine an oscillator rollout with an LSTM forecast.

    ``forecast_residual`` adds them, ``average_hybrid`` averages them, and
    ``adaptive_hybrid`` mixes them with a learned per-step gate."""

    def __init__(self, kind, history, horizon, targets, width, prior, mean, scale, dt):
        super().__init__()
        self.kind, self.targets = kind, targets
        self.physics = OscillatorRollout(prior, mean, scale, horizon, dt)
        self.temporal = TemporalNetwork("lstm", 2, history, horizon, len(targets), width, targets)
        if kind == "adaptive_hybrid":
            self.gate = nn.Sequential(
                nn.Flatten(),
                nn.Linear(history * 2, width),
                nn.Tanh(),
                nn.Linear(width, horizon * len(targets)),
                nn.Sigmoid(),
            )
        self.last_gate_mean = None

    def forward(self, x):
        physics = self.physics(x)[:, :, self.targets]
        neural = self.temporal(x)
        if self.kind == "forecast_residual":
            return physics + neural
        if self.kind == "average_hybrid":
            return (physics + neural) / 2
        gate = self.gate(x).reshape_as(neural)
        self.last_gate_mean = float(gate.detach().mean())
        return gate * physics + (1 - gate) * neural


class PhysicsLoss:
    """Short- and long-horizon MSE plus a penalty for disagreeing with the prior's vector field."""

    def __init__(self, field, dt, short_weight=1.0, long_weight=1.0, dynamics_weight=0.01):
        self.field, self.dt = field, dt
        self.weights = short_weight, long_weight, dynamics_weight

    def __call__(self, prediction, truth, inputs):
        self.field.__self__.to(inputs.device)
        mid = max(1, prediction.shape[1] // 2)
        short = (prediction[:, :mid] - truth[:, :mid]).square().mean()
        long = (
            (prediction[:, mid:] - truth[:, mid:]).square().mean()
            if mid < prediction.shape[1]
            else short
        )
        states = torch.cat([inputs[:, -1:], prediction], dim=1)
        derivative = (states[:, 1:] - states[:, :-1]) / self.dt
        expected = self.field(states[:, :-1])
        return (
            self.weights[0] * short
            + self.weights[1] * long
            + self.weights[2] * (derivative - expected).square().mean()
        )


class HybridForecaster(Forecaster):
    """Oscillator-guided forecasters for two-state (position, velocity) data.

    ``physics_guided`` trains a plain LSTM with :class:`PhysicsLoss`; the other
    kinds train a :class:`HybridNetwork`."""

    def __init__(
        self,
        kind="forecast_residual",
        width=32,
        prior=None,
        dynamics_weight=0.01,
        short_weight=1.0,
        long_weight=1.0,
        **training,
    ):
        self.kind, self.width, self.training = kind, width, training
        self.prior = prior or {"stiffness": 1.0, "damping": 0.1}
        self.weights = short_weight, long_weight, dynamics_weight

    def fit(
        self, train, validation, targets, history, horizon, dt, mean=None, scale=None, **kwargs
    ):
        if trajectories(train)[0].shape[1] != 2:
            raise ValueError("Oscillator-guided models require measured position and velocity")
        mean = np.zeros(2) if mean is None else mean
        scale = np.ones(2) if scale is None else scale
        x, y = training_windows(train, history, horizon, targets)
        vx, vy = training_windows(validation, history, horizon, targets)
        loss = None
        if self.kind == "physics_guided":
            if targets != [0, 1]:
                raise ValueError("Dynamics-consistency loss requires both ordered state outputs")
            self.network = TemporalNetwork("lstm", 2, history, horizon, 2, self.width, targets)
            prior = OscillatorRollout(self.prior, mean, scale, horizon, dt)
            loss = PhysicsLoss(prior.field, dt, *self.weights)
        else:
            self.network = HybridNetwork(
                self.kind, history, horizon, targets, self.width, self.prior, mean, scale, dt
            )
        training_network = self.network.temporal if self.kind == "average_hybrid" else self.network
        self.diagnostics = train_network(
            training_network, x, y, vx, vy, loss_fn=loss, **self.training, **kwargs
        )
        self.diagnostics.update(kind=self.kind, prior=self.prior, loss_weights=self.weights)
        if self.kind == "average_hybrid":
            self.diagnostics["training_policy"] = (
                "fit neural forecaster independently, then average 50/50 with declared physics"
            )
        self.parameters = sum(p.numel() for p in self.network.parameters())
        return self

    def predict(self, histories):
        device = next(self.network.parameters()).device
        with torch.no_grad():
            output = self.network(torch.as_tensor(histories, dtype=torch.float32, device=device))
        if hasattr(self.network, "last_gate_mean"):
            self.diagnostics["mean_physics_gate"] = self.network.last_gate_mean
        return output.cpu().numpy()
