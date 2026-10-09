"""History-inferred latent dynamics trained solely against observed future measurements."""

import torch
from torch import nn
from torchdiffeq import odeint
from dynforecast.data.pipeline import training_windows, trajectories
from dynforecast.training import train_network
from .neural import NeuralForecaster


class LatentDynamicsNetwork(nn.Module):
    """GRU encoder to a latent initial state, a neural ODE in latent space, and a linear decoder."""

    def __init__(self, features, targets, horizon, dt, width, latent_size):
        super().__init__()
        self.encoder = nn.GRU(features, width, batch_first=True)
        self.initial = nn.Linear(width, latent_size)
        self.field = nn.Sequential(
            nn.Linear(latent_size, width), nn.Tanh(), nn.Linear(width, latent_size)
        )
        self.decoder = nn.Linear(latent_size, len(targets))
        self.horizon, self.dt = horizon, dt

    def rhs(self, time, state):
        return self.field(state)

    def forward(self, x):
        state = self.initial(self.encoder(x)[0][:, -1])
        time = torch.arange(self.horizon + 1, dtype=x.dtype, device=x.device) * self.dt
        trajectory = odeint(self.rhs, state, time, method="rk4")[1:].transpose(0, 1)
        return self.decoder(trajectory)


class LatentODEForecaster(NeuralForecaster):
    """Latent ODE trained only against observed future values."""

    rollout = "latent continuous"

    def __init__(self, width=32, latent_size=4, **training):
        super().__init__("latent_ode", width, **training)
        self.latent_size = latent_size

    def fit(self, train, validation, targets, history, horizon, dt, **kwargs):
        x, y = training_windows(train, history, horizon, targets)
        vx, vy = training_windows(validation, history, horizon, targets)
        self.network = LatentDynamicsNetwork(
            trajectories(train)[0].shape[1], targets, horizon, dt, self.width, self.latent_size
        )
        self.diagnostics = train_network(self.network, x, y, vx, vy, **self.training, **kwargs)
        self.diagnostics["latent_size"] = self.latent_size
        self.diagnostics["claim"] = (
            "predictive latent coordinates; not identified physical hidden states"
        )
        self.parameters = sum(p.numel() for p in self.network.parameters())
        return self
