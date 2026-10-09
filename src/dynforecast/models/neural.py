import torch
from torch import nn
from dynforecast.data.pipeline import training_windows, trajectories
from dynforecast.training import train_network
from .base import Forecaster


class TemporalNetwork(nn.Module):
    def __init__(self, kind, features, history, horizon, outputs, width, targets=None):
        super().__init__()
        self.kind, self.horizon, self.outputs = kind, horizon, outputs
        self.targets = list(range(outputs)) if targets is None else targets
        if kind in ("nhits", "patchtst"):
            from .advanced import NHITSNetwork, PatchTSTNetwork

            cls = NHITSNetwork if kind == "nhits" else PatchTSTNetwork
            self.body = cls(features, history, horizon, self.targets, width)
            return
        if kind == "mlp":
            self.body = nn.Sequential(nn.Flatten(), nn.Linear(features * history, width), nn.Tanh())
            self.linear_skip = nn.Linear(features * history, horizon * outputs)
        elif kind in ("lstm", "gru"):
            self.body = (nn.LSTM if kind == "lstm" else nn.GRU)(features, width, batch_first=True)
        elif kind == "tcn":
            # Explicit left padding keeps convolutions causal, including the dilated layer.
            self.body = nn.Sequential(
                nn.ConstantPad1d((2, 0), 0),
                nn.Conv1d(features, width, 3),
                nn.ReLU(),
                nn.ConstantPad1d((4, 0), 0),
                nn.Conv1d(width, width, 3, dilation=2),
                nn.ReLU(),
            )
        elif kind == "transformer":
            if width % 2:
                raise ValueError("Transformer width must be divisible by two")
            self.embedding = nn.Linear(features, width)
            self.position = nn.Parameter(torch.randn(1, history, width) * 0.01)
            self.body = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(width, 2, width * 2, dropout=0.0, batch_first=True), 2
            )
        elif kind == "dlinear":
            self.trend = nn.Linear(history, horizon)
            self.seasonal = nn.Linear(history, horizon)
            self.output = nn.Linear(features, outputs)
            return
        else:
            raise ValueError(f"Unknown temporal architecture: {kind}")
        self.head = nn.Linear(width, horizon * outputs)

    def forward(self, x):
        if self.kind in ("nhits", "patchtst"):
            return self.body(x)
        baseline = None
        if self.kind == "mlp":
            # Last-value centering makes the residual predictor invariant to level shifts.
            baseline = x[:, -1:, self.targets]
            x = x - x[:, -1:, :]
        if self.kind in ("lstm", "gru"):
            z = self.body(x)[0][:, -1]
        elif self.kind == "tcn":
            z = self.body(x.transpose(1, 2))[:, :, -1]
        elif self.kind == "transformer":
            mask = torch.triu(
                torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool), 1
            )
            z = self.body(self.embedding(x) + self.position, mask=mask)[:, -1]
        elif self.kind == "dlinear":
            channels = x.transpose(1, 2)
            smooth = torch.nn.functional.avg_pool1d(
                torch.nn.functional.pad(channels, (2, 2), mode="replicate"), 5, stride=1
            )
            return self.output(
                (self.trend(smooth) + self.seasonal(channels - smooth)).transpose(1, 2)
            )
        else:
            z = self.body(x)
        output = self.head(z)
        if self.kind == "mlp":
            output = output + self.linear_skip(x.flatten(1))
        output = output.reshape(-1, self.horizon, self.outputs)
        return output if baseline is None else output + baseline


class NeuralForecaster(Forecaster):
    def __init__(self, kind="lstm", width=32, **training):
        self.kind, self.width, self.training = kind, width, training

    def fit(self, train, validation, targets, history, horizon, dt, **kwargs):
        x, y = training_windows(train, history, horizon, targets)
        vx, vy = training_windows(validation, history, horizon, targets)
        self.network = TemporalNetwork(
            self.kind,
            trajectories(train)[0].shape[1],
            history,
            horizon,
            len(targets),
            self.width,
            targets,
        )
        self.diagnostics = train_network(self.network, x, y, vx, vy, **self.training, **kwargs)
        self.parameters = sum(p.numel() for p in self.network.parameters())
        return self

    def predict(self, histories):
        device = next(self.network.parameters()).device
        with torch.no_grad():
            return (
                self.network(torch.as_tensor(histories, dtype=torch.float32, device=device))
                .cpu()
                .numpy()
            )
