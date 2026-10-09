"""Transparent multiscale and patch-based temporal architectures."""

import torch
from torch import nn
from torch.nn import functional as F


class NHITSNetwork(nn.Module):
    """Hierarchical pooled backcast/forecast blocks with interpolated forecast knots.

    A compact N-HiTS-style multivariate implementation, not a claim of byte-for-byte
    equivalence to Nixtla's canonical implementation (available as nf_nhits).
    """

    def __init__(self, features, history, horizon, targets, width):
        super().__init__()
        self.history, self.horizon, self.targets = history, horizon, targets
        self.blocks = nn.ModuleList()
        self.pool_sizes = [max(1, history // 8), max(1, history // 4), history]
        self.knots = [max(1, horizon // 4), max(1, horizon // 2), horizon]
        for size, knots in zip(self.pool_sizes, self.knots):
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(features * size, width),
                    nn.ReLU(),
                    nn.Linear(width, width),
                    nn.ReLU(),
                    nn.Linear(width, history * features + knots * len(targets)),
                )
            )
        self.features = features

    def forward(self, x):
        level = x[:, -1:, self.targets]
        residual = (x - x[:, -1:, :]).transpose(1, 2)
        forecast = torch.zeros(len(x), len(self.targets), self.horizon, device=x.device)
        for block, size, knots in zip(self.blocks, self.pool_sizes, self.knots):
            pooled = F.adaptive_avg_pool1d(residual, size).flatten(1)
            out = block(pooled)
            backcast = out[:, : self.history * self.features].reshape_as(residual)
            coefficients = out[:, self.history * self.features :].reshape(
                len(x), len(self.targets), knots
            )
            forecast = forecast + F.interpolate(
                coefficients, size=self.horizon, mode="linear", align_corners=False
            )
            residual = residual - backcast
        return forecast.transpose(1, 2) + level


class PatchTSTNetwork(nn.Module):
    """Shared channel-wise patch encoder followed by an explicit cross-channel target head."""

    def __init__(self, features, history, horizon, targets, width):
        super().__init__()
        if width % 2:
            raise ValueError("PatchTST width must be even")
        self.patch = min(8, history)
        self.stride = max(1, self.patch // 2)
        count = (history - self.patch) // self.stride + 1
        self.embedding = nn.Linear(self.patch, width)
        self.position = nn.Parameter(torch.randn(1, count, width) * 0.01)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, 2, width * 2, dropout=0.0, batch_first=True),
            2,
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(features * width, horizon * len(targets))
        self.horizon, self.targets, self.features = horizon, targets, features

    def forward(self, x):
        center = x[:, -1:, :]
        patches = (x - center).transpose(1, 2).unfold(-1, self.patch, self.stride)
        patches = patches.reshape(len(x) * self.features, patches.shape[2], self.patch)
        encoded = self.encoder(self.embedding(patches) + self.position).mean(dim=1)
        output = self.head(encoded.reshape(len(x), -1)).reshape(
            len(x), self.horizon, len(self.targets)
        )
        return output + center[:, :, self.targets]
