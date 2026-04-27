"""Pixel encoder and decoder for 64x64 RGB observations.

Mirrors the Dreamer small-image encoder: 4 stride-2 conv layers (no padding),
producing 256x2x2 = 1024 features after flatten. The decoder mirrors this
structure with ConvTranspose2d.
"""

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, embed_dim: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        return h.flatten(start_dim=1)


class Decoder(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.fc = nn.Linear(feature_dim, 1024)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        h = self.fc(feature)
        h = h.view(-1, 1024, 1, 1)
        return self.deconv(h)
