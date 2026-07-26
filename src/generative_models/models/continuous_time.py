from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, *, max_period: float = 10_000.0) -> None:
        super().__init__()
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least 2")
        if max_period <= 0:
            raise ValueError("max_period must be positive")
        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(self, times: Tensor) -> Tensor:
        if times.ndim != 1:
            raise ValueError(f"times must have shape [batch], got {tuple(times.shape)}")
        if not torch.is_floating_point(times):
            raise ValueError("times must be floating point")
        if torch.any(times < 0) or torch.any(times > 1):
            raise ValueError("times must be in [0, 1]")

        half_dim = self.embedding_dim // 2
        exponents = torch.arange(half_dim, device=times.device, dtype=times.dtype)
        exponents = exponents / half_dim
        frequencies = torch.exp(-math.log(self.max_period) * exponents)

        angles = times[:, None] * frequencies[None, :] * (2.0 * math.pi)
        embeddings = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)

        if embeddings.shape[1] < self.embedding_dim:
            padding = torch.zeros(
                times.shape[0], 1, device=times.device, dtype=times.dtype
            )
            embeddings = torch.cat([embeddings, padding], dim=1)

        return embeddings
