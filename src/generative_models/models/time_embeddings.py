from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at leaset 2")
        if max_period <= 0:
            raise ValueError("max_period must be positive")

        self.embedding_dim: int = embedding_dim
        self.max_period: float = max_period

    def forward(self, timesteps: Tensor) -> Tensor:
        if timesteps.ndim != 1:
            raise ValueError(
                f"timesteps must have shape [batch], got {tuple(timesteps.shape)}"
            )
        if timesteps.dtype != torch.long:
            raise ValueError("timesteps must have dtype torch.long")

        half_dim = self.embedding_dim // 2
        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            / half_dim
        )

        arguments = timesteps.to(dtype=torch.float32).unsqueeze(1) * frequencies
        embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=1)

        # if embedding_dim is odd, then you pad with zeros in the last position.
        if self.embedding_dim % 2 == 1:
            padding = torch.zeros(
                timesteps.shape[0],
                1,
                device=timesteps.device,
                dtype=embedding.dtype,
            )
            embedding = torch.cat([embedding, padding], dim=1)

        return embedding
