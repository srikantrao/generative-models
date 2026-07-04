from __future__ import annotations

import torch
from torch import Tensor, nn

from generative_models.models.time_embeddings import SinusoidalTimeEmbedding


class TimeConditionedMLPDenoiser(nn.Module):
    def __init__(
        self,
        *,
        data_dim: int = 2,
        time_embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        if data_dim <= 0:
            raise ValueError("data_dim must be positive")
        if time_embedding_dim < 2:
            raise ValueError("time_embedding_dim must be at least 2")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")

        self.data_dim: int = data_dim
        self.time_embedding: SinusoidalTimeEmbedding = SinusoidalTimeEmbedding(
            time_embedding_dim
        )

        layers: list[nn.Module] = []
        input_dim = data_dim + time_embedding_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.SiLU())
            # the output dim of the current layer becomes input dim of the next layer.
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, data_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, xt: Tensor, timesteps: Tensor) -> Tensor:
        if xt.ndim != 2:
            raise ValueError(
                f"xt must have shape [batch, data_dim], got {tuple(xt.shape)}"
            )
        if xt.shape[1] != self.data_dim:
            raise ValueError(
                f"xt must have data_dim {self.data_dim}, got {xt.shape[1]}"
            )
        if timesteps.ndim != 1:
            raise ValueError(
                f"timesteps must have batch [batch], got {tuple(timesteps.shape)}"
            )
        if timesteps.dtype != torch.long:
            raise ValueError("timesteps must have dtype torch.long")
        if timesteps.shape[0] != xt.shape[0]:
            raise ValueError(
                f"timesteps batch size {timesteps.shape[0]} does not match xt batch size {xt.shape[0]}"
            )

        timesteps = timesteps.to(device=xt.device)
        time_features = self.time_embedding(timesteps).to(dtype=xt.dtype)
        features = torch.cat([xt, time_features], dim=1)
        return self.network(features)
