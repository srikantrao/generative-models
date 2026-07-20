from __future__ import annotations

import torch
from torch import Tensor, nn

from generative_models.models.continuous_time import ContinuousTimeEmbedding

class TimeConditionedMLPVectorField(nn.Module):
    def __init__(
        self,
        *,
        data_dim: int = 2,
        time_embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2
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

        self.data_dim = data_dim
        self.time_embedding = ContinuousTimeEmbedding(time_embedding_dim)

        layers: list[nn.Module] = []
        input_dim = data_dim + time_embedding_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.SiLU())
            input_dim = hidden_dim
        # final linear layer
        layers.append(nn.Linear(input_dim, data_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, xt: Tensor, times: Tensor) -> Tensor:
        if xt.ndim != 2:
            raise ValueError(
                f"xt must have shape [batch, data_dim], got {tuple(xt.shape)}"
            )
        if xt.shape[1] != self.data_dim:
            raise ValueError(
                f"xt must have data_dim {self.data_dim}, got {xt.shape[1]}"
            )
        if not torch.is_floating_point(xt):
            raise ValueError("xt must be floating point")
        if times.ndim != 1:
            raise ValueError(f"times must have shape [batch], got {tuple(times.shape)}")
        if times.shape[0] != xt.shape[0]:
            raise ValueError(
                f"times batch size {times.shape[0]} does not match xt batch size {xt.shape[0]}"
            )

        times = times.to(device=xt.device)
        time_features = self.time_embedding(times).to(dtype=xt.dtype)
        features = torch.cat([xt, time_features], dim=1)
        return self.network(features)
