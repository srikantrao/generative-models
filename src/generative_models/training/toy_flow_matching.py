from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self


import torch

from generative_models.data.toy import (
    DEFAULT_TOY_GAUSSIAN_MIXTURE_SPEC,
    ToyGaussianMixtureSpec,
    sample_labeled_gaussian_mixture,
)
from generative_models.flow_matching.objectives import flow_matching_loss
from generative_models.models.flow_mlp import TimeConditionedMLPVectorField

@dataclass(frozen=True)
class ToyFlowMatchingTrainingConfig:
    num_steps: int = 1_000
    batch_size: int = 512
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    time_embedding_dim: int = 64
    num_hidden_layers: int = 2
    seed: int = 42
    data_spec: ToyGaussianMixtureSpec = DEFAULT_TOY_GAUSSIAN_MIXTURE_SPEC

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> Self:
        raw_config = dict(values)
        raw_data_spec = raw_config["data_spec"]
        raw_config["data_spec"] = ToyGaussianMixtureSpec.from_mapping(raw_data_spec)

        try:
            config = cls(**raw_config)
        except TypeError as error:
            raise ValueError(f"invalid training config fields: {error}") from error

        config.validate()
        return config

    def validate(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.time_embedding_dim < 2:
            raise ValueError("time_embedding_dim must be at least 2")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")

@dataclass
class ToyFlowMatchingTrainingResult:
    config: ToyFlowMatchingTrainingConfig
    model: TimeConditionedMLPVectorField
    losses: list[float]

def train_toy_flow_matching(
    config: ToyFlowMatchingTrainingConfig,
    *,
    device: torch.device | str = "cpu",
) -> ToyFlowMatchingTrainingResult:

    config.validate()

    train_device = torch.device(device)
    torch.manual_seed(config.seed)
    generator = torch.Generator(device=train_device).manual_seed(config.seed)

    centers, class_probs = config.data_spec.to_tensors(device=train_device)
    model = TimeConditionedMLPVectorField(
        data_dim=2,
        time_embedding_dim=config.time_embedding_dim,
        hidden_dim=config.hidden_dim,
        num_hidden_layers=config.num_hidden_layers,
    ).to(device=train_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    model.train()

    for _ in range(config.num_steps):
        batch = sample_labeled_gaussian_mixture(
            config.batch_size,
            centers=centers,
            class_probs=class_probs,
            std=config.data_spec.std
            generator=generator,
            device=train_device
        )
        optimizer.zero_grad(set_to_none=True)
        loss_output = flow_matching_loss(model, batch.x, generator=generator)
        loss_output.loss.backward()
        optimizer.step()

        losses.append(float(loss_output.loss.detach().cpu()))

    return ToyFlowMatchingTrainingResult(config=config, model=model, losses=losses)
