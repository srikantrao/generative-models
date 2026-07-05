from __future__ import annotations

from dataclasses import dataclass

import torch

from generative_models.data.toy import sample_labeled_gaussian_mixture
from generative_models.diffusion.objectives import epsilon_prediction_loss
from generative_models.diffusion.schedules import (
    DiffusionSchedule,
    build_diffusion_schedule,
    linear_beta_schedule,
)
from generative_models.models.mlp import TimeConditionedMLPDenoiser


@dataclass(frozen=True)
class ToyDDPMTrainingConfig:
    num_steps: int = 1_000
    batch_size: int = 512
    learning_rate: float = 1e-3
    num_timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    hidden_dim: int = 128
    time_embedding_dim: int = 64
    num_hidden_layers: int = 2
    seed: int = 42

    def validate(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")
        if not 0 < self.beta_start < 1:
            raise ValueError("beta_start must be in (0, 1)")
        if not 0 < self.beta_end < 1:
            raise ValueError("beta_end must be in (0, 1)")
        if self.beta_start > self.beta_end:
            raise ValueError("beta_start must be less than or equal to beta_end")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.time_embedding_dim < 2:
            raise ValueError("time_embedding_dim must be at least 2")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")


@dataclass
class ToyDDPMTrainingResult:
    config: ToyDDPMTrainingConfig
    model: TimeConditionedMLPDenoiser
    schedule: DiffusionSchedule
    losses: list[float]


def train_toy_ddpm(
    config: ToyDDPMTrainingConfig,
    *,
    device: torch.device | str = "cpu",
) -> ToyDDPMTrainingResult:
    config.validate()

    train_device = torch.device(device)
    torch.manual_seed(config.seed)
    generator = torch.Generator(device=train_device).manual_seed(config.seed)

    schedule = build_diffusion_schedule(
        linear_beta_schedule(
            config.num_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            device=train_device,
        )
    )

    model = TimeConditionedMLPDenoiser(
        data_dim=2,
        time_embedding_dim=config.time_embedding_dim,
        hidden_dim=config.hidden_dim,
        num_hidden_layers=config.num_hidden_layers,
    ).to(train_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    losses: list[float] = []
    model.train()

    for _ in range(config.num_steps):
        batch = sample_labeled_gaussian_mixture(
            config.batch_size, generator=generator, device=train_device
        )

        optimizer.zero_grad(set_to_none=True)
        loss_output = epsilon_prediction_loss(
            model, schedule, batch.x, generator=generator
        )
        loss_output.loss.backward()
        optimizer.step()

        losses.append(float(loss_output.loss.detach().cpu()))

    return ToyDDPMTrainingResult(
        config=config, model=model, schedule=schedule, losses=losses
    )
