from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from generative_models.diffusion.forward import q_sample, sample_timesteps
from generative_models.diffusion.schedules import DiffusionSchedule


@dataclass(frozen=True)
class EpsilonPredictionLoss:
    loss: Tensor
    per_example_loss: Tensor
    predicted_noise: Tensor
    target_noise: Tensor
    xt: Tensor
    timesteps: Tensor


NoisePredictor = Callable[[Tensor, Tensor], Tensor]


def epsilon_prediction_loss(
    model: NoisePredictor,
    schedule: DiffusionSchedule,
    x0: Tensor,
    *,
    timesteps: Tensor | None = None,
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> EpsilonPredictionLoss:
    if x0.ndim < 2:
        raise ValueError("x0 must have shape [batch, ...]")

    if timesteps is None:
        timesteps = sample_timesteps(
            batch_size=x0.shape[0],
            num_timesteps=schedule.num_timesteps,
            generator=generator,
            device=x0.device,
        )

    forward_sample = q_sample(
        schedule=schedule, x0=x0, timesteps=timesteps, noise=noise, generator=generator
    )

    predicted_noise = model(forward_sample.xt, forward_sample.timesteps)
    if predicted_noise.shape != x0.shape:
        raise ValueError(
            f"predicted_noise must have shape {tuple(x0.shape)}, got {tuple(predicted_noise.shape)}"
        )
    if predicted_noise.device != x0.device:
        raise ValueError("predicted_noise must be on the same device as x0")
    if predicted_noise.dtype != x0.dtype:
        raise ValueError()

    squared_error = (predicted_noise - forward_sample.noise).square()
    per_example_loss = squared_error.flatten(start_dim=1).mean(dim=1)
    loss = per_example_loss.mean()

    return EpsilonPredictionLoss(
        loss=loss,
        per_example_loss=per_example_loss,
        predicted_noise=predicted_noise,
        target_noise=forward_sample.noise,
        xt=forward_sample.xt,
        timesteps=forward_sample.timesteps,
    )
