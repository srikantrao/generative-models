from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from generative_models.diffusion.forward import q_sample, sample_timesteps
from generative_models.diffusion.samplers import (
    DDPMSample,
    ddpm_reverse_step,
)
from generative_models.diffusion.schedules import DiffusionSchedule

ConditionalNoisePredictor = Callable[[Tensor, Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class ConditionalEpsilonLoss:
    loss: Tensor
    predicted_noise: Tensor
    target_noise: Tensor
    xt: Tensor
    timesteps: Tensor
    model_times: Tensor


def normalize_ddpm_timesteps(
    timesteps: Tensor,
    *,
    num_timesteps: int,
    dtype: torch.dtype,
) -> Tensor:
    """Map integer schedule indices to the U-Net's [0, 1] time input."""
    return timesteps.to(dtype=dtype) / (num_timesteps - 1)


def conditional_epsilon_loss(
    model: ConditionalNoisePredictor,
    schedule: DiffusionSchedule,
    images: Tensor,
    labels: Tensor,
    *,
    timesteps: Tensor | None = None,
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> ConditionalEpsilonLoss:
    if timesteps is None:
        timesteps = sample_timesteps(
            images.shape[0],
            schedule.num_timesteps,
            generator=generator,
            device=images.device,
        )

    forward = q_sample(
        schedule,
        images,
        timesteps,
        noise=noise,
        generator=generator,
    )
    model_times = normalize_ddpm_timesteps(
        forward.timesteps,
        num_timesteps=schedule.num_timesteps,
        dtype=images.dtype,
    )
    predicted_noise = model(forward.xt, model_times, labels)
    loss = (predicted_noise - forward.noise).square().mean()

    return ConditionalEpsilonLoss(
        loss=loss,
        predicted_noise=predicted_noise,
        target_noise=forward.noise,
        xt=forward.xt,
        timesteps=forward.timesteps,
        model_times=model_times,
    )


@torch.no_grad()
def sample_conditional_ddpm(
    model: ConditionalNoisePredictor,
    schedule: DiffusionSchedule,
    labels: Tensor,
    initial_noise: Tensor,
    *,
    ancestral_generator: torch.Generator | None = None,
    return_trajectory: bool = False,
) -> DDPMSample:
    if isinstance(model, nn.Module):
        model.eval()
    state = initial_noise
    labels = labels.to(device=state.device)
    trajectory = [state.detach().cpu()] if return_trajectory else None

    def predict_noise(current: Tensor, timesteps: Tensor) -> Tensor:
        model_times = normalize_ddpm_timesteps(
            timesteps,
            num_timesteps=schedule.num_timesteps,
            dtype=current.dtype,
        )
        return model(current, model_times, labels)

    for timestep in reversed(range(schedule.num_timesteps)):
        timesteps = torch.full(
            (state.shape[0],),
            timestep,
            device=state.device,
            dtype=torch.long,
        )
        state = ddpm_reverse_step(
            predict_noise,
            schedule,
            state,
            timesteps,
            generator=ancestral_generator,
        ).x_previous
        if trajectory is not None:
            trajectory.append(state.detach().cpu())

    return DDPMSample(
        samples=state,
        trajectory=torch.stack(trajectory) if trajectory is not None else None,
    )
