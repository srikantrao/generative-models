from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from generative_models.diffusion.forward import extract_schedule_values
from generative_models.diffusion.objectives import NoisePredictor
from generative_models.diffusion.schedules import DiffusionSchedule


@dataclass(frozen=True)
class DDPMReverseStep:
    x_previous: Tensor
    mean: Tensor
    predicted_noise: Tensor
    timesteps: Tensor


@dataclass(frozen=True)
class DDPMSample:
    samples: Tensor
    trajectory: Tensor | None


def ddpm_reverse_mean(
    schedule: DiffusionSchedule, xt: Tensor, timesteps: Tensor, predicted_noise: Tensor
) -> Tensor:
    if xt.ndim < 2:
        raise ValueError("xt must have shape [batch, ...]")
    if predicted_noise.shape != xt.shape:
        raise ValueError(
            f"predicted_noise must have shape {tuple(xt.shape)}, got {tuple(predicted_noise.shape)}"
        )
    if predicted_noise.device != xt.device:
        raise ValueError("predicted_noise must be on the same device as xt")
    if predicted_noise.dtype != xt.dtype:
        raise ValueError("predicted_noise must have the same dtype as xt")

    timesteps = timesteps.to(device=xt.device)
    betas = extract_schedule_values(schedule.betas, timesteps, xt.shape).to(
        dtype=xt.dtype
    )
    alphas = extract_schedule_values(schedule.alphas, timesteps, xt.shape).to(
        dtype=xt.dtype
    )
    sqrt_one_minus_alpha_bars = extract_schedule_values(
        schedule.sqrt_one_minus_alpha_bars,
        timesteps,
        xt.shape,
    ).to(dtype=xt.dtype)

    return torch.rsqrt(alphas) * (
        xt - (betas / sqrt_one_minus_alpha_bars) * predicted_noise
    )


def ddpm_reverse_step(
    model: NoisePredictor,
    schedule: DiffusionSchedule,
    xt: Tensor,
    timesteps: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> DDPMReverseStep:
    timesteps = timesteps.to(device=xt.device)
    predicted_noise = model(xt, timesteps)
    mean = ddpm_reverse_mean(schedule, xt, timesteps, predicted_noise)

    betas = extract_schedule_values(schedule.betas, timesteps, xt.shape).to(
        dtype=xt.dtype
    )
    noise = torch.randn(
        xt.shape,
        generator=generator,
        device=xt.device,
        dtype=xt.dtype,
    )
    nonzero_mask = (timesteps > 0).to(dtype=xt.dtype)
    nonzero_mask = nonzero_mask.reshape((xt.shape[0],) + (1,) * (xt.ndim - 1))
    x_previous = mean + nonzero_mask * torch.sqrt(betas) * noise

    return DDPMReverseStep(
        x_previous=x_previous,
        mean=mean,
        predicted_noise=predicted_noise,
        timesteps=timesteps,
    )


@torch.no_grad()
def sample_ddpm(
    model: NoisePredictor,
    schedule: DiffusionSchedule,
    *,
    num_samples: int,
    sample_shape: tuple[int, ...] = (2,),
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    return_trajectory: bool = False,
) -> DDPMSample:
    sample_device = torch.device(device)
    xt = torch.randn(
        (num_samples,) + sample_shape,
        generator=generator,
        device=sample_device,
    )
    trajectory = [xt.detach().cpu()] if return_trajectory else None
    if hasattr(model, "eval"):
        model.eval()

    for timestep in reversed(range(schedule.num_timesteps)):
        timesteps = torch.full(
            (num_samples,),
            timestep,
            device=sample_device,
            dtype=torch.long,
        )
        step = ddpm_reverse_step(
            model,
            schedule,
            xt,
            timesteps,
            generator=generator,
        )
        xt = step.x_previous
        if trajectory is not None:
            trajectory.append(xt.detach().cpu())

    return DDPMSample(
        samples=xt,
        trajectory=torch.stack(trajectory) if trajectory is not None else None,
    )
