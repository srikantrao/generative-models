from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from generative_models.diffusion.schedules import DiffusionSchedule


@dataclass(frozen=True)
class ForwardProcessSample:
    xt: Tensor
    noise: Tensor
    timesteps: Tensor


def sample_timesteps(
    batch_size: int,
    num_timesteps: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if num_timesteps <= 0:
        raise ValueError("num_timestamps must be positive")

    return torch.randint(
        low=0,
        high=num_timesteps,
        size=(batch_size,),
        generator=generator,
        device=device,
        dtype=torch.long,
    )


def extract_schedule_values(
    values: Tensor, timesteps: Tensor, target_shape: torch.Size | tuple[int, ...]
) -> Tensor:
    if values.ndim != 1:
        raise ValueError(
            f"values must have shape [num_timesteps], got {tuple(values.shape)}"
        )
    if timesteps.ndim != 1:
        raise ValueError(
            f"timesteps must have shape [batch], got {tuple(timesteps.shape)}"
        )
    if timesteps.dtype != torch.long:
        raise ValueError("timesteps must have dtype torch.long")
    if len(target_shape) == 0:
        raise ValueError("target_shape must include a batch dimension")
    if target_shape[0] != timesteps.shape[0]:
        raise ValueError(
            f"target_batch batch_size: {target_shape[0]}, timesteps batch_size: {timesteps.shape[0]}"
        )
    if torch.any(timesteps < 0) or torch.any(timesteps >= values.shape[0]):
        raise ValueError("timesteps must be in [0, num_timesteps)")

    gathered = values.to(device=timesteps.device)[timesteps]
    broadcast_shape = (timesteps.shape[0],) + (1,) * (len(target_shape) - 1)
    return gathered.reshape(broadcast_shape)


def q_sample(
    schedule: DiffusionSchedule,
    x0: Tensor,
    timesteps: Tensor,
    *,
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> ForwardProcessSample:

    timesteps = timesteps.to(device=x0.device)
    if noise is None:
        noise = torch.randn(
            x0.shape,
            generator=generator,
            device=x0.device,
            dtype=x0.dtype,
        )
    sqrt_alpha_bars = extract_schedule_values(
        schedule.sqrt_alpha_bars,
        timesteps,
        x0.shape,
    ).to(dtype=x0.dtype)

    sqrt_one_minus_alpha_bars = extract_schedule_values(
        schedule.sqrt_one_minus_alpha_bars, timesteps, x0.shape
    )

    xt = sqrt_alpha_bars * x0 + sqrt_one_minus_alpha_bars * noise
    return ForwardProcessSample(xt=xt, noise=noise, timesteps=timesteps)
