from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DiffusionSchedule:
    betas: Tensor
    alphas: Tensor
    alpha_bars: Tensor
    sqrt_alpha_bars: Tensor
    sqrt_one_minus_alpha_bars: Tensor

    @property
    def num_timesteps(self) -> int:
        return int(self.betas.shape[0])


def linear_beta_schedule(
    num_timesteps: int,
    *,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")
    if not 0 < beta_start < 1:
        raise ValueError("beta_start must be in (0, 1)")
    if not 0 < beta_end < 1:
        raise ValueError("beta_end must be in (0, 1)")
    if beta_start > beta_end:
        raise ValueError("beta_start must be less than or equal to beta_end")

    return torch.linspace(
        beta_start, beta_end, num_timesteps, device=device, dtype=dtype
    )


def cosine_beta_schedule(
    num_timesteps: int,
    *,
    s: float = 0.008,
    max_beta: float = 0.999,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")
    if s < 0:
        raise ValueError("s must be non-negative")
    if not 0 < max_beta < 1:
        raise ValueError("max_beta must be in (0, 1)")

    steps = torch.linspace(
        0,
        num_timesteps,
        num_timesteps + 1,
        device=device,
        dtype=dtype,
    )
    t = steps / num_timesteps
    alpha_bars = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]

    betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
    return betas.clamp(min=1e-8, max=max_beta)


def build_diffusion_schedule(betas: Tensor) -> DiffusionSchedule:
    if betas.ndim != 1:
        raise ValueError(
            f"betas must have shape [num_timesteps], got {tuple(betas.shape)}"
        )
    if betas.numel() == 0:
        raise ValueError("betas must not be empty")
    if torch.any(betas <= 0) or torch.any(betas >= 1):
        raise ValueError("all betas must be in (0, 1)")

    betas = betas.to(dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=torch.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - alpha_bars),
    )
