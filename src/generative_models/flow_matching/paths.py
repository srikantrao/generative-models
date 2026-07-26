from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LinearFlowMatchingSample:
    xt: Tensor
    times: Tensor
    source_noise: Tensor
    data: Tensor
    target_velocity: Tensor


def sample_times(
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not dtype.is_floating_point:
        raise ValueError("dtype must be floating")
    return torch.rand(batch_size, dtype=dtype, device=device, generator=generator)


def expand_times(times: Tensor, target_shape: torch.Size | tuple[int, ...]) -> Tensor:

    if times.ndim != 1:
        raise ValueError(f"times must have shape [batch], got {tuple(times.shape)}")
    if not times.dtype.is_floating_point:
        raise ValueError("times must have floating point dtype")
    if len(target_shape) == 0:
        raise ValueError("target_shape must include a batch dimension")
    if target_shape[0] != times.shape[0]:
        raise ValueError(
            f"target_shape batch size {target_shape[0]} does not match times batch size {times.shape[0]}"
        )

    if torch.any(times < 0) or torch.any(times > 1):
        raise ValueError("times must be in [0, 1]")

    broadcast_shape = (times.shape[0],) + (1,) * (len(target_shape) - 1)
    return times.reshape(broadcast_shape)


def sample_linear_flow_path(
    data: Tensor,
    *,
    times: Tensor | None = None,
    source_noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> LinearFlowMatchingSample:
    if data.ndim < 2:
        raise ValueError("data must have shape [batch, ...]")

    if times is None:
        times = sample_times(
            data.shape[0], generator=generator, device=data.device, dtype=data.dtype
        )
    else:
        times = times.to(device=data.device)

    if source_noise is None:
        source_noise = torch.randn(
            data.shape,
            generator=generator,
            dtype=data.dtype,
            device=data.device,
        )
    else:
        if source_noise.shape != data.shape:
            raise ValueError(
                f"source_noise must have shape {tuple(data.shape)}, got {tuple(source_noise.shape)}"
            )
        if source_noise.device != data.device:
            raise ValueError("source_noise must be on the same device as data")
        if source_noise.dtype != data.dtype:
            raise ValueError("source_noise must have the same dtype as data")

    expanded_times = expand_times(times, data.shape).to(dtype=data.dtype)
    xt = (1.0 - expanded_times) * source_noise + expanded_times * data
    target_velocity = data - source_noise  # analytical derivative

    return LinearFlowMatchingSample(
        xt=xt,
        times=times,
        source_noise=source_noise,
        data=data,
        target_velocity=target_velocity,
    )
