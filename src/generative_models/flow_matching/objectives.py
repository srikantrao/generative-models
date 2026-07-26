from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from generative_models.flow_matching.paths import sample_linear_flow_path


@dataclass(frozen=True)
class FlowMatchingLoss:
    loss: Tensor
    per_example_loss: Tensor
    predicted_velocity: Tensor
    target_velocity: Tensor
    xt: Tensor
    times: Tensor
    source_noise: Tensor
    data: Tensor


VelocityPredictor = Callable[[Tensor, Tensor], Tensor]


def flow_matching_loss(
    model: VelocityPredictor,
    data: Tensor,
    *,
    times: Tensor | None = None,
    source_noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> FlowMatchingLoss:

    path_sample = sample_linear_flow_path(
        data, times=times, source_noise=source_noise, generator=generator
    )

    predicted_velocity = model(path_sample.xt, path_sample.times)
    if predicted_velocity.shape != data.shape:
        raise ValueError(
            f"predicted_velocity must have shape {tuple(data.shape)}, got {tuple(predicted_velocity.shape)}"
        )
    if predicted_velocity.device != data.device:
        raise ValueError("predicted_velocity must be on the same device as data")
    if predicted_velocity.dtype != data.dtype:
        raise ValueError("predicted_velocity must have the same dtype as data")

    squared_error = (predicted_velocity - path_sample.target_velocity).square()
    per_example_loss = squared_error.flatten(start_dim=1).mean(dim=1)
    loss = per_example_loss.mean()

    return FlowMatchingLoss(
        loss=loss,
        per_example_loss=per_example_loss,
        predicted_velocity=predicted_velocity,
        target_velocity=path_sample.target_velocity,
        xt=path_sample.xt,
        times=path_sample.times,
        source_noise=path_sample.source_noise,
        data=path_sample.data,
    )
