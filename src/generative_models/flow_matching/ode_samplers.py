from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from generative_models.flow_matching.objectives import VelocityPredictor

ODE_SOLVER = Literal["euler", "heun"]

@dataclass(frozen=True)
class FlowODESample:
    samples: Tensor
    times: Tensor
    trajectory: Tensor | None

def _validate_state_and_times(state: Tensor, times: Tensor) -> None:
    if state.ndim < 2:
        raise ValueError("state must have shape [batch,...]")
    if not torch.is_floating_point(state):
        raise ValueError("state must be of type floating point")
    if times.ndim != 1:
        raise ValueError(f"times must have shape [batch], got {tuple(times.shape)}")
    if times.shape[0] != state.shape[0]:
        raise ValueError(f"times batch size {times.shape[0]} does not match state batch size {state.shape[0]}")
    if not torch.is_floating_point(times):
        raise ValueError("times must be floating point")
    if times.device != state.device:
        raise ValueError("times must be on the same device as state")
    if torch.any(times < 0) or torch.any(times > 1):
        raise ValueError("times must be in [0, 1]")

def _validate_velocity(velocity: Tensor, state: Tensor) -> None:
    if velocity.shape != state.shape:
        raise ValueError(
            f"velocity must have shape {tuple(state.shape)}, "
            f"got {tuple(velocity.shape)}"
        )
    if velocity.device != state.device:
        raise ValueError("velocity must be on the same device as state")
    if velocity.dtype != state.dtype:
        raise ValueError("velocity must have the same dtype as state")

def _expand_step_size(
    times: Tensor,
    next_times: Tensor,
    target_shape: torch.Size | tuple[int, ...]
) -> Tensor:
    if next_times.shape != times.shape:
        raise ValueError("next_times must have the same shape as times")
    if next_times.device != times.device:
        raise ValueError("next_times must be on the same device as times")
    if next_times.dtype != times.dtype:
        raise ValueError("next_times must have the same dtype as times")
    if torch.any(next_times < 0) or torch.any(next_times > 1):
        raise ValueError("next_times must be in [0, 1]")
    if torch.any(next_times <= times):
        raise ValueError("next_times must be greater than times")

    step_sizes = next_times - times
    broadcast_shape = (times.shape[0],) + (1,) * (len(target_shape) - 1)
    return step_sizes.reshape(broadcast_shape)

def euler_step(
    model: VelocityPredictor,
    state: Tensor,
    times: Tensor,
    next_times: Tensor,
) -> Tensor:
    _validate_state_and_times(state, times)
    step_sizes = _expand_step_size(times, next_times, state.shape)

    velocity = model(state, times)
    _validate_velocity(velocity, state)
    return state + step_sizes * velocity
