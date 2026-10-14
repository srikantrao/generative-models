from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from generative_models.flow_matching.ode_samplers import (
    FlowODESample,
    ODESolver,
    euler_step,
    heun_step,
)

from generative_models.flow_matching.paths import sample_linear_flow_path

ConditionalVelocityPredictor = Callable[[Tensor, Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class ConditionalFlowMatchingLoss:
    loss: Tensor
    predicted_velocity: Tensor
    target_velocity: Tensor
    xt: Tensor
    times: Tensor
    source_noise: Tensor


def conditional_flow_matching_loss(
    model: ConditionalVelocityPredictor,
    data: Tensor,
    labels: Tensor,
    *,
    times: Tensor | None = None,
    source_noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> ConditionalFlowMatchingLoss:
    path = sample_linear_flow_path(
        data, times=times, source_noise=source_noise, generator=generator
    )
    predicted_velocity = model(path.xt, path.times, labels)
    loss = (predicted_velocity - path.target_velocity).square().mean()

    return ConditionalFlowMatchingLoss(
        loss=loss,
        predicted_velocity=predicted_velocity,
        target_velocity=path.target_velocity,
        xt=path.xt,
        times=path.times,
        source_noise=path.source_noise,
    )


@torch.no_grad()
def sample_conditional_flow_ode(
    model: ConditionalVelocityPredictor,
    labels: Tensor,
    source_noise: Tensor,
    *,
    num_steps: int = 100,
    solver: ODESolver = "heun",
    return_trajectory: bool = False,
) -> FlowODESample:
    if isinstance(model, nn.Module):
        model.eval()

    state = source_noise
    labels = labels.to(device=state.device)
    time_grid = torch.linspace(
        0.0,
        1.0,
        num_steps + 1,
        device=state.device,
        dtype=state.dtype,
    )
    trajectory = [state.detach().cpu()] if return_trajectory else None
    step_function = euler_step if solver == "euler" else heun_step

    def predict_velocity(current: Tensor, times: Tensor) -> Tensor:
        return model(current, times, labels)

    for step_index in range(num_steps):
        times = time_grid[step_index].expand(state.shape[0])
        next_times = time_grid[step_index + 1].expand(state.shape[0])
        state = step_function(
            model=predict_velocity, state=state, times=times, next_times=next_times
        )
        if trajectory is not None:
            trajectory.append(state.detach().cpu())

    return FlowODESample(
        samples=state,
        times=time_grid.detach().cpu(),
        trajectory=torch.stack(trajectory) if trajectory is not None else None,
    )
