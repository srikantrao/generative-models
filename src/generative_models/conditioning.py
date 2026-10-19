from __future__ import annotations

import torch
from torch import Tensor

def drop_conditions(
    labels: Tensor,
    probability: float,
    *,
    null_label: int,
    generator: torch.Generator,

) -> tuple[Tensor, Tensor]:
    """
    Replace entire class labels, with one independent draw per example.
    """
    if probability == 0.0:
        return labels.clone(), torch.zeros_like(labels, dtype=torch.bool)
    # fraction ~ probability should have values set to true in the mask tensor
    mask = (
        torch.rand(size=labels.shape, device=labels.device, generator=generator) < probability
    )
    # set the labels tensor to null where mask = True
    conditions = torch.where(condition=mask, input=torch.full_like(labels, null_label), other=labels)
    return conditions, mask
