from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

DEFAULT_CENTERS = torch.tensor(
    [[-1.5, -0.8], [1.5, -0.8], [0.0, 1.4]], dtype=torch.float32
)


@dataclass
class ToyBatch:
    x: Tensor
    y: Tensor


def sample_labeled_gaussian_mixture(
    num_samples: int,
    *,
    centers: Tensor | None = None,
    std: float = 0.15,
    class_probs: Tensor | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> ToyBatch:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if std <= 0:
        raise ValueError("std must be positive")

    output_device = torch.device("cpu" if device is None else device)
    mixture_centers = DEFAULT_CENTERS if centers is None else centers
    mixture_centers = mixture_centers.to(device=output_device, dtype=torch.float32)

    if mixture_centers.ndim != 2 or mixture_centers.shape[1] != 2:
        raise ValueError(
            f"centers must have shape [num_classes, 2], got {tuple(mixture_centers.shape)}"
        )

    num_classes = mixture_centers.shape[0]
    if num_classes < 2:
        raise ValueError("at least two classes are required")

    if class_probs is None:
        y = torch.randint(
            low=0,
            high=num_classes,
            size=(num_samples,),
            generator=generator,
            device=output_device,
        )
    else:
        probs = class_probs.to(device=output_device, dtype=torch.float32)

        if probs.shape != (num_classes,):
            raise ValueError(
                f"class_probs msut have shape [{num_classes}], got shape {tuple(probs.shape)}"
            )
        if torch.any(probs < 0):
            raise ValueError("class_probs cannot contain negative values")
        if not torch.isclose(probs.sum(), torch.tensor(1.0, device=output_device)):
            raise ValueError("class_probs must sum to 1")

        y = torch.multinomial(
            probs, num_samples=num_samples, replacement=True, generator=generator
        ).to(device=output_device)

    noise = torch.randn(
        num_samples, 2, generator=generator, device=output_device, dtype=torch.float32
    )
    x = mixture_centers[y] + std * noise

    return ToyBatch(x=x, y=y)
