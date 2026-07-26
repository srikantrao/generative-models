from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} values must be numbers")

    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} values must be finite")
    return parsed


@dataclass(frozen=True)
class ToyGaussianMixtureSpec:
    name: str
    centers: tuple[tuple[float, float], ...]
    std: float
    class_probs: tuple[float, ...]

    def validate(self) -> None:
        if not isinstance(self.name, str):
            raise ValueError("name must be a string")
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if not isinstance(self.centers, tuple):
            raise ValueError("centers must be a tuple")
        if len(self.centers) < 2:
            raise ValueError("at least two centers are required")

        for index, center in enumerate(self.centers):
            if not isinstance(center, tuple):
                raise ValueError(f"centers[{index}] must be a tuple")
            if len(center) != 2:
                raise ValueError(f"centers[{index}] must contain two coordinates")
            for value in center:
                _finite_float(value, field=f"centers[{index}]")

        std = _finite_float(self.std, field="std")
        if std <= 0:
            raise ValueError("std must be finite and positive")
        if not isinstance(self.class_probs, tuple):
            raise ValueError("class_probs must be a tuple")
        if len(self.class_probs) != len(self.centers):
            raise ValueError("class_probs must contain one value per center")
        for probability in self.class_probs:
            _finite_float(probability, field="class_probs")
        if any(probability < 0 for probability in self.class_probs):
            raise ValueError("class_probs cannot contain negative values")
        if not math.isclose(
            math.fsum(self.class_probs),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("class_probs must sum to 1")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> ToyGaussianMixtureSpec:
        if not all(isinstance(key, str) for key in values):
            raise ValueError("data spec keys must be strings")

        required_keys = frozenset({"name", "centers", "std", "class_probs"})
        actual_keys = frozenset(values)
        missing_keys = required_keys.difference(actual_keys)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"data spec is missing required keys: {missing}")

        unknown_keys = actual_keys.difference(required_keys)
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            raise ValueError(f"data spec contains unknown keys: {unknown}")

        raw_name = values["name"]
        if not isinstance(raw_name, str):
            raise ValueError("name must be a string")

        raw_centers = values["centers"]
        if not isinstance(raw_centers, (list, tuple)):
            raise ValueError("centers must be a sequence")

        centers: list[tuple[float, float]] = []
        for index, raw_center in enumerate(raw_centers):
            if not isinstance(raw_center, (list, tuple)):
                raise ValueError(f"centers[{index}] must be a sequence")
            if len(raw_center) != 2:
                raise ValueError(f"centers[{index}] must contain two coordinates")

            first = _finite_float(raw_center[0], field=f"centers[{index}]")
            second = _finite_float(raw_center[1], field=f"centers[{index}]")
            centers.append((first, second))

        raw_probs = values["class_probs"]
        if not isinstance(raw_probs, (list, tuple)):
            raise ValueError("class_probs must be a sequence")
        class_probs = tuple(
            _finite_float(probability, field="class_probs") for probability in raw_probs
        )

        spec = cls(
            name=raw_name,
            centers=tuple(centers),
            std=_finite_float(values["std"], field="std"),
            class_probs=class_probs,
        )
        spec.validate()
        return spec

    def to_tensors(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor]:
        self.validate()
        if not dtype.is_floating_point:
            raise ValueError("dtype must be floating")

        centers = torch.tensor(self.centers, device=device, dtype=dtype)
        class_probs = torch.tensor(self.class_probs, device=device, dtype=dtype)
        return centers, class_probs


DEFAULT_TOY_GAUSSIAN_MIXTURE_SPEC = ToyGaussianMixtureSpec(
    name="three_gaussians_v1",
    centers=((-1.5, -0.8), (1.5, -0.8), (0.0, 1.4)),
    std=0.15,
    class_probs=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
)
DEFAULT_TOY_GAUSSIAN_MIXTURE_SPEC.validate()

DEFAULT_CENTERS = torch.tensor(
    DEFAULT_TOY_GAUSSIAN_MIXTURE_SPEC.centers,
    dtype=torch.float32,
)


@dataclass
class ToyBatch:
    x: Tensor
    y: Tensor


def sample_labeled_gaussian_mixture(
    num_samples: int,
    *,
    centers: Tensor | None = None,
    std: float = DEFAULT_TOY_GAUSSIAN_MIXTURE_SPEC.std,
    class_probs: Tensor | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> ToyBatch:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not math.isfinite(std) or std <= 0:
        raise ValueError("std must be finite and positive")

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
                f"class_probs must have shape [{num_classes}], "
                f"got shape {tuple(probs.shape)}"
            )
        if torch.any(probs < 0):
            raise ValueError("class_probs cannot contain negative values")
        if not torch.isclose(
            probs.sum(),
            torch.tensor(1.0, device=output_device, dtype=probs.dtype),
        ):
            raise ValueError("class_probs must sum to 1")

        y = torch.multinomial(
            probs, num_samples=num_samples, replacement=True, generator=generator
        ).to(device=output_device)

    noise = torch.randn(
        num_samples, 2, generator=generator, device=output_device, dtype=torch.float32
    )
    x = mixture_centers[y] + std * noise

    return ToyBatch(x=x, y=y)
