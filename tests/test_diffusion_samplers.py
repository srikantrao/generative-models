import pytest
import torch
from torch import Tensor

from generative_models.diffusion.samplers import (
    ddpm_reverse_mean,
    ddpm_reverse_step,
    sample_ddpm,
)
from generative_models.diffusion.schedules import build_diffusion_schedule


class ZeroNoisePredictor:
    def __call__(self, xt: Tensor, timesteps: Tensor) -> Tensor:
        del timesteps
        return torch.zeros_like(xt)


class BadShapePredictor:
    def __call__(self, xt: Tensor, timesteps: Tensor) -> Tensor:
        del timesteps
        return torch.zeros(xt.shape[0], 1, device=xt.device, dtype=xt.dtype)


def test_ddpm_reverse_mean_matches_known_formula() -> None:
    schedule = build_diffusion_schedule(torch.tensor([0.1, 0.2], dtype=torch.float32))
    xt = torch.tensor([[2.0, 4.0]], dtype=torch.float32)
    timesteps = torch.tensor([1], dtype=torch.long)
    predicted_noise = torch.ones_like(xt)

    mean = ddpm_reverse_mean(schedule, xt, timesteps, predicted_noise)

    expected = xt - (0.2 / torch.sqrt(torch.tensor(0.28))) * predicted_noise
    expected = expected / torch.sqrt(torch.tensor(0.8))
    assert torch.allclose(mean, expected)


def test_ddpm_reverse_step_adds_no_noise_at_t_zero() -> None:
    schedule = build_diffusion_schedule(torch.tensor([0.1], dtype=torch.float32))
    xt = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    timesteps = torch.tensor([0], dtype=torch.long)
    generator = torch.Generator().manual_seed(123)

    step = ddpm_reverse_step(
        ZeroNoisePredictor(),
        schedule,
        xt,
        timesteps,
        generator=generator,
    )

    assert torch.allclose(step.x_previous, step.mean)


def test_ddpm_reverse_step_preserves_shape() -> None:
    schedule = build_diffusion_schedule(torch.tensor([0.1, 0.2], dtype=torch.float32))
    xt = torch.randn(4, 2)
    timesteps = torch.tensor([1, 1, 1, 1], dtype=torch.long)

    step = ddpm_reverse_step(
        ZeroNoisePredictor(),
        schedule,
        xt,
        timesteps,
        generator=torch.Generator().manual_seed(123),
    )

    assert step.x_previous.shape == xt.shape
    assert step.mean.shape == xt.shape
    assert step.predicted_noise.shape == xt.shape


def test_sample_ddpm_returns_samples_and_trajectory() -> None:
    schedule = build_diffusion_schedule(
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    )
    generator = torch.Generator().manual_seed(123)

    sample = sample_ddpm(
        ZeroNoisePredictor(),
        schedule,
        num_samples=5,
        sample_shape=(2,),
        generator=generator,
        return_trajectory=True,
    )

    assert sample.samples.shape == (5, 2)
    assert sample.trajectory is not None
    assert sample.trajectory.shape == (4, 5, 2)


def test_sample_ddpm_is_reproducible_for_same_seed() -> None:
    schedule = build_diffusion_schedule(
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    )

    sample_a = sample_ddpm(
        ZeroNoisePredictor(),
        schedule,
        num_samples=5,
        generator=torch.Generator().manual_seed(123),
    )
    sample_b = sample_ddpm(
        ZeroNoisePredictor(),
        schedule,
        num_samples=5,
        generator=torch.Generator().manual_seed(123),
    )

    assert torch.allclose(sample_a.samples, sample_b.samples)


def test_ddpm_reverse_step_rejects_bad_model_output_shape() -> None:
    schedule = build_diffusion_schedule(torch.tensor([0.1], dtype=torch.float32))

    with pytest.raises(ValueError, match="predicted_noise must have shape"):
        ddpm_reverse_step(
            BadShapePredictor(),
            schedule,
            torch.zeros(2, 2),
            torch.zeros(2, dtype=torch.long),
        )


def test_sample_ddpm_rejects_invalid_num_samples() -> None:
    schedule = build_diffusion_schedule(torch.tensor([0.1], dtype=torch.float32))

    with pytest.raises(ValueError, match="num_samples must be positive"):
        sample_ddpm(ZeroNoisePredictor(), schedule, num_samples=0)
