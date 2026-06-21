import pytest
import torch

from generative_models.diffusion.schedules import (
    build_diffusion_schedule,
    cosine_beta_schedule,
    linear_beta_schedule,
)


def test_linear_beta_schedule_shape_and_endpoints() -> None:
    betas = linear_beta_schedule(
        5,
        beta_start=0.1,
        beta_end=0.5,
    )

    assert betas.shape == (5,)
    assert torch.isclose(betas[0], torch.tensor(0.1))
    assert torch.isclose(betas[-1], torch.tensor(0.5))


def test_linear_beta_schedule_is_monotonic() -> None:
    betas = linear_beta_schedule(100)

    assert torch.all(betas[1:] >= betas[:-1])
    assert torch.all((betas > 0) & (betas < 1))


def test_cosine_beta_schedule_shape_and_range() -> None:
    betas = cosine_beta_schedule(100)

    assert betas.shape == (100,)
    assert torch.all((betas > 0) & (betas < 1))


def test_build_diffusion_schedule_known_values() -> None:
    betas = torch.tensor([0.1, 0.2], dtype=torch.float32)

    schedule = build_diffusion_schedule(betas)

    assert torch.allclose(schedule.alphas, torch.tensor([0.9, 0.8]))
    assert torch.allclose(schedule.alpha_bars, torch.tensor([0.9, 0.72]))
    assert torch.allclose(
        schedule.sqrt_alpha_bars, torch.sqrt(torch.tensor([0.9, 0.72]))
    )
    assert torch.allclose(
        schedule.sqrt_one_minus_alpha_bars,
        torch.sqrt(torch.tensor([0.1, 0.28])),
    )
    assert schedule.num_timesteps == 2


def test_alpha_bars_decrease_over_time() -> None:
    betas = linear_beta_schedule(100)
    schedule = build_diffusion_schedule(betas)

    assert torch.all(schedule.alpha_bars[1:] < schedule.alpha_bars[:-1])


def test_schedule_rejects_invalid_betas_shape() -> None:
    with pytest.raises(ValueError, match="betas must have shape"):
        build_diffusion_schedule(torch.zeros(2, 2))


def test_schedule_rejects_betas_outside_open_interval() -> None:
    with pytest.raises(ValueError, match="all betas must be in"):
        build_diffusion_schedule(torch.tensor([0.1, 1.0]))


def test_linear_beta_schedule_rejects_invalid_timesteps() -> None:
    with pytest.raises(ValueError, match="num_timesteps must be positive"):
        linear_beta_schedule(0)
