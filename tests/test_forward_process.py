import pytest
import torch

from generative_models.diffusion.forward import (
    extract_schedule_values,
    q_sample,
    sample_timesteps,
)
from generative_models.diffusion.schedules import build_diffusion_schedule


def test_sample_timesteps_returns_long_indices_in_range() -> None:
    generator = torch.Generator().manual_seed(7)

    timesteps = sample_timesteps(batch_size=128, num_timesteps=10, generator=generator)

    assert timesteps.shape == (128,)
    assert timesteps.dtype == torch.long
    assert torch.all(timesteps >= 0)
    assert torch.all(timesteps < 10)


def test_extract_schedule_values_broadcasts_for_points() -> None:
    values = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    timesteps = torch.tensor([0, 2], dtype=torch.long)

    extracted = extract_schedule_values(
        values=values, timesteps=timesteps, target_shape=torch.Size([2, 4])
    )

    assert extracted.shape == (2, 1)
    assert torch.allclose(extracted[:, 0], torch.tensor([0.1, 0.3]))


def test_extract_schedule_values_broadcast_for_images() -> None:
    values = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    timesteps = torch.tensor([1, 2], dtype=torch.long)

    extracted = extract_schedule_values(
        values=values, timesteps=timesteps, target_shape=torch.Size([2, 3, 8, 8])
    )

    assert extracted.shape == (2, 1, 1, 1)
    assert extracted.shape == (2, 1, 1, 1)
    assert torch.allclose(extracted[:, 0, 0, 0], torch.tensor([0.2, 0.3]))


def test_q_sample_matches_known_formula() -> None:
    schedule = build_diffusion_schedule(torch.tensor([0.1, 0.2], dtype=torch.float32))
    x0 = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    noise = torch.ones_like(x0)
    timesteps = torch.tensor([0, 1], dtype=torch.long)

    sample = q_sample(schedule, x0, timesteps, noise=noise)

    signal = torch.sqrt(torch.tensor([[0.9], [0.72]], dtype=torch.float32))
    noise_scale = torch.sqrt(torch.tensor([[0.1], [0.28]], dtype=torch.float32))
    expected = signal * x0 + noise_scale * noise

    assert torch.allclose(sample.xt, expected)
    assert torch.equal(sample.noise, noise)
    assert torch.equal(sample.timesteps, timesteps)


def test_q_sample_supports_image_tensors() -> None:
    schedule = build_diffusion_schedule(torch.tensor([0.1, 0.2], dtype=torch.float32))
    x0 = torch.ones(2, 3, 4, 4)
    noise = torch.zeros_like(x0)
    timesteps = torch.tensor([0, 1], dtype=torch.long)

    sample = q_sample(schedule, x0, timesteps, noise=noise)

    assert sample.xt.shape == x0.shape
    assert torch.all(sample.xt[0] == torch.sqrt(torch.tensor(0.9)))
    assert torch.all(sample.xt[1] == torch.sqrt(torch.tensor(0.72)))


def test_q_sample_can_sample_noise_reproducibly() -> None:
    schedule = build_diffusion_schedule(torch.tensor([0.1], dtype=torch.float32))
    x0 = torch.zeros(2, 2)
    timesteps = torch.tensor([0, 0], dtype=torch.long)
    generator_a = torch.Generator().manual_seed(11)
    generator_b = torch.Generator().manual_seed(11)

    sample_a = q_sample(schedule, x0, timesteps, generator=generator_a)
    sample_b = q_sample(schedule, x0, timesteps, generator=generator_b)

    assert sample_a.xt.shape == x0.shape
    assert sample_a.noise.shape == x0.shape
    assert torch.allclose(sample_a.noise, sample_b.noise)
    assert torch.allclose(sample_a.xt, sample_b.xt)


def test_q_sample_rejects_timestep_out_of_range() -> None:
    schedule = build_diffusion_schedule(torch.tensor([0.1], dtype=torch.float32))

    with pytest.raises(ValueError, match="timesteps must be in"):
        _ = q_sample(
            schedule,
            torch.zeros(2, 2),
            torch.tensor([0, 1], dtype=torch.long),
        )


def test_q_sample_rejects_noise_shape_mismatch() -> None:
    schedule = build_diffusion_schedule(torch.tensor([0.1], dtype=torch.float32))

    with pytest.raises(ValueError, match="noise must have shape"):
        _ = q_sample(
            schedule,
            torch.zeros(2, 2),
            torch.zeros(2, dtype=torch.long),
            noise=torch.zeros(2, 3),
        )
