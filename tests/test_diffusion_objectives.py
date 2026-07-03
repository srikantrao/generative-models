import pytest
import torch
from torch import Tensor

from generative_models.diffusion.objectives import epsilon_prediction_loss
from generative_models.diffusion.schedules import build_diffusion_schedule


class ZeroNoisePredictor:
    def __call__(self, xt: Tensor, timesteps: Tensor) -> Tensor:
        # suppress the unused-argument lint/static-check warning
        del timesteps
        return torch.zeros_like(xt)


class FixedNoisePredictor:
    def __init__(self, predicted_noise: Tensor) -> None:
        self._predicted_noise: Tensor = predicted_noise

    def __call__(self, xt: Tensor, timesteps: Tensor) -> Tensor:
        # suppress the unused-argument lint/static-check warning
        del timesteps
        return self._predicted_noise.to(device=xt.device, dtype=xt.dtype)


class BadShapePredictor:
    def __call__(self, xt: Tensor, timesteps: Tensor) -> Tensor:
        del timesteps
        return torch.zeros(xt.shape[0], 1, device=xt.device, dtype=xt.dtype)


def test_epsilon_prediction_loss_is_zero_for_perfect_prediction() -> None:
    noise_schedule = build_diffusion_schedule(
        betas=torch.tensor([0.1], dtype=torch.float32)
    )
    x0 = torch.zeros(2, 2)
    noise = torch.tensor(
        [
            [1.0, -1.0],
            [0.5, -0.5],
        ]
    )
    timesteps = torch.tensor([0, 0], dtype=torch.long)
    model = FixedNoisePredictor(noise)

    output = epsilon_prediction_loss(
        model,
        noise_schedule,
        x0,
        timesteps=timesteps,
        noise=noise,
    )

    assert torch.allclose(output.loss, torch.tensor(0.0))
    assert torch.allclose(output.per_example_loss, torch.zeros(2))
    assert torch.equal(output.predicted_noise, noise)
    assert torch.equal(output.target_noise, noise)
    assert torch.equal(output.timesteps, timesteps)


def test_epsilon_prediction_loss_matches_known_mse() -> None:
    schedule = build_diffusion_schedule(torch.tensor([0.1], dtype=torch.float32))
    x0 = torch.zeros(2, 2)
    noise = torch.tensor(
        [
            [1.0, 3.0],
            [2.0, 4.0],
        ],
        dtype=torch.float32,
    )
    timesteps = torch.tensor([0, 0], dtype=torch.long)

    output = epsilon_prediction_loss(
        ZeroNoisePredictor(),
        schedule,
        x0,
        timesteps=timesteps,
        noise=noise,
    )

    assert torch.allclose(output.per_example_loss, torch.tensor([5.0, 10.0]))
    assert torch.allclose(output.loss, torch.tensor(7.5))


def test_epsilon_prediction_loss_returns_forward_diagnostics() -> None:
    schedule = build_diffusion_schedule(torch.tensor([0.1], dtype=torch.float32))
    x0 = torch.ones(2, 2)
    noise = torch.zeros_like(x0)
    timesteps = torch.tensor([0, 0], dtype=torch.long)

    output = epsilon_prediction_loss(
        ZeroNoisePredictor(),
        schedule,
        x0,
        timesteps=timesteps,
        noise=noise,
    )

    assert output.xt.shape == x0.shape
    assert torch.allclose(output.xt, torch.sqrt(torch.tensor(0.9)) * x0)
    assert torch.equal(output.target_noise, noise)
    assert torch.equal(output.timesteps, timesteps)


def test_epsilon_prediction_loss_can_sample_random_terms_reproducibly() -> None:
    schedule = build_diffusion_schedule(torch.tensor([0.1, 0.2], dtype=torch.float32))
    x0 = torch.zeros(4, 2)
    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)

    output_a = epsilon_prediction_loss(
        ZeroNoisePredictor(),
        schedule,
        x0,
        generator=generator_a,
    )
    output_b = epsilon_prediction_loss(
        ZeroNoisePredictor(),
        schedule,
        x0,
        generator=generator_b,
    )

    assert torch.equal(output_a.timesteps, output_b.timesteps)
    assert torch.allclose(output_a.target_noise, output_b.target_noise)
    assert torch.allclose(output_a.xt, output_b.xt)
    assert torch.allclose(output_a.loss, output_b.loss)


def test_epsilon_prediction_loss_supports_image_tensors() -> None:
    schedule = build_diffusion_schedule(torch.tensor([0.1], dtype=torch.float32))
    x0 = torch.zeros(2, 3, 4, 4)
    noise = torch.ones_like(x0)
    timesteps = torch.tensor([0, 0], dtype=torch.long)

    output = epsilon_prediction_loss(
        ZeroNoisePredictor(),
        schedule,
        x0,
        timesteps=timesteps,
        noise=noise,
    )

    assert output.xt.shape == x0.shape
    assert output.predicted_noise.shape == x0.shape
    assert output.target_noise.shape == x0.shape
    assert output.per_example_loss.shape == (2,)
    assert torch.allclose(output.loss, torch.tensor(1.0))


def test_epsilon_prediction_loss_rejects_bad_prediction_shape() -> None:
    schedule = build_diffusion_schedule(torch.tensor([0.1], dtype=torch.float32))
    x0 = torch.zeros(2, 2)
    timesteps = torch.tensor([0, 0], dtype=torch.long)
    noise = torch.zeros_like(x0)

    with pytest.raises(ValueError, match="predicted_noise must have shape"):
        _ = epsilon_prediction_loss(
            BadShapePredictor(),
            schedule,
            x0,
            timesteps=timesteps,
            noise=noise,
        )


def test_epsilon_prediction_loss_rejects_invalid_x0_shape() -> None:
    schedule = build_diffusion_schedule(torch.tensor([0.1], dtype=torch.float32))

    with pytest.raises(ValueError, match="x0 must have shape"):
        _ = epsilon_prediction_loss(
            ZeroNoisePredictor(),
            schedule,
            torch.zeros(2),
            timesteps=torch.tensor([0, 0], dtype=torch.long),
            noise=torch.zeros(2),
        )
