import pytest
import torch
from torch import Tensor

from generative_models.diffusion.objectives import epsilon_prediction_loss
from generative_models.diffusion.schedules import build_diffusion_schedule
from generative_models.models.mlp import TimeConditionedMLPDenoiser
from generative_models.models.time_embeddings import SinusoidalTimeEmbedding


def test_sinusoidal_time_embedding_shape_and_t_zero_values() -> None:
    embedding = SinusoidalTimeEmbedding(embedding_dim=6)
    timesteps = torch.tensor([0, 1, 2], dtype=torch.long)

    output: Tensor = embedding(timesteps)
    assert output.shape == (3, 6)
    assert torch.allclose(output[0, :3], torch.ones(3))
    assert torch.allclose(output[0, 3:], torch.zeros(3))


def test_sinusoidal_time_embedding_supports_odd_dimensions() -> None:
    embedding = SinusoidalTimeEmbedding(embedding_dim=5)
    timesteps = torch.tensor([0, 1], dtype=torch.long)

    output: Tensor = embedding(timesteps)
    assert output.shape == (2, 5)
    assert torch.allclose(output[:, -1], torch.zeros(2))


def test_sinusoidal_time_embedding_rejects_bad_timestep_shape() -> None:
    embedding = SinusoidalTimeEmbedding(embedding_dim=6)

    with pytest.raises(ValueError, match="timesteps must have shape"):
        embedding(torch.zeros(2, 1, dtype=torch.long))


def test_mlp_denoiser_returns_noise_shaped_like_input() -> None:
    torch.manual_seed(0)
    model = TimeConditionedMLPDenoiser(
        data_dim=2,
        time_embedding_dim=8,
        hidden_dim=16,
        num_hidden_layers=2,
    )
    xt = torch.randn(5, 2)
    timesteps = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    predicted_noise: Tensor = model(xt, timesteps)

    assert predicted_noise.shape == xt.shape
    assert predicted_noise.dtype == xt.dtype
    assert predicted_noise.device == xt.device


def test_mlp_denoiser_integrates_with_epsilon_prediction_loss() -> None:
    torch.manual_seed(0)
    schedule = build_diffusion_schedule(torch.tensor([0.1, 0.2], dtype=torch.float32))
    model = TimeConditionedMLPDenoiser(
        data_dim=2,
        time_embedding_dim=8,
        hidden_dim=16,
        num_hidden_layers=1,
    )
    x0 = torch.randn(4, 2)
    noise = torch.rand_like(x0)
    timesteps = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    output = epsilon_prediction_loss(
        model, schedule, x0, timesteps=timesteps, noise=noise
    )

    assert output.loss.ndim == 0
    assert output.predicted_noise.shape == x0.shape
    assert output.target_noise.shape == x0.shape


def test_mlp_denoiser_loss_backpropagates_to_parameters() -> None:
    torch.manual_seed(0)
    schedule = build_diffusion_schedule(torch.tensor([0.1, 0.2], dtype=torch.float32))
    model = TimeConditionedMLPDenoiser(
        data_dim=2,
        time_embedding_dim=8,
        hidden_dim=16,
        num_hidden_layers=1,
    )

    x0 = torch.randn(4, 2)
    noise = torch.rand_like(x0)
    timesteps = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    output = epsilon_prediction_loss(
        model,
        schedule,
        x0,
        timesteps=timesteps,
        noise=noise,
    )

    output.loss.backward()

    grad_norm = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert grad_norm > 0


def test_mlp_denoiser_rejects_image_tensors() -> None:
    model = TimeConditionedMLPDenoiser()

    with pytest.raises(ValueError, match="xt must have shape"):
        model(torch.zeros(2, 3, 4, 4), torch.tensor([0, 1], dtype=torch.long))


def test_mlp_denoiser_rejects_timestep_batch_mismatch() -> None:
    model = TimeConditionedMLPDenoiser()

    with pytest.raises(ValueError, match="does not match"):
        model(torch.zeros(3, 2), torch.tensor([0, 1], dtype=torch.long))


def test_mlp_denoiser_rejects_invalid_constructor_args() -> None:
    with pytest.raises(ValueError, match="data_dim must be positive"):
        _ = TimeConditionedMLPDenoiser(data_dim=0)
