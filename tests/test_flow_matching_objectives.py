import pytest
import torch
from torch import Tensor

from generative_models.flow_matching.objectives import flow_matching_loss


class ZeroVelocityPredictor:
    def __call__(self, xt: Tensor, times: Tensor) -> Tensor:
        del times
        return torch.zeros_like(xt)


class FixedVelocityPredictor:
    def __init__(self, predicted_velocity: Tensor) -> None:
        self._predicted_velocity = predicted_velocity

    def __call__(self, xt: Tensor, times: Tensor) -> Tensor:
        del times
        return self._predicted_velocity.to(device=xt.device, dtype=xt.dtype)


class BadShapePredictor:
    def __call__(self, xt: Tensor, times: Tensor) -> Tensor:
        del times
        return torch.zeros(xt.shape[0], 1, device=xt.device, dtype=xt.dtype)


def test_flow_matching_loss_is_zero_for_perfect_prediction() -> None:
    data = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    source_noise = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    times = torch.tensor([0.25, 0.75], dtype=torch.float32)
    target_velocity = data - source_noise
    model = FixedVelocityPredictor(target_velocity)

    output = flow_matching_loss(
        model,
        data,
        times=times,
        source_noise=source_noise,
    )

    assert torch.allclose(output.loss, torch.tensor(0.0))
    assert torch.allclose(output.per_example_loss, torch.zeros(2))
    assert torch.equal(output.predicted_velocity, target_velocity)
    assert torch.equal(output.target_velocity, target_velocity)
    assert torch.equal(output.times, times)


def test_flow_matching_loss_matches_known_mse() -> None:
    data = torch.tensor([[1.0, 3.0], [2.0, 4.0]], dtype=torch.float32)
    source_noise = torch.zeros_like(data)
    times = torch.tensor([0.25, 0.75], dtype=torch.float32)

    output = flow_matching_loss(
        ZeroVelocityPredictor(),
        data,
        times=times,
        source_noise=source_noise,
    )

    assert torch.allclose(output.per_example_loss, torch.tensor([5.0, 10.0]))
    assert torch.allclose(output.loss, torch.tensor(7.5))


def test_flow_matching_loss_returns_path_diagnostics() -> None:
    data = torch.ones(2, 2)
    source_noise = torch.zeros_like(data)
    times = torch.tensor([0.25, 0.75], dtype=torch.float32)

    output = flow_matching_loss(
        ZeroVelocityPredictor(),
        data,
        times=times,
        source_noise=source_noise,
    )

    assert output.xt.shape == data.shape
    assert torch.allclose(output.xt[0], torch.full((2,), 0.25))
    assert torch.allclose(output.xt[1], torch.full((2,), 0.75))
    assert torch.equal(output.source_noise, source_noise)
    assert torch.equal(output.data, data)


def test_flow_matching_loss_can_sample_random_terms_reproducibly() -> None:
    data = torch.zeros(4, 2)
    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)

    output_a = flow_matching_loss(
        ZeroVelocityPredictor(),
        data,
        generator=generator_a,
    )
    output_b = flow_matching_loss(
        ZeroVelocityPredictor(),
        data,
        generator=generator_b,
    )

    assert torch.allclose(output_a.times, output_b.times)
    assert torch.allclose(output_a.source_noise, output_b.source_noise)
    assert torch.allclose(output_a.xt, output_b.xt)
    assert torch.allclose(output_a.target_velocity, output_b.target_velocity)
    assert torch.allclose(output_a.loss, output_b.loss)


def test_flow_matching_loss_supports_image_tensors() -> None:
    data = torch.ones(2, 3, 4, 4)
    source_noise = torch.zeros_like(data)
    times = torch.tensor([0.25, 0.75], dtype=torch.float32)

    output = flow_matching_loss(
        ZeroVelocityPredictor(),
        data,
        times=times,
        source_noise=source_noise,
    )

    assert output.xt.shape == data.shape
    assert output.predicted_velocity.shape == data.shape
    assert output.target_velocity.shape == data.shape
    assert output.per_example_loss.shape == (2,)
    assert torch.allclose(output.loss, torch.tensor(1.0))


def test_flow_matching_loss_rejects_bad_prediction_shape() -> None:
    data = torch.zeros(2, 2)

    with pytest.raises(ValueError, match="predicted_velocity must have shape"):
        flow_matching_loss(
            BadShapePredictor(),
            data,
            times=torch.tensor([0.0, 1.0], dtype=torch.float32),
            source_noise=torch.zeros_like(data),
        )


def test_flow_matching_loss_rejects_invalid_data_shape() -> None:
    with pytest.raises(ValueError, match="data must have shape"):
        flow_matching_loss(
            ZeroVelocityPredictor(),
            torch.zeros(2),
            times=torch.tensor([0.0, 1.0], dtype=torch.float32),
            source_noise=torch.zeros(2),
        )
