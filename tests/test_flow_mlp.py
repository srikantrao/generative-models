import pytest
import torch

from generative_models.flow_matching.objectives import flow_matching_loss
from generative_models.models.continuous_time import ContinuousTimeEmbedding
from generative_models.models.flow_mlp import TimeConditionedMLPVectorField


def test_continuous_time_embedding_returns_expected_shape() -> None:
    embedding = ContinuousTimeEmbedding(embedding_dim=5)
    times = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)

    output = embedding(times)

    assert output.shape == (3, 5)
    assert torch.allclose(output[0, :2], torch.ones(2))
    assert torch.allclose(output[0, 2:4], torch.zeros(2))
    assert output[0, 4].item() == pytest.approx(0.0)


def test_continuous_time_embedding_rejects_integer_times() -> None:
    embedding = ContinuousTimeEmbedding(embedding_dim=4)

    with pytest.raises(ValueError, match="times must be floating point"):
        embedding(torch.tensor([0, 1], dtype=torch.long))


def test_continuous_time_embedding_rejects_times_outside_unit_interval() -> None:
    embedding = ContinuousTimeEmbedding(embedding_dim=4)

    with pytest.raises(ValueError, match=r"times must be in \[0, 1\]"):
        embedding(torch.tensor([-0.1, 0.5], dtype=torch.float32))


def test_vector_field_returns_velocity_with_same_shape_as_input() -> None:
    model = TimeConditionedMLPVectorField(
        data_dim=2,
        time_embedding_dim=8,
        hidden_dim=16,
        num_hidden_layers=1,
    )
    xt = torch.randn(4, 2)
    times = torch.rand(4)

    output = model(xt, times)

    assert output.shape == xt.shape


def test_vector_field_can_be_used_in_flow_matching_loss() -> None:
    model = TimeConditionedMLPVectorField(
        data_dim=2,
        time_embedding_dim=8,
        hidden_dim=16,
        num_hidden_layers=1,
    )
    data = torch.randn(4, 2)
    source_noise = torch.randn_like(data)
    times = torch.tensor([0.1, 0.3, 0.7, 0.9], dtype=torch.float32)

    output = flow_matching_loss(
        model,
        data,
        times=times,
        source_noise=source_noise,
    )
    output.loss.backward()

    assert output.predicted_velocity.shape == data.shape
    assert output.target_velocity.shape == data.shape
    assert torch.isfinite(output.loss)
    assert any(param.grad is not None for param in model.parameters())


def test_vector_field_rejects_bad_point_shape() -> None:
    model = TimeConditionedMLPVectorField(data_dim=2)

    with pytest.raises(ValueError, match="xt must have shape"):
        model(torch.zeros(2, 3, 4), torch.rand(2))


def test_vector_field_rejects_time_batch_mismatch() -> None:
    model = TimeConditionedMLPVectorField(data_dim=2)

    with pytest.raises(ValueError, match="times batch size"):
        model(torch.zeros(3, 2), torch.rand(2))
