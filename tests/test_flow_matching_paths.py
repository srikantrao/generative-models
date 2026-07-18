import pytest
import torch

from generative_models.flow_matching.paths import (
    expand_times,
    sample_linear_flow_path,
    sample_times,
)


def test_sample_times_returns_values_in_unit_interval() -> None:
    generator = torch.Generator().manual_seed(123)

    times = sample_times(128, generator=generator)

    assert times.shape == (128,)
    assert times.dtype == torch.float32
    assert torch.all(times >= 0)
    assert torch.all(times <= 1)


def test_expand_times_broadcasts_for_points() -> None:
    times = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)

    expanded = expand_times(times, torch.Size([3, 2]))

    assert expanded.shape == (3, 1)
    assert torch.allclose(expanded[:, 0], times)


def test_expand_times_broadcasts_for_images() -> None:
    times = torch.tensor([0.25, 0.75], dtype=torch.float32)

    expanded = expand_times(times, torch.Size([2, 3, 8, 8]))

    assert expanded.shape == (2, 1, 1, 1)
    assert torch.allclose(expanded[:, 0, 0, 0], times)


def test_linear_flow_path_matches_known_values() -> None:
    data = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
        ],
        dtype=torch.float32,
    )
    source_noise = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    times = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)

    sample = sample_linear_flow_path(
        data,
        times=times,
        source_noise=source_noise,
    )

    expected_xt = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=torch.float32,
    )
    expected_velocity = data - source_noise

    assert torch.allclose(sample.xt, expected_xt)
    assert torch.allclose(sample.target_velocity, expected_velocity)
    assert torch.equal(sample.times, times)
    assert torch.equal(sample.source_noise, source_noise)
    assert torch.equal(sample.data, data)


def test_linear_flow_path_can_sample_noise_reproducibly() -> None:
    data = torch.zeros(4, 2)
    generator_a = torch.Generator().manual_seed(7)
    generator_b = torch.Generator().manual_seed(7)

    sample_a = sample_linear_flow_path(data, generator=generator_a)
    sample_b = sample_linear_flow_path(data, generator=generator_b)

    assert torch.allclose(sample_a.times, sample_b.times)
    assert torch.allclose(sample_a.source_noise, sample_b.source_noise)
    assert torch.allclose(sample_a.xt, sample_b.xt)
    assert torch.allclose(sample_a.target_velocity, sample_b.target_velocity)


def test_linear_flow_path_supports_image_tensors() -> None:
    data = torch.ones(2, 3, 4, 4)
    source_noise = torch.zeros_like(data)
    times = torch.tensor([0.25, 0.75], dtype=torch.float32)

    sample = sample_linear_flow_path(
        data,
        times=times,
        source_noise=source_noise,
    )

    assert sample.xt.shape == data.shape
    assert torch.allclose(sample.xt[0], torch.full((3, 4, 4), 0.25))
    assert torch.allclose(sample.xt[1], torch.full((3, 4, 4), 0.75))
    assert torch.allclose(sample.target_velocity, torch.ones_like(data))


def test_linear_flow_path_rejects_times_outside_unit_interval() -> None:
    data = torch.zeros(2, 2)

    with pytest.raises(ValueError, match="times must be in"):
        sample_linear_flow_path(
            data,
            times=torch.tensor([0.0, 1.1], dtype=torch.float32),
            source_noise=torch.zeros_like(data),
        )


def test_linear_flow_path_rejects_noise_shape_mismatch() -> None:
    data = torch.zeros(2, 2)

    with pytest.raises(ValueError, match="source_noise must have shape"):
        sample_linear_flow_path(
            data,
            times=torch.tensor([0.0, 1.0], dtype=torch.float32),
            source_noise=torch.zeros(2, 3),
        )


def test_linear_flow_path_rejects_integer_times() -> None:
    data = torch.zeros(2, 2)

    with pytest.raises(ValueError, match="floating point dtype"):
        sample_linear_flow_path(
            data,
            times=torch.tensor([0, 1], dtype=torch.long),
            source_noise=torch.zeros_like(data),
        )
