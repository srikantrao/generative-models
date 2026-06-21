import pytest
import torch

from generative_models.data.toy import sample_labeled_gaussian_mixture


def test_labeled_gaussian_mixture_shapes_and_dtypes() -> None:
    batch = sample_labeled_gaussian_mixture(
        32,
        generator=torch.Generator().manual_seed(0),
    )

    assert batch.x.shape == (32, 2)
    assert batch.y.shape == (32,)
    assert batch.x.dtype == torch.float32
    assert batch.y.dtype == torch.int64


def test_labeled_gaussian_mixture_labels_are_in_range() -> None:
    batch = sample_labeled_gaussian_mixture(
        256,
        generator=torch.Generator().manual_seed(0),
    )

    assert int(batch.y.min()) >= 0
    assert int(batch.y.max()) < 3


def test_labeled_gaussian_mixture_is_reproducible_with_same_seed() -> None:
    batch_a = sample_labeled_gaussian_mixture(
        64,
        generator=torch.Generator().manual_seed(0),
    )

    batch_b = sample_labeled_gaussian_mixture(
        64,
        generator=torch.Generator().manual_seed(0),
    )

    # batch_a and batch_b should be equal
    assert torch.equal(batch_a.y, batch_b.y)
    assert torch.allclose(batch_a.x, batch_b.x)


def test_labeled_gaussian_mixture_supports_class_probabilities() -> None:
    batch = sample_labeled_gaussian_mixture(
        32,
        class_probs=torch.tensor([1.0, 0.0, 0.0]),
        generator=torch.Generator().manual_seed(11),
    )

    assert torch.equal(batch.y, torch.zeros(32, dtype=torch.int64))


def test_labeled_gaussian_mixture_rejects_invalid_center_shape() -> None:
    with pytest.raises(ValueError, match="centers must have shape"):
        _ = sample_labeled_gaussian_mixture(
            16,
            centers=torch.zeros(3, 3),
            generator=torch.Generator().manual_seed(0),
        )


def test_labeled_gaussian_mixture_rejects_invalid_class_probs() -> None:
    with pytest.raises(ValueError, match="class_probs must sum to 1"):
        _ = sample_labeled_gaussian_mixture(
            16,
            class_probs=torch.tensor([0.2, 0.2, 0.2]),
            generator=torch.Generator().manual_seed(0),
        )
