import torch

from generative_models.conditioning import drop_conditions
from generative_models.models.mnist_unet import MNISTUNet


def test_condition_dropout_endpoints_replay_and_independent_rng():
    labels = torch.arange(100) % 10
    original = labels.clone()
    for probability, expected in [(0.0, labels), (1.0, torch.full_like(labels, 10))]:
        conditions, mask = drop_conditions(
            labels,
            probability,
            null_label=10,
            generator=torch.Generator().manual_seed(45),
        )
        torch.testing.assert_close(conditions, expected)
        assert bool(mask.all()) == (probability == 1.0)
        assert bool(mask.any()) == (probability == 1.0)
    path_generator = torch.Generator().manual_seed(44)
    state = path_generator.get_state().clone()
    first, mask = drop_conditions(
        labels,
        0.5,
        null_label=10,
        generator=torch.Generator().manual_seed(45),
    )
    replay, _ = drop_conditions(
        labels,
        0.5,
        null_label=10,
        generator=torch.Generator().manual_seed(45),
    )
    torch.testing.assert_close(first, replay)
    torch.testing.assert_close(labels, original)
    torch.testing.assert_close(path_generator.get_state(), state)
    assert mask.any() and (~mask).any()
    torch.testing.assert_close(first[~mask], labels[~mask])
    assert (first[mask] == 10).all()


def test_null_embedding_is_learned_and_old_checkpoint_shape_still_loads():
    torch.manual_seed(42)
    kwargs = dict(base_channels=8, time_embedding_dim=16, dropout=0.0)
    ordinary = MNISTUNet(**kwargs)
    restored = MNISTUNet(**kwargs)
    restored.load_state_dict(ordinary.state_dict(), strict=True)
    assert ordinary.class_embedding.num_embeddings == 10
    assert ordinary.null_label is None

    joint = MNISTUNet(**kwargs, use_null_condition=True)
    assert joint.num_classes == 10
    assert joint.null_label == 10
    assert joint.class_embedding.num_embeddings == 11
    predictions = joint(
        torch.randn(2, 1, 28, 28),
        torch.tensor([0.2, 0.8]),
        torch.tensor([3, 10]),
    )
    predictions.square().mean().backward()
    gradient = joint.class_embedding.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient[3].abs().sum() > 0
    assert gradient[10].abs().sum() > 0
