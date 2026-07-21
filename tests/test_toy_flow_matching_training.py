import math

import pytest

from generative_models.training.toy_flow_matching import (
    ToyFlowMatchingTrainingConfig,
    train_toy_flow_matching,
)


def tiny_config(*, seed: int = 123) -> ToyFlowMatchingTrainingConfig:
    return ToyFlowMatchingTrainingConfig(
        num_steps=3,
        batch_size=16,
        learning_rate=1e-3,
        hidden_dim=16,
        time_embedding_dim=8,
        num_hidden_layers=1,
        seed=seed,
    )


def test_train_toy_flow_matching_runs_for_requested_steps() -> None:
    result = train_toy_flow_matching(tiny_config())

    assert len(result.losses) == 3
    assert all(math.isfinite(loss) for loss in result.losses)
    assert result.config.num_steps == 3


def test_train_toy_flow_matching_is_reproducible_for_same_seed() -> None:
    config = tiny_config(seed=99)

    result_a = train_toy_flow_matching(config)
    result_b = train_toy_flow_matching(config)

    assert result_a.losses == pytest.approx(result_b.losses)


def test_train_toy_flow_matching_changes_with_different_seed() -> None:
    result_a = train_toy_flow_matching(tiny_config(seed=1))
    result_b = train_toy_flow_matching(tiny_config(seed=2))

    assert any(
        abs(loss_a - loss_b) > 1e-8
        for loss_a, loss_b in zip(result_a.losses, result_b.losses, strict=True)
    )


def test_train_toy_flow_matching_rejects_invalid_config() -> None:
    config = ToyFlowMatchingTrainingConfig(num_steps=0)

    with pytest.raises(ValueError, match="num_steps must be positive"):
        train_toy_flow_matching(config)
