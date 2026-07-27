from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from generative_models.models.flow_mlp import TimeConditionedMLPVectorField
from generative_models.models.mlp import TimeConditionedMLPDenoiser
from generative_models.training.toy_ddpm import ToyDDPMTrainingConfig
from generative_models.training.toy_flow_matching import (
    ToyFlowMatchingTrainingConfig,
)

REQUIRED_CHECKPOINT_KEYS = frozenset({"config", "model_state_dict"})


@dataclass(frozen=True)
class LoadedToyDDPMCheckpoint:
    model: TimeConditionedMLPDenoiser
    config: ToyDDPMTrainingConfig
    sha256: str
    num_parameters: int


@dataclass(frozen=True)
class LoadedToyFlowCheckpoint:
    model: TimeConditionedMLPVectorField
    config: ToyFlowMatchingTrainingConfig
    sha256: str
    num_parameters: int


def sha256_file(path: Path) -> str:
    with path.open("rb") as checkpoint_file:
        return hashlib.file_digest(checkpoint_file, "sha256").hexdigest()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _load_checkpoint_payload(
    path: Path,
    *,
    device: torch.device,
) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")

    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a mapping")

    missing_keys = REQUIRED_CHECKPOINT_KEYS.difference(payload)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"checkpoint is missing required keys: {missing}")

    return cast(Mapping[str, object], payload)


def _checkpoint_parts(
    payload: Mapping[str, object],
) -> tuple[dict[str, object], Mapping[str, Tensor]]:
    raw_config = payload["config"]

    raw_state_dict = payload["model_state_dict"]
    config = cast(dict[str, object], dict(raw_config))
    state_dict = cast(Mapping[str, Tensor], raw_state_dict)
    return config, state_dict


def load_toy_ddpm_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> LoadedToyDDPMCheckpoint:
    payload = _load_checkpoint_payload(path, device=device)
    raw_config, state_dict = _checkpoint_parts(payload)
    config = ToyDDPMTrainingConfig.from_mapping(raw_config)

    model = TimeConditionedMLPDenoiser(
        data_dim=2,
        time_embedding_dim=config.time_embedding_dim,
        hidden_dim=config.hidden_dim,
        num_hidden_layers=config.num_hidden_layers,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return LoadedToyDDPMCheckpoint(
        model=model,
        config=config,
        sha256=sha256_file(path),
        num_parameters=count_parameters(model),
    )


def load_toy_flow_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> LoadedToyFlowCheckpoint:
    payload = _load_checkpoint_payload(path, device=device)
    raw_config, state_dict = _checkpoint_parts(payload)
    config = ToyFlowMatchingTrainingConfig.from_mapping(raw_config)

    model = TimeConditionedMLPVectorField(
        data_dim=2,
        time_embedding_dim=config.time_embedding_dim,
        hidden_dim=config.hidden_dim,
        num_hidden_layers=config.num_hidden_layers,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return LoadedToyFlowCheckpoint(
        model=model,
        config=config,
        sha256=sha256_file(path),
        num_parameters=count_parameters(model),
    )
