from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.fft import Tensor
from torch.utils.data import DataLoader

from generative_models.data.mnist import load_mnist_split
from generative_models.flow_matching.conditional import (
    conditional_flow_matching_loss,
)
from generative_models.models.mnist_unet import MNISTUNet
from generative_models.seeds import seed_everything


@dataclass(frozen=True)
class MNISTFlowTrainingConfig:
    data_root: str = "data/mnist"
    download: bool = True
    training_seed: int = 42
    data_seed: int = 43
    path_seed: int = 44
    batch_size: int = 256
    num_workers: int = 0
    num_steps: int = 30_000
    log_every: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    base_channels: int = 32
    time_embedding_dim: int = 64
    num_classes: int = 10
    num_groups: int = 8
    dropout: float = 0.1


@dataclass
class MNISTFlowTrainingResult:
    model: MNISTUNet
    losses: list[float]


def build_mnist_flow_model(config: MNISTFlowTrainingConfig) -> MNISTUNet:
    return MNISTUNet(
        base_channels=config.base_channels,
        time_embedding_dim=config.time_embedding_dim,
        num_classes=config.num_classes,
        num_groups=config.num_groups,
        dropout=config.dropout,
    )


def train_mnist_conditional_flow(
    config: MNISTFlowTrainingConfig,
    *,
    device: torch.device | str,
) -> MNISTFlowTrainingResult:

    train_device = torch.device(device)
    seed_everything(config.training_seed)

    # Setup Data
    dataset = load_mnist_split(
        root=config.data_root, train=True, download=config.download
    )
    data_generator = torch.Generator().manual_seed(config.data_seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=train_device.type == "cuda",
        generator=data_generator,
    )

    # Setup Model
    model = build_mnist_flow_model(config).to(train_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    path_generator = torch.Generator(device=train_device).manual_seed(config.path_seed)

    losses: list[float] = []
    batches = iter(loader)
    model.train()

    for step in range(1, config.num_steps + 1):
        try:
            images, labels = next(batches)
        except StopIteration:
            batches = iter(loader)
            images, labels = next(batches)

        images: Tensor = images.to(train_device)
        labels: Tensor = labels.to(train_device)

        optimizer.zero_grad(set_to_none=True)
        output = conditional_flow_matching_loss(
            model, images, labels, generator=path_generator
        )
        output.loss.backward()
        optimizer.step()

        loss = float(output.loss.detach())
        losses.append(loss)
        if step == 1 or step % config.log_every == 0:
            print(f"step={step:05d} velocity_mse={loss:.4f}")

    return MNISTFlowTrainingResult(model=model, losses=losses)
