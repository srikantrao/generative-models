from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from generative_models.data.mnist import load_mnist_split
from generative_models.diffusion.conditional import conditional_epsilon_loss
from generative_models.diffusion.schedules import (
    DiffusionSchedule,
    build_diffusion_schedule,
    linear_beta_schedule,
)
from generative_models.models.mnist_unet import MNISTUNet
from generative_models.seeds import seed_everything


@dataclass(frozen=True)
class MNISTDDPMTrainingConfig:
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
    num_timesteps: int = 1_000
    beta_start: float = 1e-4
    beta_end: float = 2e-2


@dataclass
class MNISTDDPMTrainingResult:
    model: MNISTUNet
    schedule: DiffusionSchedule
    losses: list[float]


def build_mnist_ddpm_model(config: MNISTDDPMTrainingConfig) -> MNISTUNet:
    return MNISTUNet(
        base_channels=config.base_channels,
        time_embedding_dim=config.time_embedding_dim,
        num_classes=config.num_classes,
        num_groups=config.num_groups,
        dropout=config.dropout,
    )


def build_mnist_ddpm_schedule(
    config: MNISTDDPMTrainingConfig,
    *,
    device: torch.device,
) -> DiffusionSchedule:
    betas = linear_beta_schedule(
        config.num_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=device,
    )
    return build_diffusion_schedule(betas)


def train_mnist_conditional_ddpm(
    config: MNISTDDPMTrainingConfig,
    *,
    device: torch.device | str,
) -> MNISTDDPMTrainingResult:
    if config.num_steps < 1 or config.batch_size < 1 or config.log_every < 1:
        raise ValueError("num_steps, batch_size, and log_every must be positive")
    if config.num_timesteps < 2 or not 0 < config.beta_start <= config.beta_end < 1:
        raise ValueError(
            "require at least two timesteps and 0 < beta_start <= beta_end < 1"
        )
    if config.num_classes != 10:
        raise ValueError("MNIST uses ten class labels")
    train_device = torch.device(device)
    seed_everything(config.training_seed)

    dataset = load_mnist_split(
        config.data_root,
        train=True,
        download=config.download,
    )
    data_generator = torch.Generator().manual_seed(config.data_seed)
    if config.batch_size > len(dataset):
        raise ValueError("batch_size exceeds the training dataset with drop_last=True")
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=train_device.type == "cuda",
        generator=data_generator,
    )

    model = build_mnist_ddpm_model(config).to(train_device)
    schedule = build_mnist_ddpm_schedule(config, device=train_device)
    if float(schedule.alpha_bars[-1]) > 1e-3:
        raise ValueError(
            "terminal alpha_bar is too large for this standard-normal prior"
        )
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

        images = images.to(train_device)
        labels = labels.to(train_device)

        optimizer.zero_grad(set_to_none=True)
        output = conditional_epsilon_loss(
            model,
            schedule,
            images,
            labels,
            generator=path_generator,
        )
        if not bool(torch.isfinite(output.loss)):
            raise RuntimeError(f"non-finite epsilon loss at step {step}")
        output.loss.backward()
        optimizer.step()

        loss = float(output.loss.detach())
        losses.append(loss)
        if step == 1 or step % config.log_every == 0:
            print(f"step={step:05d} epsilon_mse={loss:.4f}")

    return MNISTDDPMTrainingResult(
        model=model,
        schedule=schedule,
        losses=losses,
    )
