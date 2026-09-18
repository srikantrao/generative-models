from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import fmean
from time import perf_counter

import matplotlib.pyplot as plt
import torch
import torchvision
import yaml

from generative_models.data.mnist import mnist_raw_file_sha256
from generative_models.training.mnist_ddpm import (
    MNISTDDPMTrainingConfig,
    train_mnist_conditional_ddpm,
)
from generative_models.viz.points import save_figure, use_clean_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mnist_conditional_ddpm.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/mnist_conditional_ddpm"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MNISTDDPMTrainingConfig(**yaml.safe_load(args.config.read_text()))
    if args.num_steps is not None:
        config = replace(config, num_steps=args.num_steps)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)
    started = perf_counter()
    result = train_mnist_conditional_ddpm(config, device=args.device)
    training_seconds = perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.output_dir / "checkpoint.pt"
    config_path = args.output_dir / "config.yaml"
    losses_path = args.output_dir / "loss.jsonl"
    loss_plot_path = args.output_dir / "loss.png"
    summary_path = args.output_dir / "summary.json"
    failure_notes_path = args.output_dir / "failure_notes.md"

    config_path.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
    with losses_path.open("w") as file:
        for step, loss in enumerate(result.losses, start=1):
            file.write(json.dumps({"step": step, "loss": loss}) + "\n")

    torch.save(
        {
            "family": "ddpm",
            "prediction": "epsilon",
            "config": asdict(config),
            "model_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in result.model.state_dict().items()
            },
            "completed_steps": len(result.losses),
            "raw_file_sha256": mnist_raw_file_sha256(config.data_root),
        },
        checkpoint_path,
    )

    use_clean_style()
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(range(1, len(result.losses) + 1), result.losses)
    axis.set(
        title="Conditional MNIST DDPM",
        xlabel="optimization step",
        ylabel="epsilon MSE",
    )
    save_figure(figure, loss_plot_path)
    plt.close(figure)

    with checkpoint_path.open("rb") as file:
        checkpoint_sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    window = min(100, len(result.losses))
    summary = {
        "family": "ddpm",
        "prediction": "epsilon",
        "config": asdict(config),
        "parameter_count": sum(
            parameter.numel() for parameter in result.model.parameters()
        ),
        "device": args.device,
        "training_seconds": training_seconds,
        "completed_steps": len(result.losses),
        "terminal_alpha_bar": float(result.schedule.alpha_bars[-1]),
        "first_window_loss": fmean(result.losses[:window]),
        "last_window_loss": fmean(result.losses[-window:]),
        "checkpoint_sha256": checkpoint_sha256,
        "raw_file_sha256": mnist_raw_file_sha256(config.data_root),
        "versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    failure_notes_path.write_text(
        "# Failure Notes\n\nTraining completed. Sample quality has not been reviewed.\n"
    )
    print(f"Wrote {checkpoint_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
