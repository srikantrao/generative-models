from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt
import torch
import torchvision
import yaml

from generative_models.data.mnist import mnist_raw_file_sha256
from generative_models.training.mnist_flow_matching import (
    MNISTFlowTrainingConfig,
    train_mnist_conditional_flow,
)
from generative_models.viz.points import save_figure, use_clean_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/mnist_conditional_flow.yaml")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/mnist_conditional_flow")
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:

    args = parse_args()
    config = MNISTFlowTrainingConfig(**yaml.safe_load(args.config.read_text()))
    result = train_mnist_conditional_flow(config, device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    conditioning = {
        "use_null_condition": config.use_null_condition,
        "null_label": config.num_classes if config.use_null_condition else None,
        "dropout_probability": config.condition_dropout,
        "condition_seed": config.condition_seed,
        "dropped_examples": result.dropped_conditions,
        "seen_examples": result.seen_conditions,
        "realized_dropout": result.dropped_conditions / result.seen_conditions
    }

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
            "family": "flow_matching",
            "prediction": "velocity",
            "path": "linear_noise_to_data",
            "time_sampling": "uniform",
            "config": asdict(config),
            "model_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in result.model.state_dict().items()
            },
            "completed_steps": len(result.losses),
            "conditioning": conditioning,
            "raw_file_sha256": mnist_raw_file_sha256(config.data_root),
        },
        checkpoint_path,
    )

    use_clean_style()
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(result.losses)
    axis.set(
        title="Conditional MNIST Flow Matching",
        xlabel="optimization step",
        ylabel="velocity MSE",
    )
    save_figure(figure, loss_plot_path)
    plt.close(figure)

    with checkpoint_path.open("rb") as file:
        checkpoint_sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    window = min(100, len(result.losses))
    summary = {
        "conditioning": conditioning,
        "family": "flow_matching",
        "prediction": "velocity",
        "path": "linear_noise_to_data",
        "time_sampling": "uniform",
        "config": asdict(config),
        "parameter_count": sum(
            parameter.numel() for parameter in result.model.parameters()
        ),
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
        "# Failure Notes\n\nNo retained failures for this run.\n"
    )
    print(f"Wrote {checkpoint_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
