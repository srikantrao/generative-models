from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from generative_models.data.mnist import denormalize_mnist_image
from generative_models.diffusion.conditional import sample_conditional_ddpm
from generative_models.evals.mnist_generation import (
    evaluate_generated_mnist,
    load_frozen_mnist_classifier,
)
from generative_models.training.mnist_ddpm import (
    MNISTDDPMTrainingConfig,
    build_mnist_ddpm_model,
    build_mnist_ddpm_schedule,
)
from generative_models.viz.points import save_figure, use_clean_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/mnist_conditional_ddpm/checkpoint.pt"),
    )
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path("runs/mnist_classifier/checkpoint.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/mnist_conditional_ddpm_samples"),
    )
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--source-seed", type=int, default=100)
    parser.add_argument("--ancestral-seed", type=int, default=101)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.samples_per_class < 1:
        parser.error("--samples-per-class must be positive")
    return args


def file_sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def save_grid(
    samples: torch.Tensor,
    path: Path,
    *,
    samples_per_class: int,
) -> None:
    use_clean_style()
    figure, axes = plt.subplots(
        10,
        samples_per_class,
        figsize=(samples_per_class, 10),
        squeeze=False,
    )
    display = denormalize_mnist_image(samples.detach().cpu()).clamp(0.0, 1.0)
    for label in range(10):
        for column in range(samples_per_class):
            index = label * samples_per_class + column
            axes[label, column].imshow(display[index, 0], cmap="gray", vmin=0, vmax=1)
            axes[label, column].set_xticks([])
            axes[label, column].set_yticks([])
            for spine in axes[label, column].spines.values():
                spine.set_visible(False)
        axes[label, 0].set_ylabel(str(label), rotation=0, labelpad=10)
    figure.tight_layout(pad=0.2)
    save_figure(figure, path)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )
    config = MNISTDDPMTrainingConfig(**checkpoint["config"])
    if checkpoint.get("family") != "ddpm" or checkpoint.get("prediction") != "epsilon":
        raise ValueError("expected an epsilon-prediction DDPM checkpoint")
    model = build_mnist_ddpm_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    schedule = build_mnist_ddpm_schedule(config, device=device)
    evaluator = load_frozen_mnist_classifier(args.evaluator, device=device)

    labels = torch.arange(10, device=device).repeat_interleave(args.samples_per_class)
    source_generator = torch.Generator(device=device).manual_seed(args.source_seed)
    initial_noise = torch.randn(
        labels.shape[0],
        1,
        28,
        28,
        device=device,
        generator=source_generator,
    )
    ancestral_generator = torch.Generator(device=device).manual_seed(
        args.ancestral_seed
    )
    generated = sample_conditional_ddpm(
        model,
        schedule,
        labels,
        initial_noise,
        ancestral_generator=ancestral_generator,
    )

    metrics = evaluate_generated_mnist(
        evaluator,
        generated.samples,
        labels,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = args.output_dir / "samples.png"
    samples_path = args.output_dir / "samples.pt"
    summary_path = args.output_dir / "summary.json"
    save_grid(
        generated.samples,
        grid_path,
        samples_per_class=args.samples_per_class,
    )
    torch.save(
        {
            "samples": generated.samples.detach().cpu(),
            "requested_labels": labels.cpu(),
            "initial_noise": initial_noise.cpu(),
        },
        samples_path,
    )
    summary = {
        "family": "ddpm",
        "prediction": "epsilon",
        "device": args.device,
        "torch_version": str(torch.__version__),
        "evaluation_preprocessing": "clamp to [-1, 1]",
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "evaluator_sha256": file_sha256(args.evaluator),
        "source_seed": args.source_seed,
        "ancestral_seed": args.ancestral_seed,
        "num_timesteps": schedule.num_timesteps,
        "nfe": schedule.num_timesteps,
        "num_samples": labels.shape[0],
        "metrics": {
            "conditional_accuracy": metrics.conditional_accuracy,
            "requested_class_confidence": metrics.requested_class_confidence,
            "out_of_range_fraction": metrics.out_of_range_fraction,
            "confusion_matrix": metrics.confusion_matrix.tolist(),
        },
        "artifacts": {
            "grid": str(grid_path),
            "samples": str(samples_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"conditional_accuracy={metrics.conditional_accuracy:.4f} "
        f"confidence={metrics.requested_class_confidence:.4f}"
    )


if __name__ == "__main__":
    main()
