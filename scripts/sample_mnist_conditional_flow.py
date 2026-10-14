from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from generative_models.data.mnist import denormalize_mnist_image
from generative_models.evals.mnist_generation import (
    evaluate_generated_mnist,
    load_frozen_mnist_classifier,
)
from generative_models.flow_matching.conditional import (
    sample_conditional_flow_ode,
)
from generative_models.training.mnist_flow_matching import (
    MNISTFlowTrainingConfig,
    build_mnist_flow_model,
)
from generative_models.viz.points import save_figure, use_clean_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/mnist_conditional_flow/checkpoint.pt"),
    )
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path("runs/mnist_classifier/checkpoint.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/mnist_conditional_flow_samples"),
    )
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument(
        "--solver",
        choices=("euler", "heun"),
        default="heun",
    )
    parser.add_argument("--source-seed", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


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
            axes[label, column].imshow(display[index, 0], cmap="gray")
            axes[label, column].axis("off")
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
    config = MNISTFlowTrainingConfig(**checkpoint["config"])
    model = build_mnist_flow_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    labels = torch.arange(10, device=device).repeat_interleave(args.samples_per_class)
    source_generator = torch.Generator(device=device).manual_seed(args.source_seed)
    source_noise = torch.randn(
        labels.shape[0],
        1,
        28,
        28,
        device=device,
        generator=source_generator,
    )
    generated = sample_conditional_flow_ode(
        model,
        labels,
        source_noise,
        num_steps=args.num_steps,
        solver=args.solver,
    )

    evaluator = load_frozen_mnist_classifier(args.evaluator, device=device)
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
            "source_noise": source_noise.cpu(),
        },
        samples_path,
    )
    calls_per_step = 1 if args.solver == "euler" else 2
    summary = {
        "family": "flow_matching",
        "prediction": "velocity",
        "path": "linear_noise_to_data",
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "evaluator_sha256": file_sha256(args.evaluator),
        "source_seed": args.source_seed,
        "solver": args.solver,
        "num_steps": args.num_steps,
        "nfe": calls_per_step * args.num_steps,
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
