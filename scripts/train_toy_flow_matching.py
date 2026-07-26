from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from generative_models.training.toy_flow_matching import (
    ToyFlowMatchingTrainingConfig,
    train_toy_flow_matching
)
from generative_models.viz.points import save_figure, use_clean_style

def load_config(path: Path) -> ToyFlowMatchingTrainingConfig:
    raw_config: Mapping[str, object] = yaml.safe_load(path.read_text())
    return ToyFlowMatchingTrainingConfig.from_mapping(raw_config)

def save_loss_plot(losses: list[float], path: Path) -> Path:
    use_clean_style()
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(range(1, len(losses) + 1), losses)
    ax.set_title("Toy Flow Matching Training Loss")
    ax.set_xlabel("step")
    ax.set_ylabel("velocity prediction MSE")
    output_path = save_figure(fig, path)
    plt.close(fig)
    return output_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/toy_flow_matching.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/toy_flow_matching"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = train_toy_flow_matching(
        config=config,
        device=args.device,
    )

    args.output_dir.mkdir(parents=True,exist_ok=True)
    checkpoint_path = args.output_dir / "checkpoint.pt"
    summary_path = args.output_dir / "summary.json"
    loss_plot_path = args.output_dir / "loss.png"

    torch.save(
        {
            "config": asdict(result.config),
            "model_state_dict": result.model.state_dict(),
            "losses": result.losses,
        },
        checkpoint_path,
    )

    summary = {
        "config": asdict(result.config),
        "num_steps": len(result.losses),
        "initial_loss": result.losses[0],
        "final_loss": result.losses[-1],
        "min_loss": min(result.losses),
        "checkpoint_path": str(checkpoint_path),
        "loss_plot_path": str(loss_plot_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    save_loss_plot(result.losses, loss_plot_path)

    print(f"Wrote {checkpoint_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {loss_plot_path}")


if __name__ == "__main__":
    main()
