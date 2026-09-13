"""Generate a Grad-CAM visualization for a fine-tuned CGIM checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import torch

from cgim.config import load_experiment, resolve_project_path
from cgim.models import SimCLRNetWithHead
from cgim.plotting import save_gradcam


DATASET = "bbbp"
FINETUNE_CONFIGS = {
    "bbbp": "configs/finetune_bbbp.yaml",
    "lightbbb": "configs/finetune_lightbbb.yaml",
    "b3db": "configs/finetune_b3db.yaml",
}
IMAGE_PATH = "data/images/bbbp/molecule_0/img_0.png"
CHECKPOINT_PATH = None
OUTPUT_PREFIX = "outputs/bbbp/gradcam_overlay"
DEVICE = None
PLOT_FORMATS = ("svg", "png")


def run_gradcam(
    image_path=IMAGE_PATH,
    *,
    dataset_name=DATASET,
    config_path=None,
    checkpoint_path=CHECKPOINT_PATH,
    output_prefix=OUTPUT_PREFIX,
    device_name=DEVICE,
    plot_formats=PLOT_FORMATS,
):
    if dataset_name not in FINETUNE_CONFIGS and config_path is None:
        raise ValueError(f"DATASET must be one of {list(FINETUNE_CONFIGS)}")
    experiment = load_experiment(config_path or FINETUNE_CONFIGS[dataset_name])
    selected_checkpoint = resolve_project_path(
        checkpoint_path or experiment["best_checkpoint"]
    )
    selected_image = resolve_project_path(image_path)
    if not selected_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {selected_checkpoint}")
    if not selected_image.is_file():
        raise FileNotFoundError(f"Molecular image not found: {selected_image}")

    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = SimCLRNetWithHead(
        embedding_dim=512,
        num_classes=experiment.get("num_classes", 1),
        device=device,
        pretrained_backbone=False,
    ).to(device)
    checkpoint = torch.load(selected_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    result = save_gradcam(
        model,
        selected_image,
        device,
        resolve_project_path(output_prefix),
        formats=plot_formats,
    )
    print(
        f"Predicted class: {result['predicted_class']} | "
        f"probability: {result['probability']:.4f}"
    )
    for path in result["paths"]:
        print(f"Saved {path}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=IMAGE_PATH)
    parser.add_argument("--dataset", choices=tuple(FINETUNE_CONFIGS), default=DATASET)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=DEVICE)
    parser.add_argument("--plot-formats", nargs="+", choices=("svg", "png"), default=PLOT_FORMATS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_gradcam(
        args.image,
        dataset_name=args.dataset,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_prefix=args.output_prefix,
        device_name=args.device,
        plot_formats=args.plot_formats,
    )
