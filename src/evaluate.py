"""Evaluate a fine-tuned CGIM checkpoint.

Edit the defaults below or pass optional arguments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import torch

from cgim.config import load_dataset_config, load_experiment, resolve_project_path
from cgim.data.loaders import build_finetune_loaders
from cgim.metrics import evaluate_repeated
from cgim.models import SimCLRNetWithHead


DATASET = "bbbp"
DATASET_REGISTRY = "configs/datasets.yaml"
FINETUNE_CONFIGS = {
    "bbbp": "configs/finetune_bbbp.yaml",
    "lightbbb": "configs/finetune_lightbbb.yaml",
    "b3db": "configs/finetune_b3db.yaml",
}
PROCESSED_CSV = None
CHECKPOINT_PATH = None
DEVICE = None
NUM_RUNS = 1


def run_evaluation(
    dataset_name=DATASET,
    *,
    config_path=None,
    registry_path=DATASET_REGISTRY,
    processed_csv=PROCESSED_CSV,
    checkpoint_path=CHECKPOINT_PATH,
    device_name=DEVICE,
    num_runs=NUM_RUNS,
    batch_size=None,
    num_workers=None,
):
    if dataset_name not in FINETUNE_CONFIGS and config_path is None:
        raise ValueError(f"DATASET must be one of {list(FINETUNE_CONFIGS)}")

    experiment = load_experiment(config_path or FINETUNE_CONFIGS[dataset_name])
    dataset = load_dataset_config(dataset_name, registry_path)
    if processed_csv:
        dataset["processed_csv"] = resolve_project_path(processed_csv)
    if batch_size is not None:
        dataset["batch_size"] = batch_size
    if num_workers is not None:
        dataset["num_workers"] = num_workers

    selected_checkpoint = checkpoint_path or experiment["best_checkpoint"]
    selected_checkpoint = resolve_project_path(selected_checkpoint)
    if not selected_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {selected_checkpoint}")

    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_loader, valid_loader, test_loader, _, _ = build_finetune_loaders(dataset)
    model = SimCLRNetWithHead(
        embedding_dim=512,
        num_classes=experiment.get("num_classes", 1),
        device=device,
        pretrained_backbone=False,
    ).to(device)

    checkpoint = torch.load(selected_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loaders = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
    }
    summary = evaluate_repeated(model, loaders, device, num_runs=num_runs)

    for split, metrics in summary.items():
        print(f"\n{split.capitalize()}:")
        for metric, (mean, std) in metrics.items():
            print(f"{metric.upper()}: {mean} +/- {std}")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(FINETUNE_CONFIGS), default=DATASET, help="downstream dataset")
    parser.add_argument("--config", help="fine-tuning YAML file")
    parser.add_argument("--registry", default=DATASET_REGISTRY, help="dataset registry YAML file")
    parser.add_argument("--processed-csv", default=PROCESSED_CSV, help="processed molecule CSV")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="fine-tuned checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=DEVICE)
    parser.add_argument("--batch-size", type=int, help="mini-batch size")
    parser.add_argument("--num-workers", type=int, help="data loader workers")
    parser.add_argument("--num-runs", type=int, default=NUM_RUNS, help="number of repeated evaluations")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        args.dataset,
        config_path=args.config,
        registry_path=args.registry,
        processed_csv=args.processed_csv,
        checkpoint_path=args.checkpoint,
        device_name=args.device,
        num_runs=args.num_runs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
