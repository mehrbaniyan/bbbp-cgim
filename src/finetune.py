"""Fine-tune the pretrained CGIM encoder on a BBBP benchmark.

Edit the defaults below or pass optional arguments. Running without arguments
uses the MoleculeNet BBBP experiment configuration.
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
from cgim.models import SimCLRNetWithHead
from cgim.training.finetune import finetune_model_bce
from cgim.utils import set_seed


DATASET = "bbbp"
DATASET_REGISTRY = "configs/datasets.yaml"
FINETUNE_CONFIGS = {
    "bbbp": "configs/finetune_bbbp.yaml",
    "lightbbb": "configs/finetune_lightbbb.yaml",
    "b3db": "configs/finetune_b3db.yaml",
}
PROCESSED_CSV = None
PRETRAINED_ENCODER_PATH = None
CHECKPOINT_PATH = None
DEVICE = None


def run_finetuning(
    dataset_name=DATASET,
    *,
    config_path=None,
    registry_path=DATASET_REGISTRY,
    processed_csv=PROCESSED_CSV,
    pretrained_encoder_path=PRETRAINED_ENCODER_PATH,
    checkpoint_path=CHECKPOINT_PATH,
    device_name=DEVICE,
    num_epochs=None,
    batch_size=None,
    num_workers=None,
    learning_rate=None,
    scheduler_factor=None,
    scheduler_patience=None,
    patience=None,
    gradient_clip=None,
    seed=None,
    best_model_path=None,
    last_checkpoint_path=None,
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

    my_seed = seed if seed is not None else experiment.get("seed", 43)
    set_seed(my_seed)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_loader, valid_loader, test_loader, frames, _ = build_finetune_loaders(dataset)
    print(
        "Split sizes:",
        {name: len(frame) for name, frame in zip(("train", "valid", "test"), frames)},
    )

    model = SimCLRNetWithHead(
        embedding_dim=512,
        num_classes=experiment.get("num_classes", 1),
        device=device,
    ).to(device)

    encoder_path = pretrained_encoder_path or experiment.get("pretrained_encoder")
    if encoder_path:
        encoder_path = resolve_project_path(encoder_path)
    training_checkpoint_path = resolve_project_path(checkpoint_path) if checkpoint_path else None
    best_path = (
        resolve_project_path(best_model_path)
        if best_model_path
        else experiment["best_checkpoint"]
    )
    last_path = (
        resolve_project_path(last_checkpoint_path)
        if last_checkpoint_path
        else experiment["last_checkpoint"]
    )

    best_auc = finetune_model_bce(
        model,
        train_loader,
        valid_loader,
        num_epochs=(num_epochs if num_epochs is not None else experiment.get("epochs", 500)),
        device=device,
        pretrained_encoder_path=str(encoder_path) if encoder_path else None,
        best_model_path=str(best_path),
        last_model_path=str(last_path),
        checkpoint_path=(
            str(training_checkpoint_path) if training_checkpoint_path else None
        ),
        seed=my_seed,
        learning_rate=(learning_rate if learning_rate is not None else experiment.get("learning_rate", 0.0001)),
        scheduler_factor=(scheduler_factor if scheduler_factor is not None else experiment.get("scheduler_factor", 0.5)),
        scheduler_patience=(scheduler_patience if scheduler_patience is not None else experiment.get("scheduler_patience", 3)),
        early_stopping_patience=(patience if patience is not None else experiment.get("early_stopping_patience", 5)),
        gradient_clip=(gradient_clip if gradient_clip is not None else experiment.get("gradient_clip", 0.1)),
    )
    return model, test_loader, best_auc


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(FINETUNE_CONFIGS), default=DATASET, help="downstream dataset")
    parser.add_argument("--config", help="fine-tuning YAML file")
    parser.add_argument("--registry", default=DATASET_REGISTRY, help="dataset registry YAML file")
    parser.add_argument("--processed-csv", default=PROCESSED_CSV, help="processed molecule CSV")
    parser.add_argument("--pretrained-encoder", default=PRETRAINED_ENCODER_PATH, help="pretrained encoder weights")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="full training checkpoint")
    parser.add_argument("--best-model-path", help="output path for the best checkpoint")
    parser.add_argument("--last-checkpoint-path", help="output path for the latest checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=DEVICE)
    parser.add_argument("--epochs", type=int, help="number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, help="mini-batch size")
    parser.add_argument("--num-workers", type=int, help="data loader workers")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--scheduler-factor", type=float)
    parser.add_argument("--scheduler-patience", type=int)
    parser.add_argument("--patience", type=int, help="early-stopping patience")
    parser.add_argument("--gradient-clip", type=float)
    parser.add_argument("--seed", type=int, help="random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_finetuning(
        args.dataset,
        config_path=args.config,
        registry_path=args.registry,
        processed_csv=args.processed_csv,
        pretrained_encoder_path=args.pretrained_encoder,
        checkpoint_path=args.checkpoint,
        device_name=args.device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        patience=args.patience,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        best_model_path=args.best_model_path,
        last_checkpoint_path=args.last_checkpoint_path,
    )
