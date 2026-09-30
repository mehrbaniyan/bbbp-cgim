"""Pretrain CGIM using only the held-out pretraining objective loss.

Edit the defaults below or pass optional arguments. With no arguments, settings
are read from configs/pretrain.yaml:
    python src/pretrain.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import torch

from cgim.config import load_dataset_config, load_experiment, resolve_project_path
from cgim.data.loaders import build_pretrain_loaders
from cgim.training.pretrain import train_model
from cgim.utils import set_seed


PRETRAIN_CONFIG = "configs/pretrain.yaml"
DATASET_REGISTRY = "configs/datasets.yaml"
PRETRAIN_DATASET = "pubchem50k"
PROCESSED_CSV = None
CHECKPOINT_PATH = None
DEVICE = None


def run_pretraining(
    *,
    config_path=PRETRAIN_CONFIG,
    registry_path=DATASET_REGISTRY,
    dataset_name=PRETRAIN_DATASET,
    processed_csv=PROCESSED_CSV,
    checkpoint_path=CHECKPOINT_PATH,
    device_name=DEVICE,
    num_epochs=None,
    patience=None,
    batch_size=None,
    num_workers=None,
    embedding_dim=None,
    regression_dim=None,
    loss_name=None,
    temperature=None,
    sigma=None,
    learning_rate=None,
    scheduler_factor=None,
    scheduler_patience=None,
    gradient_clip=None,
    seed=None,
    best_model_path=None,
    last_checkpoint_path=None,
):
    experiment = load_experiment(config_path)
    dataset = load_dataset_config(dataset_name, registry_path)
    if processed_csv:
        dataset["processed_csv"] = resolve_project_path(processed_csv)
    if batch_size is not None:
        experiment["loader"]["batch_size"] = batch_size
    if num_workers is not None:
        experiment["loader"]["num_workers"] = num_workers

    run_seed = seed if seed is not None else experiment.get("sample_seed", 42)
    set_seed(run_seed)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    pretrain_loader, valid_loader, _, _, _ = build_pretrain_loaders(
        dataset,
        experiment,
    )
    training_checkpoint_path = resolve_project_path(checkpoint_path) if checkpoint_path else None

    model_config = experiment["model"]
    training = experiment["training"]
    best_path = (
        resolve_project_path(best_model_path)
        if best_model_path
        else training["best_encoder"]
    )
    last_path = (
        resolve_project_path(last_checkpoint_path)
        if last_checkpoint_path
        else training["last_checkpoint"]
    )
    return train_model(
        pretrain_loader,
        valid_loader,
        embedding_dim=(
            embedding_dim
            if embedding_dim is not None
            else model_config.get("embedding_dim", 512)
        ),
        num_epochs=(
            num_epochs if num_epochs is not None else training.get("epochs", 100)
        ),
        device=device,
        best_model_path=str(best_path),
        last_checkpoint_path=str(last_path),
        checkpoint_path=(
            str(training_checkpoint_path) if training_checkpoint_path else None
        ),
        regression_dim=(
            regression_dim
            if regression_dim is not None
            else model_config.get("regression_dim", 64)
        ),
        pretrained_backbone=model_config.get("pretrained_backbone", True),
        contrastive_loss=loss_name or training.get("contrastive_loss", "dclw"),
        temperature=(temperature if temperature is not None else training.get("temperature", 0.1)),
        sigma=sigma if sigma is not None else training.get("sigma", 0.5),
        learning_rate=(
            learning_rate
            if learning_rate is not None
            else training.get("learning_rate", 0.001)
        ),
        scheduler_factor=(
            scheduler_factor
            if scheduler_factor is not None
            else training.get("scheduler_factor", 0.5)
        ),
        scheduler_patience=(
            scheduler_patience
            if scheduler_patience is not None
            else training.get("scheduler_patience", 3)
        ),
        early_stopping_patience=(
            patience
            if patience is not None
            else training.get("early_stopping_patience", 10)
        ),
        gradient_clip=(
            gradient_clip
            if gradient_clip is not None
            else training.get("gradient_clip", 0.1)
        ),
        clusters=tuple(experiment.get("pseudo_labels", {}).get("clusters", (100, 1000))),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=PRETRAIN_CONFIG, help="pretraining YAML file")
    parser.add_argument("--registry", default=DATASET_REGISTRY, help="dataset registry YAML file")
    parser.add_argument("--dataset", default=PRETRAIN_DATASET, help="pretraining dataset name")
    parser.add_argument("--processed-csv", default=PROCESSED_CSV, help="processed molecule CSV")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="full training checkpoint")
    parser.add_argument("--best-model-path", help="output path for the best encoder")
    parser.add_argument("--last-checkpoint-path", help="output path for the latest full checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=DEVICE)
    parser.add_argument("--epochs", type=int, help="number of training epochs")
    parser.add_argument("--patience", type=int, help="early-stopping patience")
    parser.add_argument("--batch-size", type=int, help="mini-batch size")
    parser.add_argument("--num-workers", type=int, help="data loader workers")
    parser.add_argument("--embedding-dim", type=int, help="encoder embedding dimension")
    parser.add_argument("--regression-dim", type=int, help="TruncatedSVD regression dimension")
    parser.add_argument("--loss", dest="loss_name", choices=("dclw", "simclr"), help="contrastive loss")
    parser.add_argument("--temperature", type=float, help="contrastive temperature")
    parser.add_argument("--sigma", type=float, help="DCLW weighting sigma")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--scheduler-factor", type=float)
    parser.add_argument("--scheduler-patience", type=int)
    parser.add_argument("--gradient-clip", type=float)
    parser.add_argument("--seed", type=int, help="random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pretraining(
        config_path=args.config,
        registry_path=args.registry,
        dataset_name=args.dataset,
        processed_csv=args.processed_csv,
        checkpoint_path=args.checkpoint,
        device_name=args.device,
        num_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        embedding_dim=args.embedding_dim,
        regression_dim=args.regression_dim,
        loss_name=args.loss_name,
        temperature=args.temperature,
        sigma=args.sigma,
        learning_rate=args.learning_rate,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        best_model_path=args.best_model_path,
        last_checkpoint_path=args.last_checkpoint_path,
    )
