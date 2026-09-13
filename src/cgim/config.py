from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_dataset_config(
    name: str,
    registry_path: str | Path = "configs/datasets.yaml",
) -> dict[str, Any]:
    registry = load_yaml(registry_path)
    datasets = registry.get("datasets", {})
    if name not in datasets:
        choices = ", ".join(sorted(datasets))
        raise KeyError(f"Unknown dataset '{name}'. Available datasets: {choices}")

    config = deepcopy(registry.get("defaults", {}))
    config.update(deepcopy(datasets[name]))
    config["name"] = name

    for key in ("raw_csv", "processed_csv", "image_root"):
        if key in config:
            config[key] = resolve_project_path(config[key])
    return config


def load_experiment(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    for key in ("pretrained_encoder", "best_checkpoint", "last_checkpoint"):
        if config.get(key):
            config[key] = resolve_project_path(config[key])
    training = config.get("training")
    if isinstance(training, dict):
        for key in ("best_encoder", "last_checkpoint"):
            if training.get(key):
                training[key] = resolve_project_path(training[key])
    return config

