"""Generate molecular-image data for pretraining or downstream datasets.

Edit the defaults below or pass optional arguments. Running without arguments
generates the configured PubChem pretraining dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from cgim.config import load_dataset_config, resolve_project_path
from cgim.data.generator import DatasetGenerator


DATASET_REGISTRY = "configs/datasets.yaml"
DATASETS = ["pubchem50k"]
NUM_VIEWS = 120
N_CONFORMERS = 5


def generate_dataset(
    dataset_name: str,
    *,
    registry_path=DATASET_REGISTRY,
    raw_csv=None,
    processed_csv=None,
    image_root=None,
    smiles_column=None,
    num_views=NUM_VIEWS,
    n_conformers=N_CONFORMERS,
    validation_only=False,
    image_size=None,
):
    if num_views < 1 or n_conformers < 1:
        raise ValueError("num_views and n_conformers must both be at least 1")
    config = load_dataset_config(dataset_name, registry_path)
    input_csv = resolve_project_path(raw_csv) if raw_csv else config["raw_csv"]
    output_csv = (
        resolve_project_path(processed_csv)
        if processed_csv
        else config["processed_csv"]
    )
    output_root = (
        resolve_project_path(image_root) if image_root else config["image_root"]
    )

    if not input_csv.is_file():
        raise FileNotFoundError(
            f"Raw CSV not found: {input_csv}\n"
            "Place the dataset at the configured path or pass --raw-csv."
        )

    generator = DatasetGenerator(
        input_csv=str(input_csv),
        output_csv=str(output_csv),
        base_folder=str(output_root),
        dataset_name=config["dataset_name"],
        smile_column=smiles_column or config["smiles_column"],
        image_size=tuple(image_size or config.get("image_size", (224, 224))),
        valid=validation_only,
        n_conformers=n_conformers,
        num_views=num_views,
        path_root=str(PROJECT_ROOT),
    )
    dataframe = generator.generate_dataset()
    failed = int(dataframe["image_path"].isna().sum())
    print(f"{dataset_name}: {len(dataframe)} molecules processed; {failed} failed.")
    return dataframe


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, help="dataset names from the registry")
    parser.add_argument("--registry", default=DATASET_REGISTRY, help="dataset registry YAML file")
    parser.add_argument("--raw-csv", help="raw SMILES CSV; valid with one dataset")
    parser.add_argument("--processed-csv", help="output CSV; valid with one dataset")
    parser.add_argument("--image-root", help="directory for rendered molecular images")
    parser.add_argument("--smiles-column", help="SMILES column; valid with one dataset")
    parser.add_argument("--num-views", type=int, default=NUM_VIEWS, help="views per molecule")
    parser.add_argument("--n-conformers", type=int, default=N_CONFORMERS, help="conformers per molecule")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), help="rendered image size")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Generate one deterministic view per molecule.",
    )
    args = parser.parse_args()
    if len(args.datasets) > 1 and any(
        (args.raw_csv, args.processed_csv, args.smiles_column)
    ):
        parser.error("CSV and column overrides require exactly one dataset")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    for dataset in arguments.datasets:
        generate_dataset(
            dataset,
            registry_path=arguments.registry,
            raw_csv=arguments.raw_csv,
            processed_csv=arguments.processed_csv,
            image_root=arguments.image_root,
            smiles_column=arguments.smiles_column,
            num_views=arguments.num_views,
            n_conformers=arguments.n_conformers,
            validation_only=arguments.validation_only,
            image_size=arguments.image_size,
        )
