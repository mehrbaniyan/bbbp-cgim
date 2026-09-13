from __future__ import annotations

from torch.utils.data import DataLoader

from cgim.data.datasets import (
    Molecule3DDatasetUS,
    Molecule3DDatasetUSPre,
    collate_ft,
    collate_pre,
    default_transform,
)
from cgim.data.preparation import prepare_downstream_frame, prepare_pretrain_frame
from cgim.data.splitters import make_splitter


def _split_frame(df, config, smiles_column):
    split = config.get("split", {})
    if isinstance(split, str):
        name = split
        frac_train, frac_valid, frac_test = 0.8, 0.1, 0.1
        seed = config.get("split_seed")
    else:
        name = split.get("name", "scaffold")
        frac_train = split.get("train", 0.8)
        frac_valid = split.get("valid", 0.1)
        frac_test = split.get("test", 0.1)
        seed = split.get("seed")

    train_idx, valid_idx, test_idx = make_splitter(name).split(
        df,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test,
        smile_column=smiles_column,
        seed=seed,
    )
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[valid_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def build_finetune_loaders(dataset_config, *, processed_csv=None):
    csv_path = processed_csv or dataset_config["processed_csv"]
    smiles_column = dataset_config["smiles_column"]
    df, pca = prepare_downstream_frame(
        csv_path,
        smiles_column=smiles_column,
        fingerprint=dataset_config["fingerprint"],
        feature_dim=dataset_config.get("feature_dim", 16),
        drop_duplicate_smiles=dataset_config.get("drop_duplicate_smiles", False),
    )
    train_df, valid_df, test_df = _split_frame(df, dataset_config, smiles_column)
    transform = default_transform(tuple(dataset_config.get("image_size", (224, 224))))

    common = dict(
        transform=transform,
        label_column=dataset_config["label_column"],
        smiles_column=smiles_column,
        pca_components=pca,
        scaffold_pca_dim=dataset_config.get("feature_dim", 16),
        is_validation=True,
    )
    train_dataset = Molecule3DDatasetUS(train_df, **common)
    valid_dataset = Molecule3DDatasetUS(valid_df, **common)
    test_dataset = Molecule3DDatasetUS(test_df, **common)

    batch_size = dataset_config.get("batch_size", 64)
    workers = dataset_config.get("num_workers", 0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_ft, num_workers=workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_ft, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_ft, num_workers=workers)
    return train_loader, valid_loader, test_loader, (train_df, valid_df, test_df), pca


def build_pretrain_loaders(dataset_config, experiment_config, *, processed_csv=None):
    csv_path = processed_csv or dataset_config["processed_csv"]
    pseudo = experiment_config["pseudo_labels"]
    regression = experiment_config["regression_target"]
    df, svd, scaler = prepare_pretrain_frame(
        csv_path,
        smiles_column=dataset_config["smiles_column"],
        sample_size=experiment_config.get("sample_size", 10000),
        sample_seed=experiment_config.get("sample_seed", 42),
        clusters=tuple(pseudo.get("clusters", (10, 100))),
        cluster_seed=pseudo.get("random_state", 42),
        fingerprint_bits=regression.get("fingerprint_bits", 2048),
        radius=regression.get("radius", 2),
        components=regression.get("components", 64),
        n_iter=regression.get("n_iter", 10),
        svd_seed=regression.get("random_state", 0),
    )
    train_df, valid_df, _ = _split_frame(
        df, experiment_config, dataset_config["smiles_column"]
    )
    transform = default_transform(tuple(dataset_config.get("image_size", (224, 224))))
    clusters = pseudo.get("clusters", [10, 100])
    if list(clusters) != [10, 100]:
        raise ValueError("The model heads require pseudo-label clusters [10, 100].")

    common = dict(
        transform=transform,
        label_column1="cluster_10",
        label_column2="cluster_100",
        smiles_column=dataset_config["smiles_column"],
        svd_components=svd,
        scaffold_svd_dim=regression.get("components", 64),
    )
    train_dataset = Molecule3DDatasetUSPre(train_df, **common)
    valid_dataset = Molecule3DDatasetUSPre(valid_df, **common)
    loader = experiment_config.get("loader", {})
    batch_size = loader.get("batch_size", 64)
    workers = loader.get("num_workers", 0)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=loader.get("train_shuffle", True), collate_fn=collate_pre,
        num_workers=workers,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size,
        shuffle=loader.get("valid_shuffle", True), collate_fn=collate_pre,
        num_workers=workers,
    )
    return train_loader, valid_loader, (train_df, valid_df), svd, scaler

