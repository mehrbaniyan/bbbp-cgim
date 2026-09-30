from __future__ import annotations

import gc
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from cgim.losses import dclw_loss, simclr_loss
from cgim.models import SimCLRNetWithHeadPre
from cgim.utils import make_grad_scaler


def _contrastive_function(name, temperature, sigma):
    if name == "dclw":
        return lambda z1, z2: dclw_loss(
            z1, z2, temperature=temperature, sigma=sigma
        )
    if name == "simclr":
        return lambda z1, z2: simclr_loss(z1, z2, temperature=temperature)
    raise ValueError("contrastive_loss must be 'dclw' or 'simclr'")


def _pretraining_loss(model, batch, device, contrastive, criterion_cls, criterion_reg):
    view1, view2, labels1, labels2, scaffold_img, regression_vec = batch
    view1, view2 = view1.to(device), view2.to(device)
    labels1, labels2 = labels1.to(device), labels2.to(device)
    scaffold_img = scaffold_img.to(device)
    regression_vec = regression_vec.to(device)

    _, z1, logits_i1, logits_j1, rec1, reg1 = model(view1)
    _, z2, logits_i2, logits_j2, rec2, reg2 = model(view2)

    cls_loss_1 = 0.5 * (
        criterion_cls(logits_i1, labels1) + criterion_cls(logits_j1, labels2)
    )
    cls_loss_2 = 0.5 * (
        criterion_cls(logits_i2, labels1) + criterion_cls(logits_j2, labels2)
    )
    cls_loss = 0.5 * (cls_loss_1 + cls_loss_2)
    contrastive_loss = contrastive(z1, z2)
    reconstruction_loss = 0.5 * (
        criterion_reg(rec1, scaffold_img) + criterion_reg(rec2, scaffold_img)
    )
    regression_loss = 0.5 * (
        criterion_reg(reg1, regression_vec) + criterion_reg(reg2, regression_vec)
    )

    loss = contrastive_loss + cls_loss + reconstruction_loss + regression_loss
    components = {
        "contrastive": contrastive_loss.detach(),
        "classification": cls_loss.detach(),
        "reconstruction": reconstruction_loss.detach(),
        "regression": regression_loss.detach(),
    }
    return loss, components


def _average_epoch(
    model,
    loader,
    device,
    contrastive,
    criterion_cls,
    criterion_reg,
    *,
    optimizer=None,
    scaler=None,
    gradient_clip=0.1,
    description,
    disable_progress=False,
):
    is_training = optimizer is not None
    model.train(is_training)
    totals = {
        "loss": 0.0,
        "contrastive": 0.0,
        "classification": 0.0,
        "reconstruction": 0.0,
        "regression": 0.0,
    }

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc=description, disable=disable_progress):
            if is_training:
                optimizer.zero_grad()

            loss, components = _pretraining_loss(
                model,
                batch,
                device,
                contrastive,
                criterion_cls,
                criterion_reg,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {description.lower()} loss detected")

            if is_training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()

            totals["loss"] += loss.item()
            for name, value in components.items():
                totals[name] += value.item()

            if device.type == "cuda":
                torch.cuda.empty_cache()

    if len(loader) == 0:
        raise ValueError(f"{description} loader contains no batches")
    return {name: value / len(loader) for name, value in totals.items()}


def train_model(
    pretrain_loader,
    valid_loader,
    embedding_dim: int,
    num_epochs: int,
    device,
    best_model_path="best_3bpgim.pth",
    last_checkpoint_path="last_checkpoint.pth",
    checkpoint_path=None,
    disable_progress=False,
    regression_dim=64,
    pretrained_backbone=True,
    contrastive_loss="dclw",
    temperature=0.1,
    sigma=0.5,
    learning_rate=0.001,
    scheduler_factor=0.5,
    scheduler_patience=3,
    early_stopping_patience=10,
    gradient_clip=0.1,
    clusters=(100, 1000),
):
    """Pretrain using only train/validation self-supervised objective loss."""
    if early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be at least 1")
    model = SimCLRNetWithHeadPre(
        embedding_dim=embedding_dim,
        regression_dim=regression_dim,
        clusters=clusters,
        pretrained_backbone=pretrained_backbone,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
    )
    scaler = make_grad_scaler(device)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    contrastive = _contrastive_function(contrastive_loss, temperature, sigma)

    best_valid_loss = float("inf")
    patience_counter = 0
    start_epoch = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_valid_loss = checkpoint.get("best_valid_loss", float("inf"))
        patience_counter = checkpoint.get("patience_counter", 0)
        print(
            f"Resuming epoch {start_epoch}, "
            f"best_valid_loss={best_valid_loss:.4f}"
        )

    for epoch in range(start_epoch, num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} (pretrain) ---")
        train_metrics = _average_epoch(
            model,
            pretrain_loader,
            device,
            contrastive,
            criterion_cls,
            criterion_reg,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip=gradient_clip,
            description="Pretrain",
            disable_progress=disable_progress,
        )
        valid_metrics = _average_epoch(
            model,
            valid_loader,
            device,
            contrastive,
            criterion_cls,
            criterion_reg,
            description="Validation",
            disable_progress=disable_progress,
        )

        valid_loss = valid_metrics["loss"]
        previous_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr != previous_lr:
            print(f"Learning rate reduced to {current_lr:.6g}")

        improved = valid_loss < best_valid_loss
        if improved:
            best_valid_loss = valid_loss
            patience_counter = 0
            Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
            model.save_encoder(best_model_path)
            print(f"New best validation loss: {best_valid_loss:.4f}")
        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve: "
                f"{patience_counter}/{early_stopping_patience}"
            )

        print(
            "Train Loss: "
            f"{train_metrics['loss']:.4f} "
            f"(Contrastive: {train_metrics['contrastive']:.4f}, "
            f"Class: {train_metrics['classification']:.4f}, "
            f"Recon: {train_metrics['reconstruction']:.4f}, "
            f"Regression: {train_metrics['regression']:.4f})"
        )
        print(
            "Valid Loss: "
            f"{valid_metrics['loss']:.4f} "
            f"(Contrastive: {valid_metrics['contrastive']:.4f}, "
            f"Class: {valid_metrics['classification']:.4f}, "
            f"Recon: {valid_metrics['reconstruction']:.4f}, "
            f"Regression: {valid_metrics['regression']:.4f})"
        )

        Path(last_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_metrics["loss"],
                "valid_loss": valid_loss,
                "best_valid_loss": best_valid_loss,
                "patience_counter": patience_counter,
                "learning_rate": current_lr,
            },
            last_checkpoint_path,
        )

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print("Pretraining complete.")
    return best_valid_loss
