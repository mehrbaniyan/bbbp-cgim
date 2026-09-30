from __future__ import annotations

import gc
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import AUROC
from tqdm.auto import tqdm

from cgim.utils import make_grad_scaler


def finetune_model_bce(
    model: torch.nn.Module,
    train_loader,
    valid_loader,
    num_epochs: int,
    device: torch.device,
    pretrained_encoder_path: str = None,
    best_model_path: str = 'finetune_best_checkpoint.pth',
    last_model_path: str = 'finetune_last_checkpoint.pth',
    checkpoint_path: str = None,
    disable_progress: bool = False,
    seed: int = 42,
    learning_rate: float = 1e-3,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 3,
    early_stopping_patience: int = 10,
    gradient_clip: float = 0.1,
):
    """Fine-tune a single-logit classifier with BCE loss."""
    model.to(device)
    if pretrained_encoder_path is not None:
        model.load_encoder(pretrained_encoder_path, map_location=device, device=device)
        print(f"Loaded pretrained encoder from {pretrained_encoder_path}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=scheduler_factor, patience=scheduler_patience
    )
    scaler = make_grad_scaler(device)
    criterion_cls = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    patience, patience_counter = early_stopping_patience, 0
    start_epoch = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch']
        best_val_auc = ckpt.get('best_val_auc', 0.0)
        patience_counter = ckpt.get('patience_counter', 0)
        print(f"Resuming fine-tune from epoch {start_epoch}, best_val_auc={best_val_auc:.4f}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} - Fine-tune")
        model.train()
        total_loss = 0.0
        for view1, view2, labels, scaffold_img, scaffold_pca in tqdm(
                train_loader, desc="Train", disable=disable_progress):
            view1, view2 = view1.to(device), view2.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            _, _, logits1 = model(view1)
            _, _, logits2 = model(view2)
            logit1 = logits1.squeeze(1)
            logit2 = logits2.squeeze(1)
            loss_cls = 0.5 * (criterion_cls(logit1, labels)
                              + criterion_cls(logit2, labels))
            loss = loss_cls

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        avg_train_loss = total_loss / len(train_loader)
        print(f"  Train Loss: {avg_train_loss:.4f}")

        model.eval()
        auroc = AUROC(task='binary').to(device)
        with torch.no_grad():
            for view1, _, labels, _, _ in tqdm(
                    valid_loader, desc="Valid", disable=disable_progress):
                view1 = view1.to(device)
                labels = labels.float().to(device)
                _, _, logits = model(view1)
                logit = logits.squeeze(1)
                probs = torch.sigmoid(logit)
                auroc.update(probs, labels)

        val_auc = auroc.compute().item()
        print(f"  Valid AUROC: {val_auc:.4f}")
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_auc': best_val_auc,
                'patience_counter': patience_counter
            }, best_model_path)
        else:
            patience_counter += 1
            print(f"  No improvement - Patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping.")
                break

        Path(last_model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_auc': best_val_auc,
            'patience_counter': patience_counter
        }, last_model_path)

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print("Fine-tuning complete.")
    return best_val_auc
