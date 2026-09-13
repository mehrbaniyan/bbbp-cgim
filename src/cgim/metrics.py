from __future__ import annotations

import numpy as np
import torch
from torchmetrics import AUROC, Precision, Recall, F1Score, Accuracy, CohenKappa, AveragePrecision
from tqdm.auto import tqdm


def evaluate_model(model, loader, device):
    """Evaluate binary model with single-logit output."""
    model.eval()
    all_logits, all_targets = [], []

    with torch.no_grad():
        for v1, v2, labels, *_ in loader:
            v1 = v1.to(device)
            _, _, logits = model(v1)
            all_logits.append(logits.squeeze(1).cpu())
            all_targets.append(labels)

    logits = torch.cat(all_logits).to(device)
    targets = torch.cat(all_targets).to(device)
    targets_float = targets.float()
    targets_int = targets.long()
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()

    return {
        'auroc': AUROC(task='binary').to(device)(probs, targets_float).item(),
        'aupr': AveragePrecision(task='binary').to(device)(probs, targets_int).item(),
        'precision': Precision(task='binary').to(device)(preds, targets_int).item(),
        'recall': Recall(task='binary').to(device)(preds, targets_int).item(),
        'f1': F1Score(task='binary').to(device)(preds, targets_int).item(),
        'accuracy': Accuracy(task='binary').to(device)(preds, targets_int).item(),
        'cohen_kappa': CohenKappa(task='binary', weights='quadratic').to(device)(preds, targets_int).item(),
    }


def evaluate_repeated(model, loaders, device, num_runs=1):
    results = {
        split: {metric: [] for metric in [
            'auroc', 'aupr', 'precision', 'recall', 'f1', 'accuracy', 'cohen_kappa'
        ]}
        for split in loaders
    }
    for _ in tqdm(range(num_runs), desc='Runs'):
        for split, loader in loaders.items():
            metrics = evaluate_model(model, loader, device)
            for key, value in metrics.items():
                results[split][key].append(value)
    return {
        split: {key: (np.mean(values), np.std(values)) for key, values in metrics.items()}
        for split, metrics in results.items()
    }

