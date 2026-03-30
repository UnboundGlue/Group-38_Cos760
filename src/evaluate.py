"""Standalone evaluation module for the neural authorship attribution pipeline.

Provides an ``evaluate()`` function that computes classification metrics for any
trained model and DataLoader, independent of the Trainer class.

Requirements covered:
    7.1 – Compute accuracy, macro-precision, macro-recall, macro-F1, per-class F1,
           and confusion matrix.
    7.2 – All scalar metric values are in [0.0, 1.0].
    7.3 – Confusion matrix elements sum to the total number of evaluated samples.
    7.4 – Trace of confusion matrix divided by total samples equals accuracy.
    7.5 – Inference runs in no-gradient mode (torch.no_grad()).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import numpy as np

from .models import MetricsDict


def evaluate(model: nn.Module, loader: DataLoader) -> MetricsDict:
    """Compute classification metrics for *model* evaluated on *loader*.

    Steps:
    1. Set model to eval() mode.
    2. Run inference in torch.no_grad() mode (Req 7.5).
    3. Collect all predictions and ground-truth labels.
    4. Compute metrics using scikit-learn (Req 7.1).
    5. Return a MetricsDict (Req 7.2, 7.3, 7.4).

    Args:
        model:  A trained ``nn.Module`` whose ``forward()`` accepts batched
                token-ID tensors and returns logits of shape ``[B, C]``.
        loader: A DataLoader yielding ``(inputs, labels)`` tuples or dicts
                with ``"input_ids"`` and ``"labels"`` keys.

    Returns:
        MetricsDict with accuracy, macro-precision, macro-recall, macro-F1,
        per-class F1 (keyed by integer class index), and a confusion matrix.
    """
    # Detect device from model parameters; fall back to CPU if model has none.
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    # Step 1 – eval mode disables dropout / batch-norm training behaviour.
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []

    # Step 2 & 3 – no-gradient inference, collect predictions.
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[0], batch[1]
            else:
                inputs = batch["input_ids"]
                labels = batch["labels"]

            inputs = inputs.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Step 4 – compute metrics with scikit-learn.
    classes = np.unique(np.concatenate([y_true, y_pred]))

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_per_arr = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    f1_per_class: dict[int, float] = {
        int(c): float(f) for c, f in zip(classes, f1_per_arr)
    }
    conf_mat = confusion_matrix(y_true, y_pred, labels=classes)

    # Step 5 – return MetricsDict.
    return MetricsDict(
        accuracy=acc,
        precision_macro=prec,
        recall_macro=rec,
        f1_macro=f1_mac,
        f1_per_class=f1_per_class,
        confusion_matrix=conf_mat,
    )
