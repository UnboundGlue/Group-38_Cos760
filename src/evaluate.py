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

from pathlib import Path
from typing import Sequence

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


def _eval_amp_dtype(device: torch.device, use_amp: bool) -> tuple[torch.dtype, bool]:
    """Return autocast dtype and whether AMP is enabled (CUDA only)."""
    if not use_amp or device.type != "cuda":
        return torch.float32, False
    return (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        True,
    )


def metrics_dict_to_jsonable(m: MetricsDict) -> dict:
    """Serialise ``MetricsDict`` for JSON (`confusion_matrix` → nested lists)."""
    return {
        "accuracy": m.accuracy,
        "precision_macro": m.precision_macro,
        "recall_macro": m.recall_macro,
        "f1_macro": m.f1_macro,
        "f1_per_class": {str(k): v for k, v in m.f1_per_class.items()},
        "confusion_matrix": m.confusion_matrix.tolist(),
    }


def evaluate_labels(y_true: np.ndarray, y_pred: np.ndarray) -> MetricsDict:
    """Same metrics as ``evaluate()`` but from parallel NumPy label arrays."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_per_arr = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    f1_per_class = {int(c): float(f) for c, f in zip(classes, f1_per_arr)}
    conf_mat = confusion_matrix(y_true, y_pred, labels=classes)

    return MetricsDict(
        accuracy=acc,
        precision_macro=prec,
        recall_macro=rec,
        f1_macro=f1_mac,
        f1_per_class=f1_per_class,
        confusion_matrix=conf_mat,
    )


def evaluate(model: nn.Module, loader: DataLoader, *, use_amp: bool | None = None) -> MetricsDict:
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
        use_amp: CUDA inference AMP; ``None`` (default) enables on CUDA, disables on CPU/MPS.

    Returns:
        MetricsDict with accuracy, macro-precision, macro-recall, macro-F1,
        per-class F1 (keyed by integer class index), and a confusion matrix.
    """
    # Detect device from model parameters; fall back to CPU if model has none.
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    use_amp_infer = device.type == "cuda" if use_amp is None else bool(use_amp)
    amp_dt, amp_on = _eval_amp_dtype(device, use_amp_infer)

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

            inputs = inputs.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dt, enabled=amp_on):
                logits = model(inputs)
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    return evaluate_labels(y_true, y_pred)


def ensemble_evaluate_majority_vote(
    model: nn.Module,
    checkpoint_paths: Sequence[str | Path],
    loader: DataLoader,
    device: torch.device,
    *,
    cuda_amp: bool = True,
) -> MetricsDict:
    """Ensemble by **plurality vote** per sample; ties broken by highest mean logit.

    Same behaviour as ``experiments/run_cnn_lstm`` multi-checkpoint evaluation — shared so
    ``load_cnn_checkpoint`` can score saved ensemble run directories against a dataset.
    """
    paths = [str(p) for p in checkpoint_paths]
    state_dicts = [torch.load(p, map_location="cpu") for p in paths]
    model = model.to(device)
    y_true: list[int] = []
    y_pred: list[int] = []
    amp_dt, amp_on = _eval_amp_dtype(device, cuda_amp)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[0], batch[1]
            else:
                inputs = batch["input_ids"]
                labels = batch["labels"]
            inputs = inputs.to(device, non_blocking=True)
            bsz = int(inputs.size(0))
            member_argmax: list[np.ndarray] = []
            logit_sum: torch.Tensor | None = None
            for sd in state_dicts:
                model.load_state_dict(sd)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dt, enabled=amp_on):
                    logits = model(inputs)
                logit_sum = logits if logit_sum is None else logit_sum + logits
                member_argmax.append(logits.argmax(dim=-1).detach().cpu().numpy())
            assert logit_sum is not None
            mean_logits = logit_sum / float(len(state_dicts))
            stack = np.stack(member_argmax, axis=0)
            for j in range(bsz):
                col = stack[:, j].astype(np.int64)
                vals, counts = np.unique(col, return_counts=True)
                max_count = int(counts.max())
                candidates = vals[counts == max_count].astype(np.int64)
                if candidates.size == 1:
                    y_pred.append(int(candidates[0]))
                else:
                    best = int(candidates[0])
                    best_s = float(mean_logits[j, best].item())
                    for c_int in candidates[1:]:
                        c_int = int(c_int)
                        s = float(mean_logits[j, c_int].item())
                        if s > best_s:
                            best_s = s
                            best = c_int
                    y_pred.append(best)
            y_true.extend(labels.tolist())
    return evaluate_labels(np.asarray(y_true), np.asarray(y_pred))
