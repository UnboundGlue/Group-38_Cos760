"""Trainer for the CNN-LSTM authorship attribution model."""

from __future__ import annotations

import logging
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np

from .models import MetricsDict, TrainingHistory, TrainingDivergenceError

logger = logging.getLogger(__name__)


def _rebuild_train_loader_smaller_batch(train_loader: DataLoader, new_bs: int) -> DataLoader:
    """Rebuild the training DataLoader with a smaller batch, preserving other settings (Req 6.7)."""
    n_workers = getattr(train_loader, "num_workers", 0)
    kwargs: dict = {
        "dataset": train_loader.dataset,
        "batch_size": new_bs,
        "shuffle": True,
        "num_workers": n_workers,
        "pin_memory": getattr(train_loader, "pin_memory", False),
        "drop_last": getattr(train_loader, "drop_last", False),
        "collate_fn": train_loader.collate_fn,
    }
    if n_workers > 0:
        kwargs["persistent_workers"] = getattr(train_loader, "persistent_workers", False)
        kwargs["prefetch_factor"] = getattr(train_loader, "prefetch_factor", 2)
    return DataLoader(**kwargs)


class Trainer:
    """Manages the training loop, validation, early stopping, and checkpoint saving.

    Requirements covered:
        6.1 – Adam optimiser + gradient clipping (max_norm=1.0)
        6.2 – Validate after each epoch, compute macro-F1
        6.3 – Save checkpoint and reset patience counter on improvement
        6.4 – Early stopping when patience exhausted
        6.5 – Always terminate within max epochs
        6.6 – Raise TrainingDivergenceError on NaN loss
        6.7 – Halve batch size and retry on CUDA OOM
        6.8 – Return TrainingHistory with per-epoch loss and val metrics
    """

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
        patience: int,
        checkpoint_path: str = "artifacts/checkpoints/best_model.pt",
        *,
        weight_decay: float = 0.0,
        label_smoothing: float = 0.0,
        reduce_lr_on_plateau: bool = False,
        class_weights: torch.Tensor | None = None,
        lr_schedule: str | None = None,
        cosine_t0: int = 8,
    ) -> TrainingHistory:
        """Run the training loop.

        Args:
            model: The neural network to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Maximum number of training epochs (Req 6.5).
            lr: Learning rate for Adam optimiser (Req 6.1).
            patience: Early-stopping patience in epochs (Req 6.4).
            checkpoint_path: Where to save the best model checkpoint (Req 6.3).
            weight_decay: L2 regularisation for Adam (default 0 for backward compatibility).
            label_smoothing: CrossEntropyLoss label smoothing (0 disables).
            reduce_lr_on_plateau: If True, reduce learning rate when val macro-F1 plateaus
                (used only when ``lr_schedule`` is None; legacy).
            class_weights: Per-class loss weights (e.g. inverse frequency); must match num classes.
            lr_schedule: ``"none"`` | ``"plateau"`` | ``"cosine_restarts"`` | None.
                If None, ``plateau`` is used when ``reduce_lr_on_plateau`` is True, else ``none``.
                **cosine_restarts** uses :class:`CosineAnnealingWarmRestarts` to raise LR
                periodically and escape plateaus; steps once per epoch.
            cosine_t0: Period in epochs for the first restart (cosine_restarts only).

        Returns:
            TrainingHistory with per-epoch train losses and validation metrics (Req 6.8).

        Raises:
            TrainingDivergenceError: If NaN loss is detected (Req 6.6).
        """
        history = TrainingHistory()

        # Detect device from model parameters
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        # Req 6.1 – Adam optimiser
        optimiser = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        ce_kw: dict = {"label_smoothing": label_smoothing}
        if class_weights is not None:
            ce_kw["weight"] = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(**ce_kw)

        if lr_schedule is not None:
            mode = lr_schedule
        else:
            mode = "plateau" if reduce_lr_on_plateau else "none"
        if mode not in ("none", "plateau", "cosine_restarts"):
            raise ValueError(
                f"lr_schedule must be 'none', 'plateau', or 'cosine_restarts'; got {mode!r}"
            )

        scheduler: (
            torch.optim.lr_scheduler.ReduceLROnPlateau
            | torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
            | None
        ) = None
        schedule_kind: str = "none"
        if mode == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimiser,
                mode="max",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
            )
            schedule_kind = "plateau"
        elif mode == "cosine_restarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimiser, T_0=cosine_t0, T_mult=1, eta_min=1e-6
            )
            schedule_kind = "cosine_restarts"

        best_val_f1 = -1.0
        patience_counter = 0

        # Ensure checkpoint directory exists
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Req 6.5 – loop bounded by epochs
        for epoch in range(1, epochs + 1):
            epoch_loss = self._run_epoch(
                model=model,
                train_loader=train_loader,
                optimiser=optimiser,
                criterion=criterion,
                device=device,
                epoch=epoch,
                checkpoint_path=checkpoint_path,
            )

            history.train_losses.append(epoch_loss)

            # Req 6.2 – evaluate on validation set after each epoch
            val_metrics = self.evaluate(model, val_loader)
            history.val_metrics.append(val_metrics)

            lr_now = optimiser.param_groups[0]["lr"]
            logger.info(
                "Epoch %d/%d  loss=%.4f  val_f1=%.4f  lr=%.2e  (%s)",
                epoch, epochs, epoch_loss, val_metrics.f1_macro, lr_now, schedule_kind,
            )

            if scheduler is not None:
                if schedule_kind == "plateau":
                    scheduler.step(val_metrics.f1_macro)
                else:
                    scheduler.step()

            # Req 6.3 – save checkpoint and reset patience on improvement
            if val_metrics.f1_macro > best_val_f1:
                best_val_f1 = val_metrics.f1_macro
                torch.save(model.state_dict(), checkpoint_path)
                logger.info("Checkpoint saved to %s (val_f1=%.4f)", checkpoint_path, best_val_f1)
                patience_counter = 0
            else:
                patience_counter += 1

            # Req 6.4 – early stopping
            if patience_counter >= patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d exhausted).", epoch, patience
                )
                break

        return history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimiser: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        epoch: int,
        checkpoint_path: str,
    ) -> float:
        """Run one training epoch, handling CUDA OOM by halving batch size (Req 6.7).

        Returns the mean training loss for the epoch.
        """
        try:
            return self._train_one_epoch(
                model, train_loader, optimiser, criterion, device, epoch
            )
        except torch.cuda.OutOfMemoryError:
            # Req 6.7 – halve batch size and retry
            old_bs = train_loader.batch_size
            new_bs = max(1, old_bs // 2)
            logger.warning(
                "CUDA OOM at epoch %d. Halving batch size: %d → %d and retrying.",
                epoch, old_bs, new_bs,
            )
            new_loader = _rebuild_train_loader_smaller_batch(train_loader, new_bs)
            torch.cuda.empty_cache()
            return self._train_one_epoch(
                model, new_loader, optimiser, criterion, device, epoch
            )

    def _train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimiser: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        epoch: int,
    ) -> float:
        """Core training loop for a single epoch.

        Raises:
            TrainingDivergenceError: On NaN loss (Req 6.6).
        """
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(loader):
            # Support both (inputs, labels) tuples and dict-style batches
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[0], batch[1]
            else:
                inputs = batch["input_ids"]
                labels = batch["labels"]

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimiser.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)

            # Req 6.6 – detect NaN loss
            if math.isnan(loss.item()):
                msg = f"NaN loss detected at epoch {epoch}, batch {batch_idx}."
                logger.error(msg)
                raise TrainingDivergenceError(msg)

            loss.backward()

            # Req 6.1 – gradient clipping before parameter update
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def evaluate(self, model: nn.Module, loader: DataLoader) -> MetricsDict:
        """Evaluate *model* on *loader* and return classification metrics.

        Runs inference in no-gradient mode (Req 7.5).
        Uses macro-F1 as the primary metric for early stopping (Req 6.2).
        """
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []

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

        classes = np.unique(np.concatenate([y_true, y_pred]))

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_per = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
        f1_per_class = {int(c): float(f) for c, f in zip(classes, f1_per)}
        conf_mat = confusion_matrix(y_true, y_pred, labels=classes)

        return MetricsDict(
            accuracy=acc,
            precision_macro=prec,
            recall_macro=rec,
            f1_macro=f1_mac,
            f1_per_class=f1_per_class,
            confusion_matrix=conf_mat,
        )
