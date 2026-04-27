"""End-to-end CNN-LSTM authorship attribution pipeline.

Wires together:
    DatasetLoader → Preprocessor → SubwordTokeniser → CNNLSTMModel → Trainer → Evaluator

Requirements covered:
    9.1 – Reproducible results with a fixed random seed.
    9.2 – Save evaluation metrics to results/metrics.json.
    9.3 – Save trained tokeniser to artifacts/tokeniser.json.
    9.4 – Save model checkpoints to artifacts/checkpoints/best_model.pt.
    9.6 – Produce both CNN-LSTM and baseline evaluation results.

Usage example::

    python -m experiments.run_cnn_lstm --fetch-dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from functools import partial

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

# Ensure the Repository/src package is importable when the script is run
# directly from the Repository directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import DEFAULT_CHANCHAL_200_CSV, DatasetLoader
from src.evaluate import evaluate
from src.features import BaselineFeatureExtractor
from src.model import CNNLSTMModel
from src.models import ModelConfig
from src.preprocessing import Preprocessor
from src.tokeniser import SubwordTokeniser
from src import training_hardware
from src.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    """Fix all relevant random seeds for reproducibility (Req 9.1)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism defaults; on CUDA, run() may enable cuDNN benchmark for throughput.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _configure_cudnn_for_device(device: torch.device) -> None:
    """On GPU, prefer throughput over strict cudnn bit-for-bit repeatability."""
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def _dataloader_worker_seeded_init(base_seed: int, worker_id: int) -> None:
    """Per-worker PRNG (must be top-level: Windows spawn must pickle *worker_init_fn*)."""
    s = int(base_seed) + 1000 + int(worker_id)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _make_loader(
    token_ids: np.ndarray,
    labels: list[int],
    batch_size: int,
    shuffle: bool = False,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
    dataloader_seed: int | None = None,
) -> DataLoader:
    """Wrap encoded token IDs and labels in a DataLoader.

    When *dataloader_seed* is set and *shuffle* is True, the batch order is tied to
    that seed. With ``num_workers > 0``, workers get derived numpy/torch seeds.
    """
    x = torch.tensor(token_ids, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(x, y)
    opts: dict = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if shuffle and dataloader_seed is not None:
        g = torch.Generator()
        g.manual_seed(dataloader_seed)
        opts["generator"] = g
    if dataloader_seed is not None and num_workers > 0:
        # partial(top_level_fn, int) is picklable; nested def is not (Windows spawn / DataLoader)
        opts["worker_init_fn"] = partial(
            _dataloader_worker_seeded_init, int(dataloader_seed)
        )
    if num_workers > 0:
        opts["persistent_workers"] = True
        opts["prefetch_factor"] = 2
    return DataLoader(dataset, **opts)


def _balanced_class_weights_tensor(labels: list[int], num_classes: int) -> torch.Tensor:
    """sklearn balanced weights for present classes; absent classes stay 1.0."""
    y = np.asarray(labels, dtype=np.int64)
    present = np.unique(y)
    cw = compute_class_weight("balanced", classes=present, y=y)
    w = np.ones(num_classes, dtype=np.float32)
    for c, wt in zip(present, cw):
        w[int(c)] = float(wt)
    return torch.from_numpy(w)


def _metrics_to_dict(metrics) -> dict:
    """Convert a MetricsDict to a JSON-serialisable plain dict."""
    return {
        "accuracy": metrics.accuracy,
        "precision_macro": metrics.precision_macro,
        "recall_macro": metrics.recall_macro,
        "f1_macro": metrics.f1_macro,
        "f1_per_class": {str(k): v for k, v in metrics.f1_per_class.items()},
        "confusion_matrix": metrics.confusion_matrix.tolist(),
    }


def _compute_baselines(train_split, test_split, seed: int) -> dict[str, dict]:
    """Sklearn baselines on train → test (Req 9.6)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix as sk_cm,
    )
    import numpy as _np

    baseline_results: dict[str, dict] = {}
    for method in ("bow", "tfidf", "char_ngram", "word_ngram"):
        try:
            extractor = BaselineFeatureExtractor()
            X_train = extractor.fit_transform(train_split.texts, method=method)
            X_test = extractor.transform(test_split.texts)
            clf = LogisticRegression(max_iter=1000, random_state=seed)
            clf.fit(X_train, train_split.labels)
            y_pred = clf.predict(X_test)
            y_true = _np.array(test_split.labels)
            classes = _np.unique(_np.concatenate([y_true, y_pred]))
            f1_per = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
            baseline_results[method] = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
                "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "f1_per_class": {str(int(c)): float(f) for c, f in zip(classes, f1_per)},
                "confusion_matrix": sk_cm(y_true, y_pred, labels=classes).tolist(),
            }
            logger.info(
                "Baseline %-12s — accuracy=%.4f  f1_macro=%.4f",
                method, baseline_results[method]["accuracy"], baseline_results[method]["f1_macro"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Baseline '%s' failed: %s", method, exc)
            baseline_results[method] = {"error": str(exc)}
    return baseline_results


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train and evaluate the CNN-LSTM authorship attribution model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        default=DEFAULT_CHANCHAL_200_CSV,
        help=(
            "Path to dataset file (CSV or JSON). Default: Chanchal 200-tweets slice "
            "(more data per author; see DEFAULT_CHANCHAL_200_CSV in src/dataset.py)."
        ),
    )
    p.add_argument(
        "--fetch-dataset",
        action="store_true",
        help="If the file is missing, clone chanchalIITP/AuthorIdentification into data/ (needs git).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--vocab-size", type=int, default=10_000, dest="vocab_size",
                   help="Tokeniser vocabulary size.")
    p.add_argument("--embed-dim", type=int, default=256, dest="embed_dim",
                   help="Embedding dimension.")
    p.add_argument("--num-filters", type=int, default=256, dest="num_filters",
                   help="Number of CNN filters per kernel size.")
    p.add_argument("--lstm-hidden", type=int, default=384, dest="lstm_hidden",
                   help="LSTM hidden state size.")
    p.add_argument("--lstm-layers", type=int, default=2, dest="lstm_layers",
                   help="Number of stacked LSTM layers.")
    p.add_argument("--dropout", type=float, default=0.35, help="Dropout rate.")
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=384,
        dest="max_seq_len",
        help="Maximum token sequence length (longer captures more of each tweet).",
    )
    p.add_argument("--epochs", type=int, default=100, help="Maximum training epochs.")
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        dest="batch_size",
        help="Training batch size; 0 = set from GPU memory or CPU (default).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        dest="num_workers",
        help="DataLoader worker processes; -1 = from CPU count (default). Use 0 to disable workers.",
    )
    p.add_argument("--lr", type=float, default=8e-4, help="Learning rate.")
    p.add_argument("--patience", type=int, default=16, help="Early-stopping patience.")
    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        dest="weight_decay",
        help="Adam L2 weight decay (regularisation).",
    )
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        dest="label_smoothing",
        help="Cross-entropy label smoothing (0 = off).",
    )
    p.add_argument(
        "--lr-schedule",
        choices=("none", "plateau", "cosine_restarts"),
        default="plateau",
        dest="lr_schedule",
        help=(
            "Learning rate schedule. plateau = ReduceLROnPlateau on val macro-F1 (default). "
            "cosine_restarts = CosineAnnealingWarmRestarts (opt-in; LR cycles, can slow early progress)."
        ),
    )
    p.add_argument(
        "--cosine-t0",
        type=int,
        default=8,
        dest="cosine_t0",
        help="Epochs per cosine cycle before first warm restart (cosine_restarts only).",
    )
    p.add_argument(
        "--class-weight",
        choices=("balanced", "none"),
        default="balanced",
        dest="class_weight",
        help="balanced = inverse-frequency loss weights (helps rare authors).",
    )
    return p


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict:
    """Execute the full pipeline and return a results dict.

    Returns:
        dict with keys ``"cnn_lstm"`` and ``"baselines"``, each containing
        a metrics dict (Req 9.6).
    """
    _set_seed(args.seed)
    logger.info("Random seed: %d (split, model init, DataLoader, PYTHONHASHSEED)", args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _configure_cudnn_for_device(device)

    if device.type == "cuda":
        mem_gib = training_hardware.gpu_total_memory_gib(0)
        logger.info(
            "Using device: %s  (%s, ~%.1f GiB total)",
            device,
            torch.cuda.get_device_name(0),
            mem_gib or 0.0,
        )
    else:
        logger.info("Using device: %s", device)
        hint = training_hardware.cuda_build_hint()
        if hint is not None:
            logger.warning("%s", hint)

    pin_memory = device.type == "cuda"
    if args.batch_size <= 0:
        mem = training_hardware.gpu_total_memory_gib(0) if device.type == "cuda" else None
        args.batch_size = training_hardware.suggest_batch_size(
            use_cuda=device.type == "cuda", gpu_mem_gib=mem
        )
    if args.num_workers < 0:
        args.num_workers = training_hardware.suggest_num_workers()
    logger.info(
        "Training IO: batch_size=%d, num_workers=%d, pin_memory=%s",
        args.batch_size, args.num_workers, pin_memory,
    )

    logger.info("Loading dataset from %s …", args.dataset)
    loader = DatasetLoader()
    texts, labels = loader.load(args.dataset, fetch_if_missing=args.fetch_dataset)
    num_classes = loader.num_authors
    logger.info("Loaded %d samples across %d authors.", len(texts), num_classes)

    preprocessor = Preprocessor()
    texts = preprocessor.batch_clean(texts)
    paired = [(t, l) for t, l in zip(texts, labels) if t]
    if not paired:
        raise ValueError("All texts became empty after preprocessing.")
    texts, labels = zip(*paired)
    texts = list(texts)
    labels = list(labels)

    train_split, val_split, test_split = loader.split(
        texts, labels, seed=args.seed
    )
    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(train_split.texts), len(val_split.texts), len(test_split.texts),
    )

    logger.info("Training SubwordTokeniser (vocab_size=%d) …", args.vocab_size)
    tokeniser = SubwordTokeniser()
    tokeniser.train(train_split.texts, vocab_size=args.vocab_size)

    tokeniser_path = "artifacts/tokeniser.json"
    os.makedirs(os.path.dirname(tokeniser_path), exist_ok=True)
    tokeniser.save(tokeniser_path)
    logger.info("Tokeniser saved to %s", tokeniser_path)

    logger.info("Encoding text splits …")
    train_ids = tokeniser.batch_encode(train_split.texts, max_length=args.max_seq_len)
    val_ids = tokeniser.batch_encode(val_split.texts, max_length=args.max_seq_len)
    test_ids = tokeniser.batch_encode(test_split.texts, max_length=args.max_seq_len)

    train_loader = _make_loader(
        train_ids,
        train_split.labels,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        dataloader_seed=args.seed + 999_000,
    )
    val_loader = _make_loader(
        val_ids,
        val_split.labels,
        args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = _make_loader(
        test_ids,
        test_split.labels,
        args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model_config = ModelConfig(
        vocab_size=tokeniser.vocab_size(),
        embed_dim=args.embed_dim,
        num_filters=args.num_filters,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        num_classes=num_classes,
    )
    model = CNNLSTMModel(model_config).to(device)
    logger.info("Model built: %s", model_config)

    checkpoint_path = "artifacts/checkpoints/best_model.pt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    trainer = Trainer()
    if args.class_weight == "balanced":
        class_weights = _balanced_class_weights_tensor(train_split.labels, num_classes)
        logger.info("Using balanced per-class loss weights (train distribution).")
    else:
        class_weights = None
        logger.info("Class-weighted loss disabled (--class-weight none).")

    logger.info("Starting training (max_epochs=%d, patience=%d) …", args.epochs, args.patience)
    history = trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        reduce_lr_on_plateau=False,
        class_weights=class_weights,
        lr_schedule=args.lr_schedule,
        cosine_t0=args.cosine_t0,
    )
    logger.info("Training complete. Best val F1: %.4f", max(
        (m.f1_macro for m in history.val_metrics), default=0.0
    ))

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logger.info("Loaded best checkpoint from %s", checkpoint_path)

    cnn_lstm_metrics = evaluate(model, test_loader)
    logger.info(
        "CNN-LSTM test metrics — accuracy=%.4f  f1_macro=%.4f",
        cnn_lstm_metrics.accuracy, cnn_lstm_metrics.f1_macro,
    )

    baseline_results = _compute_baselines(train_split, test_split, args.seed)
    results = {
        "cnn_lstm": _metrics_to_dict(cnn_lstm_metrics),
        "baselines": baseline_results,
    }

    results_path = "results/metrics.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Metrics saved to %s", results_path)

    return results


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
