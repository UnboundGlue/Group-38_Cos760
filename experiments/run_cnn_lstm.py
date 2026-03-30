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

    python -m experiments.run_cnn_lstm --dataset data/posts.csv --epochs 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure the Repository/src package is importable when the script is run
# directly from the Repository directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import DatasetLoader
from src.evaluate import evaluate
from src.features import BaselineFeatureExtractor
from src.model import CNNLSTMModel
from src.models import ModelConfig, TrainingConfig
from src.preprocessing import Preprocessor
from src.tokeniser import SubwordTokeniser
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _make_loader(
    token_ids: np.ndarray,
    labels: list[int],
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    """Wrap encoded token IDs and labels in a DataLoader."""
    x = torch.tensor(token_ids, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train and evaluate the CNN-LSTM authorship attribution model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", required=True, help="Path to dataset file (CSV or JSON).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--vocab-size", type=int, default=10_000, dest="vocab_size",
                   help="Tokeniser vocabulary size.")
    p.add_argument("--embed-dim", type=int, default=128, dest="embed_dim",
                   help="Embedding dimension.")
    p.add_argument("--num-filters", type=int, default=128, dest="num_filters",
                   help="Number of CNN filters per kernel size.")
    p.add_argument("--lstm-hidden", type=int, default=256, dest="lstm_hidden",
                   help="LSTM hidden state size.")
    p.add_argument("--lstm-layers", type=int, default=2, dest="lstm_layers",
                   help="Number of stacked LSTM layers.")
    p.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")
    p.add_argument("--max-seq-len", type=int, default=256, dest="max_seq_len",
                   help="Maximum token sequence length.")
    p.add_argument("--epochs", type=int, default=50, help="Maximum training epochs.")
    p.add_argument("--batch-size", type=int, default=64, dest="batch_size",
                   help="Training batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p.add_argument("--patience", type=int, default=5, help="Early-stopping patience.")
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    logger.info("Loading dataset from %s …", args.dataset)
    loader = DatasetLoader()
    texts, labels = loader.load(args.dataset)
    num_classes = loader.num_authors
    logger.info("Loaded %d samples across %d authors.", len(texts), num_classes)

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    preprocessor = Preprocessor()
    texts = preprocessor.batch_clean(texts)

    # Remove empty texts produced by cleaning
    paired = [(t, l) for t, l in zip(texts, labels) if t]
    if not paired:
        raise ValueError("All texts became empty after preprocessing.")
    texts, labels = zip(*paired)
    texts = list(texts)
    labels = list(labels)

    # ------------------------------------------------------------------
    # 3. Split dataset
    # ------------------------------------------------------------------
    train_split, val_split, test_split = loader.split(
        texts, labels, seed=args.seed
    )
    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(train_split.texts), len(val_split.texts), len(test_split.texts),
    )

    # ------------------------------------------------------------------
    # 4. Train subword tokeniser on training corpus only (Req 3.8)
    # ------------------------------------------------------------------
    logger.info("Training SubwordTokeniser (vocab_size=%d) …", args.vocab_size)
    tokeniser = SubwordTokeniser()
    tokeniser.train(train_split.texts, vocab_size=args.vocab_size)

    # Persist tokeniser (Req 9.3)
    tokeniser_path = "artifacts/tokeniser.json"
    os.makedirs(os.path.dirname(tokeniser_path), exist_ok=True)
    tokeniser.save(tokeniser_path)
    logger.info("Tokeniser saved to %s", tokeniser_path)

    # ------------------------------------------------------------------
    # 5. Encode splits
    # ------------------------------------------------------------------
    logger.info("Encoding text splits …")
    train_ids = tokeniser.batch_encode(train_split.texts, max_length=args.max_seq_len)
    val_ids = tokeniser.batch_encode(val_split.texts, max_length=args.max_seq_len)
    test_ids = tokeniser.batch_encode(test_split.texts, max_length=args.max_seq_len)

    train_loader = _make_loader(train_ids, train_split.labels, args.batch_size, shuffle=True)
    val_loader = _make_loader(val_ids, val_split.labels, args.batch_size)
    test_loader = _make_loader(test_ids, test_split.labels, args.batch_size)

    # ------------------------------------------------------------------
    # 6. Build CNN-LSTM model
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    checkpoint_path = "artifacts/checkpoints/best_model.pt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    trainer = Trainer()
    logger.info("Starting training (max_epochs=%d, patience=%d) …", args.epochs, args.patience)
    history = trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
    )
    logger.info("Training complete. Best val F1: %.4f", max(
        (m.f1_macro for m in history.val_metrics), default=0.0
    ))

    # ------------------------------------------------------------------
    # 8. Load best checkpoint and evaluate on test set
    # ------------------------------------------------------------------
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logger.info("Loaded best checkpoint from %s", checkpoint_path)

    cnn_lstm_metrics = evaluate(model, test_loader)
    logger.info(
        "CNN-LSTM test metrics — accuracy=%.4f  f1_macro=%.4f",
        cnn_lstm_metrics.accuracy, cnn_lstm_metrics.f1_macro,
    )

    # ------------------------------------------------------------------
    # 9. Baseline comparison (Req 9.6)
    # ------------------------------------------------------------------
    logger.info("Computing baseline features …")
    baseline_results: dict[str, dict] = {}
    for method in ("bow", "tfidf", "char_ngram", "word_ngram"):
        try:
            extractor = BaselineFeatureExtractor()
            X_train = extractor.fit_transform(train_split.texts, method=method)
            X_test = extractor.transform(test_split.texts)

            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (
                accuracy_score, f1_score, precision_score,
                recall_score, confusion_matrix as sk_cm,
            )
            import numpy as _np

            clf = LogisticRegression(max_iter=1000, random_state=args.seed)
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

    # ------------------------------------------------------------------
    # 10. Save results (Req 9.2)
    # ------------------------------------------------------------------
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
