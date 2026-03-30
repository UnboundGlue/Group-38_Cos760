"""Baseline-only authorship attribution pipeline.

Wires together:
    DatasetLoader → Preprocessor → BaselineFeatureExtractor → SVM/LogReg → Evaluator

For each feature method (bow, tfidf, char_ngram, word_ngram) and each
classifier (LinearSVC, LogisticRegression), the script fits on the training
split and evaluates on the test split, then saves all metrics to a JSON file.

Requirements covered:
    9.2 – Save evaluation metrics to results/metrics.json.
    9.6 – Produce baseline evaluation results for direct comparison.

Usage example::

    python -m experiments.run_baselines --dataset data/posts.csv
    python -m experiments.run_baselines --dataset data/posts.csv --seed 0 --output results/baselines.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys

import numpy as np

# Ensure the Repository/src package is importable when the script is run
# directly from the Repository directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import DatasetLoader
from src.features import BaselineFeatureExtractor
from src.preprocessing import Preprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_METHODS = ("bow", "tfidf", "char_ngram", "word_ngram")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility (Req 9.1)."""
    random.seed(seed)
    np.random.seed(seed)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a JSON-serialisable metrics dict for the given predictions."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1_per_arr = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_class": {str(int(c)): float(f) for c, f in zip(classes, f1_per_arr)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    }


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train and evaluate baseline classifiers for authorship attribution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", required=True, help="Path to dataset file (CSV or JSON).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--output",
        default="results/metrics.json",
        help="Path to save the metrics JSON file.",
    )
    return p


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    """Execute the baseline pipeline and return a results dict.

    Returns:
        Nested dict keyed by method → classifier → metrics.
    """
    _set_seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    logger.info("Loading dataset from %s …", args.dataset)
    loader = DatasetLoader()
    texts, labels = loader.load(args.dataset)
    logger.info("Loaded %d samples across %d authors.", len(texts), loader.num_authors)

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
    train_split, _val_split, test_split = loader.split(
        texts, labels, seed=args.seed
    )
    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(train_split.texts), len(_val_split.texts), len(test_split.texts),
    )

    y_test = np.array(test_split.labels)

    # ------------------------------------------------------------------
    # 4. Baseline feature extraction + classification
    # ------------------------------------------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    classifiers = {
        "svm": LinearSVC(max_iter=2000, random_state=args.seed),
        "logreg": LogisticRegression(max_iter=1000, random_state=args.seed),
    }

    results: dict[str, dict] = {}

    for method in _METHODS:
        results[method] = {}
        logger.info("Extracting features: %s …", method)

        try:
            extractor = BaselineFeatureExtractor(random_seed=args.seed)
            X_train = extractor.fit_transform(train_split.texts, method=method)
            X_test = extractor.transform(test_split.texts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Feature extraction failed for '%s': %s", method, exc)
            for clf_name in classifiers:
                results[method][clf_name] = {"error": str(exc)}
            continue

        for clf_name, clf in classifiers.items():
            # Re-instantiate to avoid state leakage between methods
            if clf_name == "svm":
                clf = LinearSVC(max_iter=2000, random_state=args.seed)
            else:
                clf = LogisticRegression(max_iter=1000, random_state=args.seed)

            try:
                clf.fit(X_train, train_split.labels)
                y_pred = clf.predict(X_test)
                metrics = _compute_metrics(y_test, np.array(y_pred))
                results[method][clf_name] = metrics
                logger.info(
                    "%-12s / %-6s — accuracy=%.4f  f1_macro=%.4f",
                    method, clf_name,
                    metrics["accuracy"], metrics["f1_macro"],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Classifier '%s' failed for method '%s': %s", clf_name, method, exc
                )
                results[method][clf_name] = {"error": str(exc)}

    # ------------------------------------------------------------------
    # 5. Save results (Req 9.2)
    # ------------------------------------------------------------------
    output_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Baseline metrics saved to %s", output_path)

    return results


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
