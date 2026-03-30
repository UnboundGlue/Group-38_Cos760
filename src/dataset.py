"""DatasetLoader: load and split social media authorship datasets."""

from __future__ import annotations

import csv
import json
from collections import Counter

from sklearn.model_selection import StratifiedShuffleSplit

from src.models import InsufficientSamplesError, Split


class DatasetLoader:
    """Load CSV/JSON authorship datasets and produce stratified splits."""

    def __init__(self) -> None:
        self.author_map: dict[int, str] = {}       # int id → author name
        self.num_authors: int = 0
        self.samples_per_author: dict[int, int] = {}  # int id → count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str) -> tuple[list[str], list[int]]:
        """Load dataset from *path* (CSV or JSON) and return (texts, labels).

        CSV files must have columns ``text`` and ``author``.
        JSON files must be a list of objects with ``text`` and ``author`` keys.
        Author names are mapped to 0-indexed integer labels.
        """
        if path.endswith(".json"):
            raw = self._load_json(path)
        else:
            raw = self._load_csv(path)

        # Build author → id mapping (sorted for determinism)
        unique_authors = sorted({author for _, author in raw})
        name_to_id: dict[str, int] = {name: idx for idx, name in enumerate(unique_authors)}
        self.author_map = {idx: name for name, idx in name_to_id.items()}

        texts: list[str] = []
        labels: list[int] = []
        for text, author in raw:
            texts.append(text)
            labels.append(name_to_id[author])

        self.num_authors = len(unique_authors)
        counts = Counter(labels)
        self.samples_per_author = dict(counts)

        return texts, labels

    def split(
        self,
        texts: list[str],
        labels: list[int],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        min_samples: int = 10,
    ) -> tuple[Split, Split, Split]:
        """Stratified split into train / val / test partitions.

        Parameters
        ----------
        texts:       list of text strings
        labels:      corresponding integer author labels
        train_ratio: fraction of data for training (default 0.70)
        val_ratio:   fraction of data for validation (default 0.15)
        seed:        random seed for reproducibility (default 42)
        min_samples: minimum samples per author; raises
                     ``InsufficientSamplesError`` if any author falls below
                     this threshold (default 10)

        Returns
        -------
        (train, val, test) as ``Split`` namedtuples
        """
        # Validate minimum samples per author
        counts = Counter(labels)
        for author_id, count in counts.items():
            if count < min_samples:
                author_name = self.author_map.get(author_id, str(author_id))
                raise InsufficientSamplesError(
                    f"Author '{author_name}' (id={author_id}) has only {count} "
                    f"sample(s), which is below the minimum threshold of {min_samples}."
                )

        # --- Step 1: carve out the training split ---
        test_val_ratio = 1.0 - train_ratio  # fraction going to val+test
        sss_train = StratifiedShuffleSplit(
            n_splits=1, test_size=test_val_ratio, random_state=seed
        )
        train_idx, remainder_idx = next(sss_train.split(texts, labels))

        # --- Step 2: split remainder into val / test ---
        remainder_texts = [texts[i] for i in remainder_idx]
        remainder_labels = [labels[i] for i in remainder_idx]

        # val_ratio relative to the full dataset; compute relative to remainder
        val_relative = val_ratio / test_val_ratio
        sss_val = StratifiedShuffleSplit(
            n_splits=1, test_size=1.0 - val_relative, random_state=seed
        )
        val_idx_local, test_idx_local = next(
            sss_val.split(remainder_texts, remainder_labels)
        )

        # Map local indices back to original indices
        val_idx = [remainder_idx[i] for i in val_idx_local]
        test_idx = [remainder_idx[i] for i in test_idx_local]

        def _make_split(indices: list[int]) -> Split:
            return Split(
                texts=[texts[i] for i in indices],
                labels=[labels[i] for i in indices],
                author_map=dict(self.author_map),
            )

        return _make_split(list(train_idx)), _make_split(val_idx), _make_split(test_idx)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_csv(path: str) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append((row["text"], row["author"]))
        return rows

    @staticmethod
    def _load_json(path: str) -> list[tuple[str, str]]:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return [(item["text"], item["author"]) for item in data]
