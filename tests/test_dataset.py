"""Unit tests for DatasetLoader (Task 2.1)."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter

import pytest

from src.dataset import DatasetLoader
from src.models import InsufficientSamplesError, Split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(path: str, rows: list[tuple[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["text", "author"])
        writer.writeheader()
        for text, author in rows:
            writer.writerow({"text": text, "author": author})


def _make_json(path: str, rows: list[tuple[str, str]]) -> None:
    data = [{"text": t, "author": a} for t, a in rows]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def _synthetic_rows(n_authors: int = 3, samples_per: int = 15) -> list[tuple[str, str]]:
    rows = []
    for a in range(n_authors):
        for s in range(samples_per):
            rows.append((f"text from author {a} sample {s}", f"author_{a}"))
    return rows


# ---------------------------------------------------------------------------
# load() — CSV
# ---------------------------------------------------------------------------

class TestLoadCSV:
    def test_returns_texts_and_labels(self, tmp_path):
        rows = _synthetic_rows(3, 5)
        p = str(tmp_path / "data.csv")
        _make_csv(p, rows)
        loader = DatasetLoader()
        texts, labels = loader.load(p)
        assert len(texts) == len(rows)
        assert len(labels) == len(rows)

    def test_labels_are_zero_indexed(self, tmp_path):
        rows = _synthetic_rows(4, 5)
        p = str(tmp_path / "data.csv")
        _make_csv(p, rows)
        loader = DatasetLoader()
        _, labels = loader.load(p)
        assert set(labels) == {0, 1, 2, 3}

    def test_num_authors_set(self, tmp_path):
        rows = _synthetic_rows(5, 5)
        p = str(tmp_path / "data.csv")
        _make_csv(p, rows)
        loader = DatasetLoader()
        loader.load(p)
        assert loader.num_authors == 5

    def test_samples_per_author_set(self, tmp_path):
        rows = _synthetic_rows(3, 7)
        p = str(tmp_path / "data.csv")
        _make_csv(p, rows)
        loader = DatasetLoader()
        loader.load(p)
        assert all(v == 7 for v in loader.samples_per_author.values())

    def test_author_map_populated(self, tmp_path):
        rows = _synthetic_rows(3, 5)
        p = str(tmp_path / "data.csv")
        _make_csv(p, rows)
        loader = DatasetLoader()
        loader.load(p)
        assert len(loader.author_map) == 3
        assert all(isinstance(k, int) for k in loader.author_map)
        assert all(isinstance(v, str) for v in loader.author_map.values())


# ---------------------------------------------------------------------------
# load() — JSON
# ---------------------------------------------------------------------------

class TestLoadJSON:
    def test_json_returns_same_count(self, tmp_path):
        rows = _synthetic_rows(2, 10)
        p = str(tmp_path / "data.json")
        _make_json(p, rows)
        loader = DatasetLoader()
        texts, labels = loader.load(p)
        assert len(texts) == len(rows)
        assert len(labels) == len(rows)

    def test_json_labels_zero_indexed(self, tmp_path):
        rows = _synthetic_rows(3, 5)
        p = str(tmp_path / "data.json")
        _make_json(p, rows)
        loader = DatasetLoader()
        _, labels = loader.load(p)
        assert set(labels) == {0, 1, 2}


# ---------------------------------------------------------------------------
# split()
# ---------------------------------------------------------------------------

class TestSplit:
    def _loader_with_data(self, tmp_path, n_authors=3, samples_per=15):
        rows = _synthetic_rows(n_authors, samples_per)
        p = str(tmp_path / "data.csv")
        _make_csv(p, rows)
        loader = DatasetLoader()
        texts, labels = loader.load(p)
        return loader, texts, labels

    def test_returns_three_splits(self, tmp_path):
        loader, texts, labels = self._loader_with_data(tmp_path)
        result = loader.split(texts, labels)
        assert len(result) == 3

    def test_splits_are_split_objects(self, tmp_path):
        loader, texts, labels = self._loader_with_data(tmp_path)
        train, val, test = loader.split(texts, labels)
        for s in (train, val, test):
            assert isinstance(s, Split)

    def test_no_overlap_between_splits(self, tmp_path):
        loader, texts, labels = self._loader_with_data(tmp_path, n_authors=3, samples_per=20)
        train, val, test = loader.split(texts, labels)
        train_set = set(train.texts)
        val_set = set(val.texts)
        test_set = set(test.texts)
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_all_samples_accounted_for(self, tmp_path):
        loader, texts, labels = self._loader_with_data(tmp_path, n_authors=3, samples_per=20)
        train, val, test = loader.split(texts, labels)
        total = len(train.texts) + len(val.texts) + len(test.texts)
        assert total == len(texts)

    def test_all_authors_in_each_split(self, tmp_path):
        loader, texts, labels = self._loader_with_data(tmp_path, n_authors=3, samples_per=20)
        train, val, test = loader.split(texts, labels)
        all_ids = set(labels)
        assert set(train.labels) == all_ids
        assert set(val.labels) == all_ids
        assert set(test.labels) == all_ids

    def test_author_map_propagated(self, tmp_path):
        loader, texts, labels = self._loader_with_data(tmp_path)
        train, val, test = loader.split(texts, labels)
        for s in (train, val, test):
            assert s.author_map == loader.author_map

    def test_approximate_train_ratio(self, tmp_path):
        loader, texts, labels = self._loader_with_data(tmp_path, n_authors=3, samples_per=30)
        train, val, test = loader.split(texts, labels, train_ratio=0.7, val_ratio=0.15)
        n = len(texts)
        # Allow ±5% tolerance due to stratification rounding
        assert abs(len(train.texts) / n - 0.7) < 0.05

    def test_insufficient_samples_raises(self, tmp_path):
        # author_0 has 5 samples, below default threshold of 10
        rows = (
            [(f"text {i}", "author_0") for i in range(5)]
            + [(f"text {i}", "author_1") for i in range(15)]
            + [(f"text {i}", "author_2") for i in range(15)]
        )
        p = str(tmp_path / "data.csv")
        _make_csv(p, rows)
        loader = DatasetLoader()
        texts, labels = loader.load(p)
        with pytest.raises(InsufficientSamplesError):
            loader.split(texts, labels)

    def test_insufficient_samples_error_names_author(self, tmp_path):
        rows = (
            [(f"text {i}", "rare_author") for i in range(3)]
            + [(f"text {i}", "common_author") for i in range(20)]
        )
        p = str(tmp_path / "data.csv")
        _make_csv(p, rows)
        loader = DatasetLoader()
        texts, labels = loader.load(p)
        with pytest.raises(InsufficientSamplesError, match="rare_author"):
            loader.split(texts, labels)

    def test_custom_min_samples_threshold(self, tmp_path):
        # 8 samples per author — passes with min_samples=5, fails with default 10
        rows = _synthetic_rows(3, 8)
        p = str(tmp_path / "data.csv")
        _make_csv(p, rows)
        loader = DatasetLoader()
        texts, labels = loader.load(p)
        # Should not raise with lower threshold
        train, val, test = loader.split(texts, labels, min_samples=5)
        assert len(train.texts) > 0
