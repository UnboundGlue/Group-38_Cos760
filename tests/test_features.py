"""Unit tests for BaselineFeatureExtractor (Task 6.4).

Validates: Requirements 4.1, 4.2, 4.3, 4.4
"""

from __future__ import annotations

import pytest
import scipy.sparse

from src.features import BaselineFeatureExtractor

# ---------------------------------------------------------------------------
# Shared corpus
# ---------------------------------------------------------------------------

_TRAIN_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "hello world this is a test sentence",
    "neural networks learn representations from data",
    "authorship attribution identifies the author of a text",
    "byte pair encoding merges frequent character pairs",
    "social media posts contain informal language",
    "deep learning models outperform traditional baselines",
    "subword tokenisation splits words into smaller units",
]

_UNSEEN_TEXTS = [
    "a completely different sentence about nothing",
    "another unseen document for transform testing",
    "yet another text not in the training set",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extractor(**kwargs) -> BaselineFeatureExtractor:
    return BaselineFeatureExtractor(**kwargs)


# ---------------------------------------------------------------------------
# Req 4.1 — fit_transform returns sparse matrix with correct row count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["bow", "tfidf", "char_ngram", "word_ngram"])
def test_fit_transform_returns_sparse_matrix(method: str) -> None:
    """Req 4.1: fit_transform returns a sparse matrix for every supported method."""
    extractor = _extractor()
    result = extractor.fit_transform(_TRAIN_TEXTS, method=method)

    assert scipy.sparse.issparse(result)


@pytest.mark.parametrize("method", ["bow", "tfidf", "char_ngram", "word_ngram"])
def test_fit_transform_row_count_matches_input(method: str) -> None:
    """Req 4.1: returned matrix has exactly N rows (one per input text)."""
    extractor = _extractor()
    result = extractor.fit_transform(_TRAIN_TEXTS, method=method)

    assert result.shape[0] == len(_TRAIN_TEXTS)


# ---------------------------------------------------------------------------
# Req 4.2 — transform on unseen texts uses fitted vocabulary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["bow", "tfidf", "char_ngram", "word_ngram"])
def test_transform_returns_same_column_count_as_fit_transform(method: str) -> None:
    """Req 4.2: transform on unseen texts produces same number of features as fit_transform."""
    extractor = _extractor()
    train_matrix = extractor.fit_transform(_TRAIN_TEXTS, method=method)
    unseen_matrix = extractor.transform(_UNSEEN_TEXTS)

    assert unseen_matrix.shape[1] == train_matrix.shape[1]


@pytest.mark.parametrize("method", ["bow", "tfidf", "char_ngram", "word_ngram"])
def test_transform_row_count_matches_unseen_input(method: str) -> None:
    """Req 4.2: transform returns one row per unseen text."""
    extractor = _extractor()
    extractor.fit_transform(_TRAIN_TEXTS, method=method)
    result = extractor.transform(_UNSEEN_TEXTS)

    assert result.shape[0] == len(_UNSEEN_TEXTS)


@pytest.mark.parametrize("method", ["bow", "tfidf", "char_ngram", "word_ngram"])
def test_transform_returns_sparse_matrix(method: str) -> None:
    """Req 4.2: transform returns a sparse matrix."""
    extractor = _extractor()
    extractor.fit_transform(_TRAIN_TEXTS, method=method)
    result = extractor.transform(_UNSEEN_TEXTS)

    assert scipy.sparse.issparse(result)


# ---------------------------------------------------------------------------
# Req 4.3 — configurable n-gram ranges
# ---------------------------------------------------------------------------

def test_char_ngram_range_is_respected() -> None:
    """Req 4.3: char_ngram_range config changes the feature space."""
    extractor_narrow = _extractor(char_ngram_range=(2, 2))
    extractor_wide = _extractor(char_ngram_range=(2, 6))

    narrow = extractor_narrow.fit_transform(_TRAIN_TEXTS, method="char_ngram")
    wide = extractor_wide.fit_transform(_TRAIN_TEXTS, method="char_ngram")

    # Wider range should produce more features
    assert wide.shape[1] > narrow.shape[1]


def test_word_ngram_range_is_respected() -> None:
    """Req 4.3: word_ngram_range config changes the feature space."""
    extractor_unigram = _extractor(word_ngram_range=(1, 1))
    extractor_trigram = _extractor(word_ngram_range=(1, 3))

    unigram = extractor_unigram.fit_transform(_TRAIN_TEXTS, method="word_ngram")
    trigram = extractor_trigram.fit_transform(_TRAIN_TEXTS, method="word_ngram")

    # Trigram range should produce more features than unigram only
    assert trigram.shape[1] > unigram.shape[1]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_transform_before_fit_transform_raises_runtime_error() -> None:
    """transform() before fit_transform() must raise RuntimeError."""
    extractor = _extractor()
    with pytest.raises(RuntimeError):
        extractor.transform(_UNSEEN_TEXTS)


def test_invalid_method_raises_value_error() -> None:
    """fit_transform() with an unsupported method must raise ValueError."""
    extractor = _extractor()
    with pytest.raises(ValueError):
        extractor.fit_transform(_TRAIN_TEXTS, method="invalid_method")


def test_invalid_method_error_message_mentions_method() -> None:
    """ValueError message should reference the invalid method name."""
    extractor = _extractor()
    with pytest.raises(ValueError, match="invalid_method"):
        extractor.fit_transform(_TRAIN_TEXTS, method="invalid_method")


# ---------------------------------------------------------------------------
# Req 4.4 — reproducibility with fixed random seed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["bow", "tfidf", "char_ngram", "word_ngram"])
def test_fit_transform_is_reproducible_with_same_seed(method: str) -> None:
    """Req 4.4: same seed and inputs produce identical feature matrices."""
    extractor_a = _extractor(random_seed=42)
    extractor_b = _extractor(random_seed=42)

    matrix_a = extractor_a.fit_transform(_TRAIN_TEXTS, method=method)
    matrix_b = extractor_b.fit_transform(_TRAIN_TEXTS, method=method)

    diff = (matrix_a - matrix_b)
    assert diff.nnz == 0, "Matrices should be identical for the same seed and inputs"
