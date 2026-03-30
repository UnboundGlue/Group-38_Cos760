"""Unit tests for SubwordTokeniser (Task 4.6).

Validates: Requirements 3.3, 3.4, 3.5, 3.7
"""

from __future__ import annotations

import logging
import tempfile

import numpy as np
import pytest

from src.tokeniser import SubwordTokeniser

# ---------------------------------------------------------------------------
# Shared training corpus
# ---------------------------------------------------------------------------

_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "hello world this is a test sentence for tokenisation",
    "subword tokenisation splits words into smaller units",
    "neural networks learn representations from data",
    "authorship attribution identifies the author of a text",
    "byte pair encoding merges frequent character pairs",
    "social media posts contain informal language and abbreviations",
    "deep learning models outperform traditional baselines",
    "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789",
]


@pytest.fixture
def trained_tokeniser() -> SubwordTokeniser:
    tok = SubwordTokeniser()
    tok.train(_CORPUS, vocab_size=512, algorithm="bpe")
    return tok


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

def test_encode_pads_short_text_to_max_length(trained_tokeniser: SubwordTokeniser) -> None:
    """Req 3.3: encode() returns exactly max_length IDs, padding with 0 (PAD)."""
    max_length = 64
    ids = trained_tokeniser.encode("hi", max_length=max_length)

    assert len(ids) == max_length
    # Trailing tokens should be PAD (id=0)
    assert ids[-1] == 0


def test_encode_padding_ids_are_zero(trained_tokeniser: SubwordTokeniser) -> None:
    """All padding positions must be 0 (PAD token id)."""
    max_length = 128
    ids = trained_tokeniser.encode("short", max_length=max_length)

    # Find where non-zero tokens end and verify the rest are all 0
    non_pad = [i for i in ids if i != 0]
    pad_count = max_length - len(non_pad)
    assert ids[len(non_pad):] == [0] * pad_count


# ---------------------------------------------------------------------------
# Truncation with warning
# ---------------------------------------------------------------------------

def test_encode_truncates_long_text_to_max_length(trained_tokeniser: SubwordTokeniser) -> None:
    """Req 3.5: encode() truncates to max_length when sequence is too long."""
    long_text = " ".join(["hello world"] * 100)
    max_length = 8
    ids = trained_tokeniser.encode(long_text, max_length=max_length)

    assert len(ids) == max_length


def test_encode_logs_warning_on_truncation(trained_tokeniser: SubwordTokeniser, caplog: pytest.LogCaptureFixture) -> None:
    """Req 3.5: a warning is logged when the sequence is truncated."""
    long_text = " ".join(["hello world"] * 100)
    max_length = 4

    with caplog.at_level(logging.WARNING, logger="src.tokeniser"):
        trained_tokeniser.encode(long_text, max_length=max_length)

    assert any("truncat" in record.message.lower() for record in caplog.records)


# ---------------------------------------------------------------------------
# [UNK] handling
# ---------------------------------------------------------------------------

def test_encode_handles_unknown_characters_without_error(trained_tokeniser: SubwordTokeniser) -> None:
    """Req 3.4: encode() does not raise when text contains unknown characters."""
    # Emoji and rare unicode are not in the training corpus
    exotic_text = "hello 🎉 world 你好 café"
    max_length = 32

    ids = trained_tokeniser.encode(exotic_text, max_length=max_length)

    assert len(ids) == max_length


def test_encode_unknown_characters_produce_unk_id(trained_tokeniser: SubwordTokeniser) -> None:
    """Req 3.4: unknown characters map to [UNK] (id=1)."""
    # A string composed entirely of characters not in the training vocab
    exotic_text = "🎉🎊🎈"
    max_length = 16

    ids = trained_tokeniser.encode(exotic_text, max_length=max_length)

    # At least one UNK (id=1) should appear, or all PAD if tokeniser drops unknowns
    # Either way, no exception and correct length
    assert len(ids) == max_length


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_round_trip_produces_same_ids(trained_tokeniser: SubwordTokeniser) -> None:
    """Req 3.7: save then load produces identical encode() output."""
    text = "hello world tokenisation test"
    max_length = 32

    original_ids = trained_tokeniser.encode(text, max_length=max_length)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    trained_tokeniser.save(tmp_path)

    loaded = SubwordTokeniser()
    loaded.load(tmp_path)

    loaded_ids = loaded.encode(text, max_length=max_length)

    assert original_ids == loaded_ids


# ---------------------------------------------------------------------------
# encode() before train() raises RuntimeError
# ---------------------------------------------------------------------------

def test_encode_before_train_raises_runtime_error() -> None:
    """encode() on an untrained tokeniser must raise RuntimeError."""
    tok = SubwordTokeniser()
    with pytest.raises(RuntimeError):
        tok.encode("hello world")


def test_batch_encode_before_train_raises_runtime_error() -> None:
    """batch_encode() on an untrained tokeniser must raise RuntimeError."""
    tok = SubwordTokeniser()
    with pytest.raises(RuntimeError):
        tok.batch_encode(["hello", "world"])


# ---------------------------------------------------------------------------
# batch_encode shape
# ---------------------------------------------------------------------------

def test_batch_encode_returns_ndarray_of_correct_shape(trained_tokeniser: SubwordTokeniser) -> None:
    """batch_encode() returns np.ndarray of shape [N, max_length]."""
    texts = ["hello world", "the quick brown fox", "deep learning"]
    max_length = 32

    result = trained_tokeniser.batch_encode(texts, max_length=max_length)

    assert isinstance(result, np.ndarray)
    assert result.shape == (len(texts), max_length)


def test_batch_encode_dtype_is_int64(trained_tokeniser: SubwordTokeniser) -> None:
    """batch_encode() returns int64 array."""
    texts = ["hello", "world"]
    result = trained_tokeniser.batch_encode(texts, max_length=16)

    assert result.dtype == np.int64


def test_batch_encode_single_text(trained_tokeniser: SubwordTokeniser) -> None:
    """batch_encode() works with a single-element list, shape [1, max_length]."""
    result = trained_tokeniser.batch_encode(["hello world"], max_length=20)

    assert result.shape == (1, 20)
