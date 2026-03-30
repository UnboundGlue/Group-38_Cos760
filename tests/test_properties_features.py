"""Property-based tests for BaselineFeatureExtractor (Task 6.2).

**Validates: Requirements 4.1, 4.2**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.features import BaselineFeatureExtractor

# Fixed small alphabet to keep generation fast
_ALPHABET = "abcdefghij"
_METHODS = ["bow", "tfidf", "char_ngram", "word_ngram"]

# Strategy: non-empty words (min 2 chars) joined by spaces so sklearn tokeniser
# always finds at least one token and never produces an empty vocabulary.
_word_strategy = st.text(alphabet=_ALPHABET, min_size=2, max_size=10)
_text_strategy = st.lists(_word_strategy, min_size=1, max_size=8).map(" ".join)


# ---------------------------------------------------------------------------
# Property 12: Baseline Feature Matrix Row Count
# ---------------------------------------------------------------------------

@given(
    texts=st.lists(_text_strategy, min_size=1, max_size=20),
    second_texts=st.lists(_text_strategy, min_size=1, max_size=20),
    method=st.sampled_from(_METHODS),
)
@settings(max_examples=50)
def test_baseline_feature_matrix_row_count(
    texts: list[str],
    second_texts: list[str],
    method: str,
) -> None:
    """**Validates: Requirements 4.1, 4.2**

    For any list of N texts, calling fit_transform() or transform() on the
    BaselineFeatureExtractor should return a sparse matrix with exactly N rows,
    and transform() should produce a matrix with the same number of feature
    columns as fit_transform().
    """
    extractor = BaselineFeatureExtractor()

    # fit_transform on first list
    matrix_fit = extractor.fit_transform(texts, method=method)

    assert matrix_fit.shape[0] == len(texts), (
        f"fit_transform() returned {matrix_fit.shape[0]} rows for {len(texts)} texts "
        f"(method={method!r})"
    )

    # transform on second list — must have same column count
    matrix_transform = extractor.transform(second_texts)

    assert matrix_transform.shape[0] == len(second_texts), (
        f"transform() returned {matrix_transform.shape[0]} rows for {len(second_texts)} texts "
        f"(method={method!r})"
    )

    assert matrix_transform.shape[1] == matrix_fit.shape[1], (
        f"transform() produced {matrix_transform.shape[1]} columns but "
        f"fit_transform() produced {matrix_fit.shape[1]} columns (method={method!r})"
    )


# ---------------------------------------------------------------------------
# Property 13: Baseline Reproducibility
# ---------------------------------------------------------------------------

@given(
    texts=st.lists(_text_strategy, min_size=1, max_size=20),
    method=st.sampled_from(_METHODS),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=50)
def test_baseline_reproducibility(
    texts: list[str],
    method: str,
    seed: int,
) -> None:
    """**Validates: Requirements 4.4**

    For any fixed random seed, input texts, and extraction method, calling
    fit_transform() twice on two separate BaselineFeatureExtractor instances
    (both with the same random_seed) should produce identical feature matrices.
    """
    import numpy as np

    extractor_a = BaselineFeatureExtractor(random_seed=seed)
    extractor_b = BaselineFeatureExtractor(random_seed=seed)

    matrix_a = extractor_a.fit_transform(texts, method=method)
    matrix_b = extractor_b.fit_transform(texts, method=method)

    assert matrix_a.shape == matrix_b.shape, (
        f"Shapes differ: {matrix_a.shape} vs {matrix_b.shape} "
        f"(method={method!r}, seed={seed})"
    )

    np.testing.assert_array_equal(
        matrix_a.toarray(),
        matrix_b.toarray(),
        err_msg=(
            f"Feature matrices differ for method={method!r}, seed={seed}, "
            f"texts={texts!r}"
        ),
    )
