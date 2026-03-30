"""Baseline feature extraction for authorship attribution comparison."""

from __future__ import annotations

import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

_VALID_METHODS = {"bow", "tfidf", "char_ngram", "word_ngram"}


class BaselineFeatureExtractor:
    """Extract traditional text features for baseline classifier comparison.

    Supports Bag-of-Words, TF-IDF, character n-grams, and word n-grams via
    scikit-learn vectorisers. After calling ``fit_transform``, the fitted
    vocabulary can be applied to unseen texts with ``transform``.

    Parameters
    ----------
    char_ngram_range:
        (min_n, max_n) for character n-gram extraction. Default ``(2, 5)``.
    word_ngram_range:
        (min_n, max_n) for word n-gram extraction. Default ``(1, 3)``.
    random_seed:
        Random seed for reproducibility. Default ``42``.
    """

    def __init__(
        self,
        char_ngram_range: tuple[int, int] = (2, 5),
        word_ngram_range: tuple[int, int] = (1, 3),
        random_seed: int = 42,
    ) -> None:
        self.char_ngram_range = char_ngram_range
        self.word_ngram_range = word_ngram_range
        self.random_seed = random_seed
        self._vectorizer: CountVectorizer | TfidfVectorizer | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self, texts: list[str], method: str
    ) -> scipy.sparse.csr_matrix:
        """Fit a vocabulary on *texts* and return the feature matrix.

        Parameters
        ----------
        texts:
            Input documents to fit and transform.
        method:
            Feature extraction strategy. Must be one of
            ``{"bow", "tfidf", "char_ngram", "word_ngram"}``.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse feature matrix of shape ``[N, F]``.

        Raises
        ------
        ValueError
            If *method* is not one of the supported strategies.
        """
        if method not in _VALID_METHODS:
            raise ValueError(
                f"method must be one of {sorted(_VALID_METHODS)}, got {method!r}"
            )

        self._vectorizer = self._build_vectorizer(method)
        result = self._vectorizer.fit_transform(texts)
        return result.tocsr()

    def transform(self, texts: list[str]) -> scipy.sparse.csr_matrix:
        """Apply the fitted vocabulary to unseen *texts*.

        Parameters
        ----------
        texts:
            Input documents to transform.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse feature matrix of shape ``[N, F]``.

        Raises
        ------
        RuntimeError
            If ``fit_transform`` has not been called yet.
        """
        if self._vectorizer is None:
            raise RuntimeError(
                "BaselineFeatureExtractor has not been fitted yet. "
                "Call fit_transform() before transform()."
            )
        result = self._vectorizer.transform(texts)
        return result.tocsr()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_vectorizer(
        self, method: str
    ) -> CountVectorizer | TfidfVectorizer:
        """Instantiate the appropriate scikit-learn vectoriser."""
        if method == "bow":
            return CountVectorizer()
        if method == "tfidf":
            return TfidfVectorizer()
        if method == "char_ngram":
            return TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=self.char_ngram_range,
            )
        # method == "word_ngram"
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=self.word_ngram_range,
        )
