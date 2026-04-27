"""
Baseline feature extraction for authorship attribution.

Wraps scikit-learn vectorizers for four classical stylometric baselines:
    - bow:   word-level bag-of-words counts
    - tfidf: word-level TF-IDF
    - char:  character n-gram TF-IDF (the classical stylometry workhorse,
             cf. Stamatatos 2009 — captures morphology, function-word
             fragments, and punctuation patterns invisible to word-level
             features)
    - word:  word n-gram TF-IDF with configurable range (e.g. (1, 2)
             for unigrams + bigrams)

The extractor is single-method-per-instance: instantiate once per
baseline you want to compare. This keeps the API parallel with sklearn's
fit/transform contract and matches the experiment script's needs in
Task 12.2 (separate matrices feeding separate classifiers).

Implements Task 6.1. References Requirements 4.1–4.4.
"""
from __future__ import annotations

from typing import Iterable, Literal, Sequence

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

Method = Literal["bow", "tfidf", "char", "word"]
_VALID_METHODS: tuple[Method, ...] = ("bow", "tfidf", "char", "word")


class BaselineFeatureExtractor:
    """
    Fit a classical text-feature vocabulary on a training corpus and
    transform both that corpus and any unseen texts into a sparse
    document-term matrix.

    Parameters
    ----------
    method : {"bow", "tfidf", "char", "word"}
        Which feature family to use. See module docstring.
    ngram_range : tuple[int, int], optional
        (min_n, max_n) for n-gram extraction. Defaults differ by method:
            bow / tfidf -> (1, 1)   (unigrams only)
            word        -> (1, 2)   (uni + bigrams; standard baseline)
            char        -> (3, 5)   (char 3-to-5-grams; Stamatatos default)
        Pass an explicit value to override.
    max_features : int | None
        Cap on vocabulary size. None = unbounded.
    min_df : int | float
        Minimum document frequency for a term to be kept. Same semantics
        as sklearn (int = absolute count, float = proportion).
    lowercase : bool
        Whether to lowercase before tokenising.
    random_seed : int | None
        Stored for downstream consumers (classifiers, splitters). The
        underlying sklearn vectorizers are deterministic and do not
        themselves consume a seed; this attribute exists so the calling
        experiment script has a single place to read its seed from.

    Attributes
    ----------
    vectorizer_ : sklearn vectorizer
        The fitted underlying estimator. Available after fit_transform().
    method : Method
        The method passed at construction (echoed for introspection).

    Raises
    ------
    ValueError
        If `method` is not one of the four supported names, or if
        `transform` is called before `fit_transform`.
    """

    def __init__(
        self,
        method: Method = "tfidf",
        *,
        ngram_range: tuple[int, int] | None = None,
        max_features: int | None = None,
        min_df: int | float = 1,
        lowercase: bool = True,
        random_seed: int | None = None,
    ) -> None:
        if method not in _VALID_METHODS:
            raise ValueError(
                f"method must be one of {_VALID_METHODS}, got {method!r}"
            )
        self.method: Method = method
        self.ngram_range: tuple[int, int] = ngram_range or self._default_ngram(method)
        self.max_features = max_features
        self.min_df = min_df
        self.lowercase = lowercase
        self.random_seed = random_seed
        self.vectorizer_: CountVectorizer | TfidfVectorizer | None = None

    @staticmethod
    def _default_ngram(method: Method) -> tuple[int, int]:
        if method == "char":
            return (3, 5)
        if method == "word":
            return (1, 2)
        return (1, 1)  # bow, tfidf

    def _build_vectorizer(self) -> CountVectorizer | TfidfVectorizer:
        common = dict(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            lowercase=self.lowercase,
        )
        if self.method == "bow":
            return CountVectorizer(**common)
        if self.method == "tfidf":
            return TfidfVectorizer(**common)
        if self.method == "word":
            return TfidfVectorizer(analyzer="word", **common)
        # method == "char"
        # `char_wb` keeps n-grams within word boundaries (Stamatatos-style);
        # plain `char` lets them straddle whitespace. char_wb is the more
        # standard authorship-attribution choice.
        return TfidfVectorizer(analyzer="char_wb", **common)

    def fit_transform(self, texts: Sequence[str] | Iterable[str]) -> csr_matrix:
        """
        Learn the vocabulary on `texts` and return the [N, F] sparse
        document-term matrix.

        Calling this twice rebuilds the vocabulary from scratch — this
        matches sklearn semantics and means the second call's output is
        independent of the first.
        """
        texts = list(texts)
        self.vectorizer_ = self._build_vectorizer()
        return self.vectorizer_.fit_transform(texts)

    def transform(self, texts: Sequence[str] | Iterable[str]) -> csr_matrix:
        """
        Apply the already-fitted vocabulary to `texts`. Unseen tokens
        are dropped (sklearn default — no [UNK] mechanism for classical
        baselines).

        Raises ValueError if called before fit_transform().
        """
        if self.vectorizer_ is None:
            raise ValueError(
                "transform() called before fit_transform(); "
                "the extractor has no vocabulary yet."
            )
        return self.vectorizer_.transform(list(texts))

    @property
    def vocabulary_size(self) -> int:
        """Number of features in the fitted vocabulary."""
        if self.vectorizer_ is None:
            raise ValueError("Extractor is not fitted.")
        return len(self.vectorizer_.vocabulary_)
