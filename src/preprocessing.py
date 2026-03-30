"""Text preprocessing for the neural authorship attribution pipeline."""

from __future__ import annotations

import re


class Preprocessor:
    """Clean and normalise raw social media text before tokenisation.

    Parameters
    ----------
    preserve_punctuation:
        When True (default), punctuation is kept because it carries
        stylometric signal.  When False, punctuation is stripped.
    """

    # Compiled patterns (class-level for efficiency)
    _URL_RE = re.compile(
        r"https?://\S+|www\.\S+",
        re.IGNORECASE,
    )
    _MENTION_RE = re.compile(r"@\w+")
    _WHITESPACE_RE = re.compile(r"\s+")
    _PUNCTUATION_RE = re.compile(r"[^\w\s]")

    def __init__(self, preserve_punctuation: bool = True) -> None:
        self.preserve_punctuation = preserve_punctuation

    def clean(self, text: str) -> str:
        """Remove URLs and @mentions, normalise whitespace.

        Punctuation is preserved by default (stylometric signal).
        Returns an empty string if the text becomes empty after cleaning.
        """
        if not text:
            return ""

        # Remove URLs
        text = self._URL_RE.sub(" ", text)

        # Remove @mentions
        text = self._MENTION_RE.sub(" ", text)

        # Optionally strip punctuation
        if not self.preserve_punctuation:
            text = self._PUNCTUATION_RE.sub(" ", text)

        # Normalise whitespace (collapse multiple spaces/tabs/newlines)
        text = self._WHITESPACE_RE.sub(" ", text).strip()

        return text

    def batch_clean(self, texts: list[str]) -> list[str]:
        """Apply clean() to every element in *texts*.

        Returns a list of the same length as the input.
        Empty strings in the input pass through as empty strings.
        """
        return [self.clean(t) for t in texts]
