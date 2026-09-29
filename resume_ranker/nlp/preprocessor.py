"""Text preprocessing and normalization interface using SpaCy."""

import re
from typing import List, Optional, Set

import spacy
from spacy.language import Language


# Crucial technical terms, abbreviations, and acronyms that must not be filtered out as stopwords
PRESERVED_TECH_TERMS: Set[str] = {
    "ai",
    "ml",
    "nlp",
    "it",
    "ci",
    "cd",
    "sql",
    "aws",
    "gcp",
    "api",
    "apis",
    "rest",
    "ui",
    "ux",
    "c",
    "r",
    "go",
    "net",
    "c++",
    "c#",
    "react",
    "python",
    "docker",
    "flask",
    "fastapi",
    "git",
    "pytorch",
    "tensorflow",
    "keras",
    "pandas",
    "numpy",
    "scikit",
    "sklearn",
}


class TextPreprocessor:
    """Handles text cleaning, tokenization, lemmatization, and stopword removal using SpaCy."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize preprocessor with specified SpaCy language model identifier.

        Args:
            model_name: SpaCy model identifier (default: "en_core_web_sm").
        """
        self.model_name = model_name
        self._nlp: Optional[Language] = None

    def _get_nlp(self) -> Language:
        """Lazy-load and return the SpaCy NLP pipeline instance."""
        if self._nlp is None:
            try:
                # Disable parser and ner components to optimize tokenization and lemmatization speed
                self._nlp = spacy.load(self.model_name, disable=["ner", "parser"])
            except OSError:
                # Fallback to blank model if full model is unavailable in environment
                self._nlp = spacy.blank("en")
        return self._nlp

    def clean_text(self, text: Optional[str]) -> str:
        """Clean noise, special formatting characters, URLs, and extraneous whitespace.

        Args:
            text: Raw input text.

        Returns:
            Cleaned and normalized text string.
        """
        if not text or not isinstance(text, str):
            return ""

        # Remove URLs
        cleaned = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # Remove email addresses
        cleaned = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", cleaned)

        # Replace bullet points and special bullet unicode symbols with space
        cleaned = re.sub(r"[\u2022\u2023\u25e6\u2043\u2219\uf0b7\u25aa\u25cf\u25cb\u25ba\u25a0*•]", " ", cleaned)

        # Normalize linebreaks, carriage returns, and tabs to spaces
        cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)

        # Remove non-printable / control characters while preserving standard punctuation
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", cleaned)

        # Collapse multiple spaces into single space and strip edges
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    def tokenize_and_lemmatize(self, text: Optional[str]) -> List[str]:
        """Tokenize, lemmatize, and filter stopwords/punctuation using SpaCy.

        Keeps technically meaningful tokens (e.g., Python, SQL, NLP, AWS, React, etc.)
        while filtering common English stopwords and non-informative tokens.

        Args:
            text: Input text string.

        Returns:
            List of normalized lemma tokens.
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        nlp = self._get_nlp()
        doc = nlp(cleaned)

        tokens: List[str] = []
        for token in doc:
            token_lower = token.text.lower()

            # Preserve critical technical terms regardless of stopword or length status
            if token_lower in PRESERVED_TECH_TERMS:
                tokens.append(token_lower)
                continue

            # Filter whitespace and punctuation
            if token.is_space or token.is_punct:
                continue

            # Filter pure numeric tokens
            if token.like_num or token.text.isdigit():
                continue

            # Filter stopwords
            if token.is_stop:
                continue

            # Filter single-character noise unless in preserved technical list
            if len(token_lower) <= 1:
                continue

            # Lemmatize token
            lemma = token.lemma_.lower().strip() if hasattr(token, "lemma_") else token_lower
            if lemma and lemma != "-pron-":
                tokens.append(lemma)
            elif token_lower:
                tokens.append(token_lower)

        return tokens

    def preprocess(self, text: Optional[str]) -> str:
        """Run full preprocessing pipeline on raw text for TF-IDF vectorization.

        Args:
            text: Raw input text string.

        Returns:
            Fully preprocessed, space-separated normalized token string.
        """
        tokens = self.tokenize_and_lemmatize(text)
        return " ".join(tokens)
