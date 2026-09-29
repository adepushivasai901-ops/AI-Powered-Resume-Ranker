"""Statistical lexical TF-IDF vectorization and cosine similarity computation using scikit-learn."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFMatcher:
    """Computes statistical lexical TF-IDF vectors and cosine similarity between resumes and job descriptions."""

    def __init__(
        self,
        max_features: Optional[int] = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
    ):
        """Initialize TF-IDF vectorizer configuration parameters.

        Args:
            max_features: Maximum number of vocabulary features to retain.
            ngram_range: Lower and upper boundary of n-values for n-grams.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer: Optional[TfidfVectorizer] = None

    def _create_vectorizer(self) -> TfidfVectorizer:
        """Instantiate a new TfidfVectorizer with configured lexical parameters."""
        return TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            token_pattern=r"(?u)c\+\+|c\#|\.net|\b\w+\b",
        )

    def compute_similarity(
        self, resume_text: Optional[str], job_description_text: Optional[str]
    ) -> float:
        """Calculate lexical cosine similarity score between a single resume and job description.

        Fits a shared TF-IDF vocabulary across the comparison pair to ensure aligned
        feature weights, returning a normalized score between 0.0 and 1.0.

        Args:
            resume_text: Preprocessed or raw resume text.
            job_description_text: Preprocessed or raw job description text.

        Returns:
            Cosine similarity score as a float bounded within [0.0, 1.0].
        """
        if (
            not resume_text
            or not isinstance(resume_text, str)
            or not resume_text.strip()
            or not job_description_text
            or not isinstance(job_description_text, str)
            or not job_description_text.strip()
        ):
            return 0.0

        corpus = [job_description_text, resume_text]
        self.vectorizer = self._create_vectorizer()

        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
        except ValueError:
            # Raised by scikit-learn if no valid vocabulary remains after tokenization
            return 0.0

        # Cosine similarity between resume (index 1) and job description (index 0)
        job_vec = tfidf_matrix[0:1]
        resume_vec = tfidf_matrix[1:2]
        raw_sim = cosine_similarity(resume_vec, job_vec)[0, 0]

        # Clamp to strictly normalized [0.0, 1.0] interval to account for float precision
        normalized_score = float(np.clip(raw_sim, 0.0, 1.0))
        return round(normalized_score, 4)

    def compute_batch_similarity(
        self, resumes: Dict[str, str], job_description_text: Optional[str]
    ) -> Dict[str, float]:
        """Compute lexical similarity scores for multiple resumes against a single job description.

        Fits a unified TF-IDF feature space on the combined corpus (job description + all resumes),
        ensuring equitable IDF weighting and consistent statistical comparison.

        Args:
            resumes: Mapping of candidate/file identifier to resume text.
            job_description_text: Target job description text.

        Returns:
            Mapping of candidate identifier to normalized similarity score within [0.0, 1.0].
        """
        if not resumes:
            return {}

        results: Dict[str, float] = {k: 0.0 for k in resumes}

        if (
            not job_description_text
            or not isinstance(job_description_text, str)
            or not job_description_text.strip()
        ):
            return results

        candidate_keys: List[str] = list(resumes.keys())
        candidate_texts: List[str] = [
            resumes[k] if resumes[k] and isinstance(resumes[k], str) else ""
            for k in candidate_keys
        ]

        # If all candidates have empty or whitespace text, return zero scores
        if not any(t.strip() for t in candidate_texts):
            return results

        # Construct unified corpus: index 0 is job description, subsequent indices are resumes
        corpus = [job_description_text] + candidate_texts
        self.vectorizer = self._create_vectorizer()

        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
        except ValueError:
            # Handle empty vocabulary case across all inputs
            return results

        job_vec = tfidf_matrix[0:1]
        resume_vecs = tfidf_matrix[1:]

        sim_matrix = cosine_similarity(resume_vecs, job_vec)

        for idx, key in enumerate(candidate_keys):
            # If the candidate's resume text was empty, preserve 0.0 score
            if not candidate_texts[idx].strip():
                results[key] = 0.0
            else:
                score = float(np.clip(sim_matrix[idx, 0], 0.0, 1.0))
                results[key] = round(score, 4)

        return results
