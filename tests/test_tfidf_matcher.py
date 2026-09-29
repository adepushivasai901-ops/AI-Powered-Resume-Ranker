"""Unit tests for TF-IDF vectorization and similarity matching."""

import pytest
from resume_ranker.vectorization.tfidf_matcher import TFIDFMatcher


@pytest.fixture
def matcher() -> TFIDFMatcher:
    """Fixture providing initialized TFIDFMatcher."""
    return TFIDFMatcher()


def test_tfidf_matcher_initialization():
    """Verify TFIDFMatcher initialization with default and custom settings."""
    default_m = TFIDFMatcher()
    assert default_m.max_features == 5000
    assert default_m.ngram_range == (1, 2)

    custom_m = TFIDFMatcher(max_features=250, ngram_range=(1, 3))
    assert custom_m.max_features == 250
    assert custom_m.ngram_range == (1, 3)


def test_identical_texts_similarity(matcher: TFIDFMatcher):
    """Verify identical texts produce a similarity score close to 1.0."""
    text = "python developer building scalable microservices with flask and sql"
    similarity = matcher.compute_similarity(text, text)

    assert 0.99 <= similarity <= 1.0
    assert isinstance(similarity, float)


def test_matching_texts_positive_similarity(matcher: TFIDFMatcher):
    """Verify related texts with shared vocabulary produce positive similarity."""
    resume = "experienced python engineer developing web applications and database APIs"
    job_desc = "seeking python developer with experience in web applications"

    similarity = matcher.compute_similarity(resume, job_desc)
    assert 0.1 <= similarity <= 1.0


def test_unrelated_texts_lower_similarity(matcher: TFIDFMatcher):
    """Verify unrelated texts produce lower (or zero) similarity compared to matching texts."""
    tech_resume = "python developer backend distributed systems databases"
    culinary_text = "french pastry chef specialized in sourdough baking and desserts"
    tech_job = "hiring python backend engineer with distributed database experience"

    related_sim = matcher.compute_similarity(tech_resume, tech_job)
    unrelated_sim = matcher.compute_similarity(culinary_text, tech_job)

    assert unrelated_sim < related_sim
    assert unrelated_sim == 0.0


def test_empty_resume_returns_zero(matcher: TFIDFMatcher):
    """Verify empty or None resume returns 0.0 safely."""
    job_desc = "python developer needed"
    assert matcher.compute_similarity("", job_desc) == 0.0
    assert matcher.compute_similarity("   ", job_desc) == 0.0
    assert matcher.compute_similarity(None, job_desc) == 0.0


def test_empty_job_description_returns_zero(matcher: TFIDFMatcher):
    """Verify empty or None job description returns 0.0 safely."""
    resume = "python developer with 5 years experience"
    assert matcher.compute_similarity(resume, "") == 0.0
    assert matcher.compute_similarity(resume, "   \n\t") == 0.0
    assert matcher.compute_similarity(resume, None) == 0.0


def test_both_inputs_empty_or_whitespace(matcher: TFIDFMatcher):
    """Verify empty or whitespace-only inputs on both sides return 0.0."""
    assert matcher.compute_similarity("", "") == 0.0
    assert matcher.compute_similarity("   ", "   \t\n") == 0.0


def test_similarity_range_bounds(matcher: TFIDFMatcher):
    """Verify similarity outputs strictly remain within the [0.0, 1.0] interval."""
    resume = "machine learning engineer with pytorch and scikit-learn"
    job_desc = "machine learning engineer experienced in model training"

    score = matcher.compute_similarity(resume, job_desc)
    assert 0.0 <= score <= 1.0


def test_batch_resume_comparison(matcher: TFIDFMatcher):
    """Verify batch comparison compares multiple resumes against one job description in a shared space."""
    resumes = {
        "candidate_strong": "senior python developer with sql docker and api design",
        "candidate_weak": "junior python learner with basic knowledge",
        "candidate_unrelated": "lead pastry chef in commercial bakery",
        "candidate_empty": "",
    }
    job_desc = "looking for senior python developer with docker and sql experience"

    batch_scores = matcher.compute_batch_similarity(resumes, job_desc)

    assert isinstance(batch_scores, dict)
    assert len(batch_scores) == 4
    assert batch_scores["candidate_empty"] == 0.0
    assert batch_scores["candidate_unrelated"] == 0.0
    assert batch_scores["candidate_strong"] > batch_scores["candidate_weak"]
    for cid, score in batch_scores.items():
        assert 0.0 <= score <= 1.0


def test_batch_with_empty_job_description(matcher: TFIDFMatcher):
    """Verify batch comparison with empty job description returns 0.0 for all candidates."""
    resumes = {
        "c1": "python developer",
        "c2": "java developer",
    }
    scores = matcher.compute_batch_similarity(resumes, "")
    assert scores == {"c1": 0.0, "c2": 0.0}


def test_configured_max_features_and_ngram(matcher: TFIDFMatcher):
    """Verify that configured max_features and ngram_range are applied to the vectorizer."""
    custom_matcher = TFIDFMatcher(max_features=5, ngram_range=(1, 1))
    text1 = "python flask sql docker aws kubernetes linux"
    text2 = "python flask sql"

    score = custom_matcher.compute_similarity(text1, text2)
    assert 0.0 <= score <= 1.0
    assert custom_matcher.vectorizer is not None
    assert custom_matcher.vectorizer.max_features == 5
    assert custom_matcher.vectorizer.ngram_range == (1, 1)
    # Check that vocabulary size does not exceed max_features
    assert len(custom_matcher.vectorizer.vocabulary_) <= 5
