"""Unit tests for SpaCy NLP preprocessing module."""

from unittest.mock import MagicMock, patch
import pytest

from resume_ranker.nlp.preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor() -> TextPreprocessor:
    """Fixture providing initialized preprocessor."""
    return TextPreprocessor(model_name="en_core_web_sm")


def test_preprocessor_initialization():
    """Verify preprocessor configuration with default and custom model names."""
    default_p = TextPreprocessor()
    assert default_p.model_name == "en_core_web_sm"

    custom_p = TextPreprocessor(model_name="custom_model")
    assert custom_p.model_name == "custom_model"


def test_clean_text(preprocessor: TextPreprocessor):
    """Verify noise removal, URL stripping, and whitespace normalization."""
    raw_text = (
        "Experienced Engineer with   multiple   spaces.\n\n"
        "Check my portfolio: https://github.com/developer and http://example.com/site\n"
        "• Bullet item 1\n"
        "\uf0b7 Bullet item 2\twith tabs."
    )
    cleaned = preprocessor.clean_text(raw_text)

    assert "https://github.com/developer" not in cleaned
    assert "http://example.com/site" not in cleaned
    assert "•" not in cleaned
    assert "\uf0b7" not in cleaned
    assert "\t" not in cleaned
    assert "\n" not in cleaned
    assert "   " not in cleaned
    assert "Experienced Engineer with multiple spaces." in cleaned
    assert "Bullet item 1 Bullet item 2 with tabs." in cleaned


def test_clean_text_empty_or_none(preprocessor: TextPreprocessor):
    """Verify that empty, whitespace-only, or None text returns an empty string."""
    assert preprocessor.clean_text("") == ""
    assert preprocessor.clean_text("   \n\t  ") == ""
    assert preprocessor.clean_text(None) == ""


def test_tokenize_and_lemmatize(preprocessor: TextPreprocessor):
    """Verify lemmatization, stopword removal, and token normalization."""
    sample = "The senior developers were programming complex predictive models with precision."
    tokens = preprocessor.tokenize_and_lemmatize(sample)

    assert isinstance(tokens, list)
    # Stopwords like "the", "were", "with" must be filtered out
    assert "the" not in tokens
    assert "were" not in tokens
    assert "with" not in tokens

    # Lemmatized root forms
    assert "develop" in tokens or "developer" in tokens
    assert "model" in tokens
    assert "predictive" in tokens or "predict" in tokens


def test_preserves_technical_terms(preprocessor: TextPreprocessor):
    """Verify that important resume technical terminology is strictly preserved."""
    skills_text = (
        "Skilled in Python, SQL, NLP, AWS, React, Docker, Kubernetes, and Machine Learning. "
        "Also proficient in AI, ML, CI/CD, and REST APIs."
    )
    tokens = preprocessor.tokenize_and_lemmatize(skills_text)

    # Core required technical keywords
    assert "python" in tokens
    assert "sql" in tokens
    assert "nlp" in tokens
    assert "aws" in tokens
    assert "react" in tokens
    assert "docker" in tokens
    assert "ai" in tokens
    assert "ml" in tokens


def test_preprocess_end_to_end(preprocessor: TextPreprocessor):
    """Verify full preprocessing pipeline outputs a space-separated normalized string for TF-IDF."""
    resume_snippet = (
        "Senior Software Engineer specializing in Python and AWS cloud infrastructure.\n"
        "Responsible for designing scalable databases with PostgreSQL and Docker.\n"
        "Portfolio: https://linkedin.com/in/test-profile"
    )
    result = preprocessor.preprocess(resume_snippet)

    assert isinstance(result, str)
    assert "https" not in result
    assert "python" in result
    assert "aws" in result
    assert "docker" in result
    # Check that it's a space-separated string of tokens
    token_list = result.split()
    assert len(token_list) > 3


def test_preprocess_empty_returns_empty_string(preprocessor: TextPreprocessor):
    """Verify preprocess returns empty string for empty input."""
    assert preprocessor.preprocess("") == ""
    assert preprocessor.preprocess("   ") == ""
    assert preprocessor.preprocess(None) == ""


def test_mocked_model_loading():
    """Verify that SpaCy model is loaded with configured model name."""
    mock_nlp = MagicMock()
    with patch("resume_ranker.nlp.preprocessor.spacy.load", return_value=mock_nlp) as mock_load:
        p = TextPreprocessor(model_name="en_core_web_sm")
        nlp_instance = p._get_nlp()

        mock_load.assert_called_once_with("en_core_web_sm", disable=["ner", "parser"])
        assert nlp_instance == mock_nlp
