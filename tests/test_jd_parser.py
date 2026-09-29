"""Unit tests for JDParser structured requirement extraction."""

import pytest
from resume_ranker.models.job_description import ImportanceLevel, RequirementType
from resume_ranker.nlp.jd_parser import JDParser


@pytest.fixture
def parser() -> JDParser:
    """Fixture providing initialized JDParser instance."""
    return JDParser()


def test_parse_sample_data_science_jd(parser: JDParser):
    """Verify parsing of realistic multi-section job description."""
    jd_text = """
    Position: Senior Machine Learning Engineer

    Requirements:
    - 4+ years of professional experience in software engineering and ML.
    - Strong proficiency in Python and SQL.
    - Hands-on experience with PyTorch and Scikit-learn.
    - Bachelor's Degree in Computer Science or related field.

    Preferred Qualifications:
    - Experience with Docker and AWS cloud deployments.
    - Familiarity with CI/CD and Kubernetes.

    Responsibilities:
    - Build and scale production deep learning recommendation systems.
    - Mentor junior data scientists.
    """

    parsed = parser.parse_job_description(jd_text)

    assert parsed.title == "Senior Machine Learning Engineer"
    assert parsed.seniority == "Senior"
    assert parsed.min_total_experience_years == 4.0
    assert "Bachelor's Degree" in parsed.required_degrees

    # Check requirement types
    req_map = {r.canonical_name: r for r in parsed.requirements}

    # Python should be must-have
    assert "python" in req_map
    assert req_map["python"].req_type == RequirementType.MUST_HAVE_SKILL

    # PyTorch and Scikit-learn should be must-have
    assert "pytorch" in req_map
    assert "scikit-learn" in req_map
    assert req_map["scikit-learn"].req_type == RequirementType.MUST_HAVE_SKILL

    # Docker and Kubernetes should be preferred
    assert "docker" in req_map
    assert req_map["docker"].req_type == RequirementType.PREFERRED_SKILL
    assert "kubernetes" in req_map
    assert req_map["kubernetes"].req_type == RequirementType.PREFERRED_SKILL

    # Responsibilities extracted
    assert len(parsed.responsibilities) > 0


def test_empty_jd_handling(parser: JDParser):
    """Verify empty or whitespace-only job description returns safe empty model."""
    parsed = parser.parse_job_description("   ")
    assert parsed.title is None
    assert parsed.requirements == []
    assert parsed.detected_skills == []


def test_seniority_detection_variations(parser: JDParser):
    """Verify seniority keyword extraction."""
    t1, s1 = parser.extract_job_title_and_seniority("Staff Backend Engineer")
    assert s1 == "Staff"

    t2, s2 = parser.extract_job_title_and_seniority("Junior Python Developer")
    assert s2 == "Junior"

    t3, s3 = parser.extract_job_title_and_seniority("Lead AI Architect")
    assert s3 == "Lead"
