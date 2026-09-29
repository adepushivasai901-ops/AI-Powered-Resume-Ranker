"""Unit tests for deterministic resume section detector."""

import pytest
from resume_ranker.extraction.section_detector import SectionDetector
from resume_ranker.models.evidence import DocumentBlock


@pytest.fixture
def detector() -> SectionDetector:
    """Fixture providing initialized SectionDetector."""
    return SectionDetector()


def test_classify_canonical_headings(detector: SectionDetector):
    """Verify recognition of common canonical heading variants."""
    assert detector.classify_line_as_heading("SUMMARY") == "SUMMARY"
    assert detector.classify_line_as_heading("Professional Summary") == "SUMMARY"
    assert detector.classify_line_as_heading("CAREER OBJECTIVE") == "SUMMARY"

    assert detector.classify_line_as_heading("WORK EXPERIENCE") == "EXPERIENCE"
    assert detector.classify_line_as_heading("Employment History") == "EXPERIENCE"
    assert detector.classify_line_as_heading("Professional Experience") == "EXPERIENCE"

    assert detector.classify_line_as_heading("EDUCATION") == "EDUCATION"
    assert detector.classify_line_as_heading("Academic Background") == "EDUCATION"

    assert detector.classify_line_as_heading("TECHNICAL SKILLS") == "SKILLS"
    assert detector.classify_line_as_heading("Core Competencies") == "SKILLS"
    assert detector.classify_line_as_heading("Tech Stack") == "SKILLS"

    assert detector.classify_line_as_heading("PROJECTS") == "PROJECTS"
    assert detector.classify_line_as_heading("Key Projects") == "PROJECTS"

    assert detector.classify_line_as_heading("CERTIFICATIONS") == "CERTIFICATIONS"
    assert detector.classify_line_as_heading("Licenses & Certifications") == "CERTIFICATIONS"

    assert detector.classify_line_as_heading("ACHIEVEMENTS") == "ACHIEVEMENTS"
    assert detector.classify_line_as_heading("Honors & Awards") == "ACHIEVEMENTS"

    assert detector.classify_line_as_heading("PUBLICATIONS") == "PUBLICATIONS"
    assert detector.classify_line_as_heading("LANGUAGES") == "LANGUAGES"
    assert detector.classify_line_as_heading("INTERESTS") == "INTERESTS"


def test_classify_headings_with_decorators_and_formatting(detector: SectionDetector):
    """Verify headers with markdown, dashes, numbers, and trailing colons."""
    assert detector.classify_line_as_heading("### Work Experience:") == "EXPERIENCE"
    assert detector.classify_line_as_heading("--- EDUCATION ---") == "EDUCATION"
    assert detector.classify_line_as_heading("1. Technical Skills") == "SKILLS"
    assert detector.classify_line_as_heading("**Certifications**") == "CERTIFICATIONS"


def test_negative_cases_ordinary_prose_not_headings(detector: SectionDetector):
    """Verify normal prose sentences containing section words are NOT detected as headings."""
    assert detector.classify_line_as_heading("I have 5 years of professional experience in software.") is None
    assert detector.classify_line_as_heading("My education at MIT taught me machine learning fundamentals.") is None
    assert detector.classify_line_as_heading("Developed strong technical skills in Python and PyTorch.") is None
    assert detector.classify_line_as_heading("Summary: A passionate developer who loves cloud infrastructure.") is None
    assert detector.classify_line_as_heading("Projects completed include recommendation systems and search engines.") is None


def test_segment_text_full_resume(detector: SectionDetector):
    """Verify end-to-end segmentation of multi-section resume text."""
    resume_text = """
    Alice Johnson
    Senior Machine Learning Engineer
    alice@example.com

    SUMMARY
    Machine learning specialist with 5 years experience in deep learning.

    EXPERIENCE
    Staff AI Engineer - TechCorp (2021 - Present)
    - Designed real-time NLP classification models.
    - Mentored junior engineers.

    EDUCATION
    Master of Science in Computer Science
    Stanford University (2018 - 2020)

    TECHNICAL SKILLS
    Python, PyTorch, Scikit-learn, SQL, Docker, AWS
    """

    sections = detector.segment_text(resume_text)

    assert "HEADER" in sections
    assert any("Alice Johnson" in line for line in sections["HEADER"])

    assert "SUMMARY" in sections
    assert any("5 years experience" in line for line in sections["SUMMARY"])

    assert "EXPERIENCE" in sections
    assert any("TechCorp" in line for line in sections["EXPERIENCE"])

    assert "EDUCATION" in sections
    assert any("Stanford University" in line for line in sections["EDUCATION"])

    assert "SKILLS" in sections
    assert any("PyTorch" in line for line in sections["SKILLS"])


def test_tag_document_blocks(detector: SectionDetector):
    """Verify section tagging onto DocumentBlock instances."""
    blocks = [
        DocumentBlock(block_id="b1", page=1, text="John Doe", reading_order=0),
        DocumentBlock(block_id="b2", page=1, text="EXPERIENCE", reading_order=1),
        DocumentBlock(block_id="b3", page=1, text="Software Engineer at Acme (2020 - 2023)", reading_order=2),
        DocumentBlock(block_id="b4", page=1, text="SKILLS", reading_order=3),
        DocumentBlock(block_id="b5", page=1, text="Python, SQL, C++", reading_order=4),
    ]

    tagged = detector.tag_blocks(blocks)
    assert tagged[0].section == "HEADER"
    assert tagged[1].section == "EXPERIENCE"
    assert tagged[2].section == "EXPERIENCE"
    assert tagged[3].section == "SKILLS"
    assert tagged[4].section == "SKILLS"
