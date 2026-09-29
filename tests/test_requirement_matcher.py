"""Unit tests for RequirementMatcher evaluating candidates against JD requirements."""

import pytest
from resume_ranker.matching.requirement_matcher import RequirementMatcher
from resume_ranker.models import (
    DocumentMetadata,
    EducationItem,
    EvidenceReference,
    EvidenceType,
    ExperienceItem,
    ImportanceLevel,
    JDRequirement,
    ParsedResume,
    RequirementMatchStatus,
    RequirementType,
    SkillItem,
    TimelineAnalysis,
)


@pytest.fixture
def matcher() -> RequirementMatcher:
    """Fixture providing initialized RequirementMatcher."""
    return RequirementMatcher()


def test_match_verified_skill_requirement(matcher: RequirementMatcher):
    """Verify that a verified skill with practical experience is marked SATISFIED."""
    req = JDRequirement(
        id="REQ-01",
        req_type=RequirementType.MUST_HAVE_SKILL,
        canonical_name="python",
        importance=ImportanceLevel.CRITICAL,
        original_text="Proficiency in Python programming.",
    )
    skill = SkillItem(
        canonical_name="python",
        evidence_type=EvidenceType.VERIFIED,
        confidence=1.0,
        evidence_refs=[
            EvidenceReference(section="EXPERIENCE", text_snippet="Built Python APIs", evidence_type=EvidenceType.VERIFIED)
        ],
    )
    doc_meta = DocumentMetadata(id="1", filename="a.pdf")
    resume = ParsedResume(document=doc_meta, skills=[skill])

    result = matcher.match_requirement(req, resume)
    assert result.status == RequirementMatchStatus.SATISFIED
    assert result.score == 1.0
    assert len(result.evidence_refs) == 1
    assert "verified" in result.explanation.lower()


def test_match_missing_skill_requirement(matcher: RequirementMatcher):
    """Verify that an unmentioned skill is marked NOT_FOUND with score 0."""
    req = JDRequirement(
        id="REQ-02",
        req_type=RequirementType.MUST_HAVE_SKILL,
        canonical_name="kubernetes",
        importance=ImportanceLevel.HIGH,
        original_text="Experience with Kubernetes orchestration.",
    )
    doc_meta = DocumentMetadata(id="1", filename="a.pdf")
    resume = ParsedResume(document=doc_meta, skills=[])

    result = matcher.match_requirement(req, resume)
    assert result.status == RequirementMatchStatus.NOT_FOUND
    assert result.score == 0.0
    assert "not detected" in result.explanation.lower()


def test_match_experience_requirement_exceeded(matcher: RequirementMatcher):
    """Verify meeting and exceeding minimum experience requirement."""
    req = JDRequirement(
        id="REQ-03",
        req_type=RequirementType.MINIMUM_EXPERIENCE,
        canonical_name="3+ years experience",
        min_experience_months=36,
        original_text="At least 3 years software engineering experience.",
    )
    timeline = TimelineAnalysis(total_experience_months=48, total_experience_years=4.0)
    doc_meta = DocumentMetadata(id="1", filename="a.pdf")
    resume = ParsedResume(document=doc_meta, timeline=timeline)

    result = matcher.match_requirement(req, resume)
    assert result.status == RequirementMatchStatus.SATISFIED
    assert result.score == 1.0
    assert "4.0 years" in result.explanation


def test_match_experience_requirement_partial(matcher: RequirementMatcher):
    """Verify partial fulfillment of experience requirement."""
    req = JDRequirement(
        id="REQ-04",
        req_type=RequirementType.MINIMUM_EXPERIENCE,
        canonical_name="5+ years experience",
        min_experience_months=60,
        original_text="5+ years of software experience required.",
    )
    timeline = TimelineAnalysis(total_experience_months=24, total_experience_years=2.0)
    doc_meta = DocumentMetadata(id="1", filename="a.pdf")
    resume = ParsedResume(document=doc_meta, timeline=timeline)

    result = matcher.match_requirement(req, resume)
    assert result.status == RequirementMatchStatus.PARTIALLY_SATISFIED
    assert 0.0 < result.score < 1.0
    assert "2.0 years" in result.explanation


def test_match_education_requirement(matcher: RequirementMatcher):
    """Verify education matching against candidate academic history."""
    req = JDRequirement(
        id="REQ-05",
        req_type=RequirementType.EDUCATION,
        canonical_name="bachelor's degree",
        original_text="Bachelor's Degree in Computer Science or related.",
    )
    edu = EducationItem(institution="MIT", degree="Bachelor of Science", field_of_study="Computer Science")
    doc_meta = DocumentMetadata(id="1", filename="a.pdf")
    resume = ParsedResume(document=doc_meta, education=[edu])

    result = matcher.match_requirement(req, resume)
    assert result.status == RequirementMatchStatus.SATISFIED
    assert result.score == 1.0
