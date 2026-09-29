"""Unit tests for Phase 1 canonical domain models and evidence schemas."""

import pytest
from pydantic import ValidationError

from resume_ranker.models import (
    AnomalyFlag,
    AnomalySeverity,
    CandidateInfo,
    CertificationItem,
    DocumentBlock,
    DocumentMetadata,
    EducationItem,
    EvidenceReference,
    EvidenceType,
    ExperienceItem,
    ImportanceLevel,
    JDRequirement,
    ParsedJobDescription,
    ParsedResume,
    ProjectItem,
    Provenance,
    RequirementMatchResult,
    RequirementMatchStatus,
    RequirementType,
    ScoredCandidateProfile,
    SkillItem,
    TimelineAnalysis,
    TimelineGap,
    TimelineOverlap,
)


def test_evidence_type_enum():
    """Verify all required evidence certainty types exist."""
    assert EvidenceType.EXPLICIT == "EXPLICIT"
    assert EvidenceType.VERIFIED == "VERIFIED"
    assert EvidenceType.INFERRED == "INFERRED"
    assert EvidenceType.UNCERTAIN == "UNCERTAIN"
    assert EvidenceType.MISSING == "MISSING"


def test_document_block_and_evidence_reference():
    """Verify document block creation and linking into evidence reference."""
    block = DocumentBlock(
        block_id="b-001",
        page=1,
        text="Proficient in Python, SQL, and Docker for scalable services.",
        section="SKILLS",
        reading_order=2,
        source="pypdf",
        confidence=0.98,
    )
    assert block.block_id == "b-001"
    assert block.page == 1
    assert block.section == "SKILLS"

    ref = EvidenceReference(
        block_id=block.block_id,
        section=block.section,
        text_snippet=block.text,
        page=block.page,
        evidence_type=EvidenceType.VERIFIED,
        confidence=0.95,
        note="Observed in Technical Skills section",
    )
    assert ref.block_id == "b-001"
    assert ref.evidence_type == EvidenceType.VERIFIED
    assert ref.page == 1


def test_experience_item_validation():
    """Verify ExperienceItem schema instantiation with evidence references."""
    exp = ExperienceItem(
        company="Acme AI Corp",
        raw_title="Senior Machine Learning Engineer",
        normalized_title="Machine Learning Engineer",
        start_date="2021-03",
        end_date="2024-01",
        is_current=False,
        duration_months=34,
        employment_type="Full-time",
        responsibilities=["Trained transformer models using PyTorch."],
        achievements=["Reduced inference latency by 45%."],
        technologies_used=["Python", "PyTorch", "AWS"],
    )
    assert exp.company == "Acme AI Corp"
    assert exp.duration_months == 34
    assert len(exp.technologies_used) == 3


def test_skill_item_with_aliases_and_provenance():
    """Verify SkillItem captures aliases, duration, and evidence."""
    skill = SkillItem(
        canonical_name="scikit-learn",
        aliases_detected=["sklearn", "scikit-learn"],
        evidence_type=EvidenceType.VERIFIED,
        confidence=1.0,
        first_used="2020-01",
        last_used="2023-12",
        duration_months=48,
    )
    assert skill.canonical_name == "scikit-learn"
    assert "sklearn" in skill.aliases_detected
    assert skill.duration_months == 48


def test_timeline_analysis():
    """Verify TimelineAnalysis handles overlaps and gaps."""
    gap = TimelineGap(start_date="2020-06", end_date="2020-11", duration_months=5)
    overlap = TimelineOverlap(role_1="Role A", role_2="Role B", overlap_months=3)
    timeline = TimelineAnalysis(
        total_experience_months=42,
        total_experience_years=3.5,
        earliest_date="2018-01",
        latest_date="2023-12",
        gaps=[gap],
        overlaps=[overlap],
    )
    assert timeline.total_experience_years == 3.5
    assert len(timeline.gaps) == 1
    assert timeline.gaps[0].duration_months == 5
    assert len(timeline.overlaps) == 1


def test_parsed_resume_full_roundtrip():
    """Verify complete ParsedResume model serialization to dict and JSON."""
    doc_meta = DocumentMetadata(
        id="doc-1234",
        filename="alice_ml_resume.pdf",
        source_type="pdf",
        page_count=2,
        parser_path="pypdf",
        parse_confidence=0.96,
        warnings=[],
    )
    cand_info = CandidateInfo(
        name="Alice Johnson",
        email="alice@example.com",
        location="San Francisco, CA",
        links=["https://github.com/alice"],
    )
    provenance = Provenance(parser="pypdf", parser_version="6.19.0", model_name="en_core_web_sm")

    resume = ParsedResume(
        document=doc_meta,
        candidate=cand_info,
        summary="Experienced ML engineer building NLP solutions.",
        skills=[
            SkillItem(canonical_name="python", evidence_type=EvidenceType.VERIFIED),
            SkillItem(canonical_name="pytorch", evidence_type=EvidenceType.EXPLICIT),
        ],
        provenance=provenance,
        raw_text="Alice Johnson Python PyTorch ML Engineer",
    )

    data_dict = resume.model_dump()
    assert data_dict["document"]["filename"] == "alice_ml_resume.pdf"
    assert data_dict["candidate"]["name"] == "Alice Johnson"
    assert len(data_dict["skills"]) == 2

    json_str = resume.model_dump_json()
    assert "alice@example.com" in json_str


def test_parsed_job_description():
    """Verify JDRequirement and ParsedJobDescription schema."""
    req1 = JDRequirement(
        id="REQ-01",
        req_type=RequirementType.MUST_HAVE_SKILL,
        canonical_name="python",
        importance=ImportanceLevel.CRITICAL,
        min_experience_months=36,
        original_text="3+ years of professional Python programming.",
        confidence=1.0,
    )
    req2 = JDRequirement(
        id="REQ-02",
        req_type=RequirementType.PREFERRED_SKILL,
        canonical_name="docker",
        importance=ImportanceLevel.MEDIUM,
        original_text="Experience with containerization using Docker.",
    )
    jd = ParsedJobDescription(
        raw_text="Senior Python Developer needed. 3+ years Python. Docker is a plus.",
        title="Senior Python Developer",
        seniority="Senior",
        min_total_experience_years=3.0,
        requirements=[req1, req2],
        detected_skills=["python", "docker"],
    )
    assert jd.seniority == "Senior"
    assert len(jd.requirements) == 2
    assert jd.requirements[0].canonical_name == "python"
    assert jd.requirements[0].importance == ImportanceLevel.CRITICAL


def test_scored_candidate_profile():
    """Verify ScoredCandidateProfile score validation and anomaly tracking."""
    match_res = RequirementMatchResult(
        requirement_id="REQ-01",
        canonical_name="python",
        status=RequirementMatchStatus.SATISFIED,
        score=1.0,
        confidence=0.95,
        explanation="Candidate demonstrates 4+ years of Python across two roles.",
    )
    anomaly = AnomalyFlag(
        code="OVERLAPPING_DATES",
        severity=AnomalySeverity.LOW,
        message="Candidate worked two concurrent roles for 2 months.",
    )
    profile = ScoredCandidateProfile(
        candidate_id="cand-001",
        filename="alice.pdf",
        rank=1,
        final_score=88.5,
        similarity_score=82.0,
        required_skill_score=95.0,
        preferred_skill_score=80.0,
        experience_score=90.0,
        requirement_matches=[match_res],
        matched_required_skills=["python"],
        missing_required_skills=[],
        anomalies=[anomaly],
        total_experience_years=4.5,
        explanation="Candidate meets all critical requirements with solid evidence.",
    )
    assert profile.final_score == 88.5
    assert profile.requirement_matches[0].status == RequirementMatchStatus.SATISFIED
    assert len(profile.anomalies) == 1
    assert profile.anomalies[0].severity == AnomalySeverity.LOW


def test_score_validation_bounds():
    """Verify that score bounds strictly enforce 0.0 to 100.0."""
    with pytest.raises(ValidationError):
        ScoredCandidateProfile(candidate_id="test", final_score=105.0)  # > 100

    with pytest.raises(ValidationError):
        ScoredCandidateProfile(candidate_id="test", final_score=-5.0)  # < 0
