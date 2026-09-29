"""Unit tests for Phase 1: Scoring integrity, real evidence, dynamic renormalization, and analysis confidence."""

import pytest
from resume_ranker.extraction.quality_assessor import DocumentQualityReport
from resume_ranker.matching.requirement_matcher import RequirementMatcher
from resume_ranker.models import (
    AnomalyFlag,
    AnomalySeverity,
    DocumentMetadata,
    EducationItem,
    EvidenceReference,
    EvidenceType,
    ExperienceItem,
    ImportanceLevel,
    JDRequirement,
    ParsedJobDescription,
    ParsedResume,
    RequirementMatchStatus,
    RequirementType,
    SkillItem,
    TimelineAnalysis,
)
from resume_ranker.scoring.scorer import CandidateScorer


@pytest.fixture
def scorer() -> CandidateScorer:
    """Fixture providing initialized CandidateScorer."""
    return CandidateScorer()


def test_case_a_missing_experience_not_penalized_when_jd_omits(scorer: CandidateScorer):
    """Case A1: When JD does not require experience and candidate has none, experience_score must be None (not 50.0)."""
    jd_reqs = [
        JDRequirement(
            id="REQ-01",
            req_type=RequirementType.MUST_HAVE_SKILL,
            canonical_name="python",
            importance=ImportanceLevel.HIGH,
            original_text="Python knowledge required.",
        )
    ]
    parsed_jd = ParsedJobDescription(raw_text="Python developer", requirements=jd_reqs)

    # Resume with Python skill but zero experience timeline
    skills = [
        SkillItem(
            canonical_name="python",
            evidence_type=EvidenceType.EXPLICIT,
            evidence_refs=[EvidenceReference(section="SKILLS", text_snippet="Python")],
        )
    ]
    resume = ParsedResume(
        document=DocumentMetadata(id="1", filename="fresh_grad.pdf"),
        skills=skills,
        timeline=TimelineAnalysis(total_experience_months=0, total_experience_years=0.0),
        raw_text="Fresh graduate skilled in Python.",
    )

    profile = scorer.score_candidate_profile(parsed_resume=resume, parsed_jd=parsed_jd, similarity_score=0.80)

    # Experience must be None ("Not found") and NOT 50.0
    assert profile.experience_score is None
    assert "Experience: Not found" in profile.explanation
    # Subscore for required skills must be evaluated
    assert profile.required_skill_score is not None
    assert profile.required_skill_score > 0.0
    # Dynamic renormalization must ensure final score is computed over active components
    assert profile.final_score > 0.0


def test_case_a_missing_experience_scores_zero_when_required(scorer: CandidateScorer):
    """Case A2: When JD requires minimum experience and candidate has none, score must be 0.0 (not 50.0)."""
    jd_reqs = [
        JDRequirement(
            id="REQ-EXP",
            req_type=RequirementType.MINIMUM_EXPERIENCE,
            canonical_name="3+ years experience",
            min_experience_months=36,
            importance=ImportanceLevel.HIGH,
            original_text="3+ years experience required.",
        )
    ]
    parsed_jd = ParsedJobDescription(raw_text="Need experienced dev", requirements=jd_reqs)

    resume = ParsedResume(
        document=DocumentMetadata(id="2", filename="no_exp.pdf"),
        skills=[],
        timeline=TimelineAnalysis(total_experience_months=0, total_experience_years=0.0),
        raw_text="Applicant with no experience.",
    )

    profile = scorer.score_candidate_profile(parsed_resume=resume, parsed_jd=parsed_jd, similarity_score=0.20)

    # Must be 0.0 because it was required and not met
    assert profile.experience_score == 0.0
    assert any(
        m.req_type == RequirementType.MINIMUM_EXPERIENCE and m.status == RequirementMatchStatus.NOT_FOUND
        for m in profile.requirement_matches
    )


def test_case_b_missing_education_not_penalized_when_jd_omits(scorer: CandidateScorer):
    """Case B1: When JD does not specify education, education_score must be None (not 100.0)."""
    jd_reqs = [
        JDRequirement(
            id="REQ-01",
            req_type=RequirementType.MUST_HAVE_SKILL,
            canonical_name="python",
            importance=ImportanceLevel.HIGH,
            original_text="Python knowledge required.",
        )
    ]
    parsed_jd = ParsedJobDescription(raw_text="Python developer", requirements=jd_reqs)

    resume = ParsedResume(
        document=DocumentMetadata(id="3", filename="no_edu.pdf"),
        skills=[SkillItem(canonical_name="python", evidence_type=EvidenceType.EXPLICIT)],
        education=[],
        raw_text="Self-taught developer.",
    )

    profile = scorer.score_candidate_profile(parsed_resume=resume, parsed_jd=parsed_jd, similarity_score=0.70)

    assert profile.education_score is None
    assert "Education: Not specified in JD" in profile.explanation


def test_case_b_missing_education_scores_zero_when_required(scorer: CandidateScorer):
    """Case B2: When JD requires degree and candidate has no education entries, score must be 0.0 (not 0.5 or 50.0)."""
    jd_reqs = [
        JDRequirement(
            id="REQ-EDU",
            req_type=RequirementType.EDUCATION,
            canonical_name="bachelor's degree",
            importance=ImportanceLevel.HIGH,
            original_text="Bachelor's degree required.",
        )
    ]
    parsed_jd = ParsedJobDescription(raw_text="Degree required", requirements=jd_reqs)

    resume = ParsedResume(
        document=DocumentMetadata(id="4", filename="no_degree.pdf"),
        skills=[],
        education=[],
        raw_text="No degree listed.",
    )

    profile = scorer.score_candidate_profile(parsed_resume=resume, parsed_jd=parsed_jd, similarity_score=0.10)

    assert profile.education_score == 0.0
    assert any(
        m.req_type == RequirementType.EDUCATION and m.status == RequirementMatchStatus.NOT_FOUND and m.score == 0.0
        for m in profile.requirement_matches
    )


def test_case_c_dynamic_renormalization_weights(scorer: CandidateScorer):
    """Case C: Active component weights renormalize dynamically when preferred skills and education are omitted."""
    # Only similarity and required skills are active
    jd_reqs = [
        JDRequirement(
            id="REQ-01",
            req_type=RequirementType.MUST_HAVE_SKILL,
            canonical_name="python",
            importance=ImportanceLevel.CRITICAL,
            original_text="Python required.",
        )
    ]
    parsed_jd = ParsedJobDescription(raw_text="Python only", requirements=jd_reqs)

    resume = ParsedResume(
        document=DocumentMetadata(id="5", filename="cand.pdf"),
        skills=[
            SkillItem(
                canonical_name="python",
                evidence_type=EvidenceType.VERIFIED,
                evidence_refs=[EvidenceReference(section="EXPERIENCE", text_snippet="Worked with Python")],
            )
        ],
        raw_text="Python engineer with verified experience.",
    )

    sim_score = 0.80  # 80% similarity
    profile = scorer.score_candidate_profile(parsed_resume=resume, parsed_jd=parsed_jd, similarity_score=sim_score)

    # Active weights: similarity (0.50) + required_skills (0.30) = 0.80 total
    # Req skill score = 100.0, similarity = 80.0
    # Expected weighted score = (80.0 * 0.50 + 100.0 * 0.30) / 0.80 = (40.0 + 30.0) / 0.80 = 70.0 / 0.80 = 87.5
    assert profile.required_skill_score == 100.0
    assert profile.preferred_skill_score is None
    assert profile.experience_score is None
    assert profile.education_score is None
    assert profile.final_score == 87.5


def test_case_d_analysis_confidence_bounds_and_realism(scorer: CandidateScorer):
    """Case D: Analysis confidence never claims 100% (capped at 0.88) and responds to anomalies and text quality."""
    resume = ParsedResume(
        document=DocumentMetadata(id="6", filename="clean.pdf"),
        skills=[
            SkillItem(
                canonical_name="python",
                evidence_type=EvidenceType.VERIFIED,
                evidence_refs=[EvidenceReference(section="EXPERIENCE", text_snippet="Python dev")],
            )
        ],
        timeline=TimelineAnalysis(total_experience_months=24, total_experience_years=2.0),
        sections_detected={"HEADER": ["Jane Doe"], "EXPERIENCE": ["Python Dev"], "SKILLS": ["Python"], "EDUCATION": ["BS CS"]},
        raw_text="Jane Doe. Experience: 2 years Python. Skills: Python. Education: BS Computer Science. " * 15,
    )

    quality = DocumentQualityReport(
        page_count=1,
        total_chars=1500,
        total_words=300,
        confidence=1.0,
    )

    clean_conf = scorer.compute_analysis_confidence(resume, quality_report=quality, anomalies=[])
    # Must not claim 100% certainty for self-reported resumes
    assert clean_conf <= 0.88
    assert clean_conf >= 0.80

    # With a high-severity anomaly (prompt injection or date contradiction), confidence drops
    high_anomaly = AnomalyFlag(
        code="PROMPT_INJECTION_DETECTED",
        severity=AnomalySeverity.HIGH,
        message="Prompt injection detected.",
    )
    anomaly_conf = scorer.compute_analysis_confidence(resume, quality_report=quality, anomalies=[high_anomaly])
    assert anomaly_conf < clean_conf
    assert anomaly_conf <= 0.65


def test_case_e_score_bounds_and_determinism(scorer: CandidateScorer):
    """Case E: Scoring is deterministic and strictly bounded within [0.0, 100.0]."""
    jd_reqs = [
        JDRequirement(
            id="REQ-01",
            req_type=RequirementType.MUST_HAVE_SKILL,
            canonical_name="sql",
            original_text="SQL expertise",
        )
    ]
    parsed_jd = ParsedJobDescription(raw_text="SQL job", requirements=jd_reqs)
    resume = ParsedResume(
        document=DocumentMetadata(id="7", filename="det.pdf"),
        skills=[SkillItem(canonical_name="sql", evidence_type=EvidenceType.EXPLICIT)],
        raw_text="SQL applicant.",
    )

    run1 = scorer.score_candidate_profile(parsed_resume=resume, parsed_jd=parsed_jd, similarity_score=0.65)
    run2 = scorer.score_candidate_profile(parsed_resume=resume, parsed_jd=parsed_jd, similarity_score=0.65)

    assert run1.final_score == run2.final_score
    assert run1.confidence == run2.confidence
    assert 0.0 <= run1.final_score <= 100.0
