"""Targeted unit tests for Phase 2: External skill taxonomy, non-programming/designer skills, aliases, strict boundary matching, and claimed vs evidenced skill distinction."""

from pathlib import Path
import pytest

from resume_ranker.matching.requirement_matcher import RequirementMatcher
from resume_ranker.models import (
    DocumentMetadata,
    EvidenceReference,
    EvidenceType,
    ImportanceLevel,
    JDRequirement,
    ParsedJobDescription,
    ParsedResume,
    RequirementMatchStatus,
    RequirementType,
    SkillItem,
    TimelineAnalysis,
)
from resume_ranker.nlp.skill_extractor import SkillIntelligence
from resume_ranker.scoring.scorer import CandidateScorer


@pytest.fixture
def skill_engine() -> SkillIntelligence:
    """Fixture providing initialized SkillIntelligence engine."""
    return SkillIntelligence()


@pytest.fixture
def matcher() -> RequirementMatcher:
    """Fixture providing RequirementMatcher."""
    return RequirementMatcher()


def test_external_taxonomy_loading(skill_engine: SkillIntelligence):
    """Verify that external skill taxonomy JSON file loads properly with categories."""
    assert skill_engine.taxonomy_path is not None
    assert skill_engine.taxonomy_path.exists()
    assert len(skill_engine._flat_taxonomy) >= 100
    assert len(skill_engine._categories) >= 7

    # Verify categories
    assert "programming_languages" in skill_engine._categories
    assert "design_and_creative" in skill_engine._categories
    assert "product_and_management" in skill_engine._categories
    assert "content_marketing_and_communication" in skill_engine._categories

    # Verify category lookup
    assert skill_engine.get_skill_category("figma") == "design_and_creative"
    assert skill_engine.get_skill_category("python") == "programming_languages"
    assert skill_engine.get_skill_category("scrum") == "product_and_management"
    assert skill_engine.get_skill_category("copywriting") == "content_marketing_and_communication"


def test_designer_and_non_programming_skills_detection(skill_engine: SkillIntelligence):
    """Verify detection of UI/UX, product, management, and content writing skills."""
    designer_text = (
        "Lead Product Designer proficient in Figma, Adobe XD, and Sketch. "
        "Experienced in UI/UX Design, user research, wireframing, prototyping, and maintaining design systems."
    )
    detected = {canonical for canonical, _, _ in skill_engine.extract_skills_from_text(designer_text)}
    assert "figma" in detected
    assert "adobe xd" in detected
    assert "sketch" in detected
    assert "ui/ux design" in detected
    assert "user research" in detected
    assert "wireframing" in detected
    assert "prototyping" in detected
    assert "design systems" in detected

    pm_text = (
        "Agile Product Manager driving Scrum ceremonies, sprint planning in Jira, "
        "managing Kanban boards, and developing product management roadmaps."
    )
    pm_detected = {canonical for canonical, _, _ in skill_engine.extract_skills_from_text(pm_text)}
    assert "agile" in pm_detected
    assert "scrum" in pm_detected
    assert "jira" in pm_detected
    assert "kanban" in pm_detected
    assert "product management" in pm_detected

    content_text = (
        "Content Marketing Specialist with expertise in copywriting, content strategy, "
        "SEO blogging, storytelling, and technical writing."
    )
    content_detected = {canonical for canonical, _, _ in skill_engine.extract_skills_from_text(content_text)}
    assert "copywriting" in content_detected
    assert "content strategy" in content_detected
    assert "seo" in content_detected
    assert "storytelling" in content_detected
    assert "technical writing" in content_detected


def test_aliases_resolution(skill_engine: SkillIntelligence):
    """Verify alias mapping across technical and non-technical domains."""
    assert skill_engine.normalize_skill_name("sklearn") == "scikit-learn"
    assert skill_engine.normalize_skill_name("scikit-learn") == "scikit-learn"
    assert skill_engine.normalize_skill_name("k8s") == "kubernetes"
    assert skill_engine.normalize_skill_name("postgres") == "postgresql"
    assert skill_engine.normalize_skill_name("psql") == "postgresql"
    assert skill_engine.normalize_skill_name("adobe photoshop") == "photoshop"
    assert skill_engine.normalize_skill_name("usability testing") == "user research"
    assert skill_engine.normalize_skill_name("scrum master") == "scrum"
    assert skill_engine.normalize_skill_name("seo blogging") == "seo"
    assert skill_engine.normalize_skill_name("ux research") == "user research"
    assert skill_engine.normalize_skill_name("golang") == "go"


def test_java_vs_javascript_strict_boundaries(skill_engine: SkillIntelligence):
    """Verify Java never matches JavaScript variants and vice versa."""
    # Only JavaScript
    t1 = "Senior JavaScript developer building single page applications."
    d1 = {c for c, _, _ in skill_engine.extract_skills_from_text(t1)}
    assert "javascript" in d1
    assert "java" not in d1

    # Java Script spaced
    t2 = "Front-end engineering in Java Script and TypeScript."
    d2 = {c for c, _, _ in skill_engine.extract_skills_from_text(t2)}
    assert "javascript" in d2
    assert "java" not in d2

    # Java-Script hyphenated
    t3 = "Expert in Java-Script, HTML, and CSS."
    d3 = {c for c, _, _ in skill_engine.extract_skills_from_text(t3)}
    assert "javascript" in d3
    assert "java" not in d3

    # Only Java
    t4 = "Core Java developer specializing in Java 17 and Spring Boot microservices."
    d4 = {c for c, _, _ in skill_engine.extract_skills_from_text(t4)}
    assert "java" in d4
    assert "javascript" not in d4

    # Both Java and JavaScript
    t5 = "Full stack engineer proficient in both Java and JavaScript."
    d5 = {c for c, _, _ in skill_engine.extract_skills_from_text(t5)}
    assert "java" in d5
    assert "javascript" in d5


def test_c_cpp_csharp_dot_net_strict_boundaries(skill_engine: SkillIntelligence):
    """Verify C, C++, C#, and .NET boundary handling and non-contamination."""
    # Pure C
    c_text = "Embedded firmware engineering in C programming language and assembly."
    c_detected = {c for c, _, _ in skill_engine.extract_skills_from_text(c_text)}
    assert "c" in c_detected
    assert "c++" not in c_detected
    assert "c#" not in c_detected

    # Pure C++
    cpp_text = "Low latency trading systems built in C++ and Linux."
    cpp_detected = {c for c, _, _ in skill_engine.extract_skills_from_text(cpp_text)}
    assert "c++" in cpp_detected
    assert "c" not in cpp_detected
    assert "c#" not in cpp_detected

    # C# and .NET
    cs_text = "Enterprise architecture using C# and .NET Core."
    cs_detected = {c for c, _, _ in skill_engine.extract_skills_from_text(cs_text)}
    assert "c#" in cs_detected
    assert ".net" in cs_detected
    assert "c" not in cs_detected
    assert "c++" not in cs_detected

    # Email domain test: example.net must NOT trigger .net
    email_text = "Contact me at alice@consulting.net or visit https://myportfolio.net"
    email_detected = {c for c, _, _ in skill_engine.extract_skills_from_text(email_text)}
    assert ".net" not in email_detected

    # Ambiguous letter 'c' in general English text must NOT trigger C
    ordinary_c = "Reviewed Section C of the document in Building C with Vitamin C supplements."
    ord_detected = {c for c, _, _ in skill_engine.extract_skills_from_text(ordinary_c)}
    assert "c" not in ord_detected


def test_go_and_r_conversational_protection(skill_engine: SkillIntelligence):
    """Verify conversational 'go' and 'r' are rejected while technical usage is captured."""
    casual = "I will go to the office and write about my hobbies in sector R."
    d_casual = {c for c, _, _ in skill_engine.extract_skills_from_text(casual)}
    assert "go" not in d_casual
    assert "r" not in d_casual

    technical = "Architected backend in Golang and modeled statistical data using R programming."
    d_tech = {c for c, _, _ in skill_engine.extract_skills_from_text(technical)}
    assert "go" in d_tech
    assert "r" in d_tech


def test_claimed_vs_evidenced_skill_distinction(matcher: RequirementMatcher):
    """Verify that evidenced skills score higher than claimed-only skills."""
    req = JDRequirement(
        id="REQ-PYTHON",
        req_type=RequirementType.MUST_HAVE_SKILL,
        canonical_name="python",
        importance=ImportanceLevel.HIGH,
        original_text="Python programming experience required.",
    )

    # 1. Evidenced candidate (demonstrated in EXPERIENCE)
    evidenced_skill = SkillItem(
        canonical_name="python",
        evidence_type=EvidenceType.VERIFIED,
        confidence=1.0,
        evidence_refs=[
            EvidenceReference(section="EXPERIENCE", text_snippet="Built microservices in Python", evidence_type=EvidenceType.VERIFIED)
        ],
    )
    evidenced_resume = ParsedResume(
        document=DocumentMetadata(id="c1", filename="evidenced.pdf"),
        skills=[evidenced_skill],
    )
    res_evidenced = matcher.match_requirement(req, evidenced_resume)
    assert res_evidenced.status == RequirementMatchStatus.SATISFIED
    assert res_evidenced.score == 1.0
    assert "evidenced" in res_evidenced.explanation.lower()

    # 2. Claimed-only candidate (only listed in SKILLS list)
    claimed_skill = SkillItem(
        canonical_name="python",
        evidence_type=EvidenceType.EXPLICIT,
        confidence=0.85,
        evidence_refs=[
            EvidenceReference(section="SKILLS", text_snippet="Python, SQL, AWS", evidence_type=EvidenceType.EXPLICIT)
        ],
    )
    claimed_resume = ParsedResume(
        document=DocumentMetadata(id="c2", filename="claimed.pdf"),
        skills=[claimed_skill],
    )
    res_claimed = matcher.match_requirement(req, claimed_resume)
    # Must be recognized as claimed with lower score (0.70 vs 1.0)
    assert res_claimed.status == RequirementMatchStatus.PARTIALLY_SATISFIED
    assert res_claimed.score == 0.70
    assert "claimed" in res_claimed.explanation.lower()
    assert res_evidenced.score > res_claimed.score


def test_scored_profile_populates_evidenced_and_claimed_skills():
    """Verify that ScoredCandidateProfile separates evidenced and claimed skills."""
    scorer = CandidateScorer()
    jd = ParsedJobDescription(
        raw_text="Job",
        requirements=[
            JDRequirement(id="R1", req_type=RequirementType.MUST_HAVE_SKILL, canonical_name="python", original_text="Python"),
            JDRequirement(id="R2", req_type=RequirementType.MUST_HAVE_SKILL, canonical_name="sql", original_text="SQL"),
        ],
    )

    resume = ParsedResume(
        document=DocumentMetadata(id="1", filename="test.pdf"),
        skills=[
            SkillItem(
                canonical_name="python",
                evidence_type=EvidenceType.VERIFIED,
                evidence_refs=[EvidenceReference(section="EXPERIENCE", text_snippet="Used Python")],
            ),
            SkillItem(
                canonical_name="sql",
                evidence_type=EvidenceType.EXPLICIT,
                evidence_refs=[EvidenceReference(section="SKILLS", text_snippet="SQL")],
            ),
        ],
        timeline=TimelineAnalysis(total_experience_months=24, total_experience_years=2.0),
        raw_text="Experience: Used Python. Skills: SQL.",
    )

    profile = scorer.score_candidate_profile(resume, jd, similarity_score=0.8)
    assert "python" in profile.evidenced_skills
    assert "python" not in profile.claimed_skills
    assert "sql" in profile.claimed_skills
    assert "sql" not in profile.evidenced_skills
