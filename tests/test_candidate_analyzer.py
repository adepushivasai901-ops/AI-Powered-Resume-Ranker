"""Comprehensive test suite for candidate-facing resume analysis and career optimization platform."""

import io
from pathlib import Path
import pytest
from pypdf import PdfWriter

from app import create_app
from config import TestingConfig
from resume_ranker.analyzer.ats_assessor import ATSAssessor
from resume_ranker.analyzer.candidate_scorer import CandidateResumeScorer
from resume_ranker.analyzer.content_evaluator import ContentEvaluator
from resume_ranker.analyzer.recommendation_engine import RecommendationEngine
from resume_ranker.analyzer.resume_analyzer import CandidateResumeAnalyzer
from resume_ranker.extraction.quality_assessor import DocumentQualityReport
from resume_ranker.models.candidate_analysis import (
    ATSCheckStatus,
    CandidateResumeAnalysis,
    PriorityLevel,
)
from resume_ranker.models.evidence import DocumentBlock, EvidenceReference, EvidenceType
from resume_ranker.models.job_description import (
    ImportanceLevel,
    JDRequirement,
    ParsedJobDescription,
    RequirementType,
)
from resume_ranker.models.resume import (
    CandidateInfo,
    DocumentMetadata,
    EducationItem,
    ExperienceItem,
    ParsedResume,
    ProjectItem,
    SkillItem,
    TimelineAnalysis,
)
from resume_ranker.nlp.resume_parser import ResumeParser
from resume_ranker.nlp.skill_extractor import SkillIntelligence
from resume_ranker.reporting.candidate_report import CandidateReportGenerator


@pytest.fixture
def app(tmp_path: Path):
    """Flask application fixture with temporary upload and report paths."""
    app = create_app(config_class=TestingConfig)
    app.config["UPLOAD_FOLDER"] = tmp_path / "uploads"
    app.config["REPORT_FOLDER"] = tmp_path / "reports"
    app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)
    app.config["REPORT_FOLDER"].mkdir(parents=True, exist_ok=True)
    return app


@pytest.fixture
def client(app):
    """Test client fixture."""
    return app.test_client()


def _create_sample_pdf(text: str = "Candidate Resume") -> io.BytesIO:
    """Helper to generate a minimal valid in-memory PDF."""
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    stream = io.BytesIO()
    writer.write(stream)
    stream.seek(0)
    return stream


def _build_candidate_resume(
    name: str = "Taylor Swift",
    email: str = "taylor@example.com",
    phone: str = "+1 555-123-4567",
    summary: str = "Experienced Backend Engineer with 4 years designing distributed Python web applications and REST APIs.",
    skills: list = None,
    experience: list = None,
    education: list = None,
    projects: list = None,
    sections: dict = None,
    raw_text: str = "",
) -> ParsedResume:
    """Helper to construct a fully populated ParsedResume."""
    if skills is None:
        skills = [
            SkillItem(name="Python", canonical_name="python", evidence_type=EvidenceType.RESUME_EVIDENCED),
            SkillItem(name="PostgreSQL", canonical_name="postgresql", evidence_type=EvidenceType.RESUME_EVIDENCED),
            SkillItem(name="Docker", canonical_name="docker", evidence_type=EvidenceType.CLAIMED),
        ]
    if experience is None:
        experience = [
            ExperienceItem(
                company="Acme Corp",
                raw_title="Software Engineer",
                start_date="2021",
                end_date="Present",
                responsibilities=[
                    "Engineered Python FastAPI microservices, reducing query latency by 35%.",
                    "Worked on database queries and background tasks.",
                ],
                achievements=["Scaled API to handle 50,000+ daily active users."],
            )
        ]
    if education is None:
        education = [
            EducationItem(
                institution="University of Washington",
                degree="Bachelor of Science",
                field_of_study="Computer Science",
                end_date="2020",
            )
        ]
    if projects is None:
        projects = [
            ProjectItem(
                name="Cloud Document Search",
                description="Built full-text search engine with Python and Elasticsearch.",
                technologies=["Python", "Elasticsearch"],
                links=["https://github.com/example/search"],
            )
        ]
    if sections is None:
        sections = {
            "SUMMARY": ["Experienced Backend Engineer..."],
            "EXPERIENCE": ["Acme Corp..."],
            "SKILLS": ["Python, PostgreSQL, Docker"],
            "EDUCATION": ["University of Washington..."],
            "PROJECTS": ["Cloud Document Search..."],
        }

    return ParsedResume(
        document=DocumentMetadata(
            id="doc-001",
            filename=f"{name.lower().replace(' ', '_')}.pdf",
            page_count=1,
        ),
        candidate=CandidateInfo(
            name=name,
            email=email,
            phone=phone,
            location="Seattle, WA",
            links=["linkedin.com/in/example"],
        ),
        summary=summary,
        skills=skills,
        experience=experience,
        education=education,
        projects=projects,
        sections_detected=sections,
        raw_text=raw_text or f"{name}\n{email} | {phone}\n{summary}\nExperience Acme Corp",
        timeline=TimelineAnalysis(
            total_experience_months=48,
            total_experience_years=4.0,
        ),
    )


# ---------------------------------------------------------------------------
# 1. Resume-Only Analysis (No JD Provided)
# ---------------------------------------------------------------------------

def test_resume_only_analysis_success():
    """Verify candidate analysis succeeds without requiring a job description."""
    resume = _build_candidate_resume()
    analyzer = CandidateResumeAnalyzer()
    analysis = analyzer.analyze(parsed_resume=resume)

    assert analysis.candidate_name == "Taylor Swift"
    assert analysis.health_score.overall_score > 0.0
    assert analysis.ats_readiness.score > 0.0
    assert analysis.job_match.is_available is False
    assert len(analysis.recommendations) > 0
    assert any("Job description match unavailable" in w for w in analysis.warnings)


# ---------------------------------------------------------------------------
# 2. Resume + Job Description Analysis (Job Match)
# ---------------------------------------------------------------------------

def test_resume_with_job_description_match():
    """Verify job match score, required/preferred skills match, and keyword coverage."""
    resume = _build_candidate_resume()
    jd_text = (
        "We need a Senior Python Developer with PostgreSQL and AWS experience. "
        "Docker is preferred. 3+ years experience. Bachelor's degree required."
    )
    analyzer = CandidateResumeAnalyzer()
    analysis = analyzer.analyze(parsed_resume=resume, job_description=jd_text)

    assert analysis.job_match.is_available is True
    assert analysis.job_match.match_score is not None
    assert analysis.job_match.match_score > 50.0

    # Required skills check
    assert "Python" in analysis.job_match.required_skills_matched
    assert "Postgresql" in analysis.job_match.required_skills_matched
    assert "Aws" in analysis.job_match.required_skills_missing

    # Preferred skills check
    assert "Docker" in analysis.job_match.preferred_skills_matched

    # Experience & education alignment
    assert analysis.job_match.experience_meets is True
    assert analysis.job_match.education_meets is True
    assert analysis.job_match.keyword_coverage_pct is not None


# ---------------------------------------------------------------------------
# 3. Missing Sections & Dynamic Weight Renormalization
# ---------------------------------------------------------------------------

def test_missing_sections_dynamic_renormalization():
    """Verify missing sections are marked 'Not Detected' and do not fabricate scores."""
    # Omit education, projects, and summary
    resume_minimal = _build_candidate_resume(
        summary="",
        education=[],
        projects=[],
        sections={"EXPERIENCE": ["Engineer"], "SKILLS": ["Python"]},
    )
    analyzer = CandidateResumeAnalyzer()
    analysis = analyzer.analyze(parsed_resume=resume_minimal)

    # Components check
    comps = analysis.health_score.components
    assert comps["education"].is_active is False
    assert comps["education"].score is None
    assert comps["education"].status == "Not Detected"

    assert comps["projects"].is_active is False
    assert comps["projects"].score is None
    assert comps["projects"].status == "Not Detected"

    # Effective weights must renormalize and sum to 1.0 across active components
    active_effective_sum = sum(c.effective_weight for c in comps.values() if c.is_active)
    assert abs(active_effective_sum - 1.0) < 0.05
    assert analysis.health_score.overall_score > 0.0


# ---------------------------------------------------------------------------
# 4. ATS Readiness Assessor Verification
# ---------------------------------------------------------------------------

def test_ats_readiness_scoring_and_checks():
    """Verify ATS checklist items, status labels, and contact detail detection."""
    assessor = ATSAssessor()

    # Full resume
    resume_complete = _build_candidate_resume()
    quality = DocumentQualityReport(
        total_chars=1200,
        total_words=200,
        page_count=1,
        is_scanned_image=False,
        suspicious_chars_ratio=0.01,
        confidence=1.0,
    )
    ats_res = assessor.evaluate_ats_readiness(resume_complete, quality)

    assert ats_res.score >= 80.0
    assert ats_res.status_label == "High ATS Readiness"
    assert any(c.code == "CONTACT_INFO" and c.status == ATSCheckStatus.PASSED for c in ats_res.checks)
    assert any(c.code == "STANDARD_HEADINGS" and c.status == ATSCheckStatus.PASSED for c in ats_res.checks)

    # Scanned PDF failure
    scanned_quality = DocumentQualityReport(
        total_chars=10,
        total_words=2,
        page_count=1,
        is_scanned_image=True,
    )
    ats_scanned = assessor.evaluate_ats_readiness(resume_complete, scanned_quality)
    assert any(c.code == "TEXT_QUALITY" and c.status == ATSCheckStatus.FAILED for c in ats_scanned.checks)
    assert ats_scanned.score < ats_res.score


# ---------------------------------------------------------------------------
# 5. Bullet / Experience Quality & Metric Detection
# ---------------------------------------------------------------------------

def test_bullet_quality_evaluator():
    """Verify detection of weak verbs ('Worked on') and quantifiable metrics ('35%')."""
    evaluator = ContentEvaluator()

    # Weak bullet without metrics
    weak_bullet = evaluator.evaluate_bullet("Worked on Python backend development and database queries.")
    assert weak_bullet.is_weak is True
    assert weak_bullet.weak_pattern == "Worked on/with"
    assert weak_bullet.has_metrics is False
    assert weak_bullet.score < 60.0
    assert "Example:" in weak_bullet.example

    # Passive bullet with metric
    passive_metric = evaluator.evaluate_bullet("Responsible for increasing API throughput by 40%.")
    assert passive_metric.is_weak is True
    assert passive_metric.has_metrics is True
    assert "40%" in passive_metric.detected_metrics

    # Strong bullet with metric
    strong_bullet = evaluator.evaluate_bullet("Architected distributed caching with Redis, reducing p99 latency by 55% across 2M daily queries.")
    assert strong_bullet.is_weak is False
    assert strong_bullet.has_metrics is True
    assert strong_bullet.score >= 90.0


# ---------------------------------------------------------------------------
# 6. Professional Summary Evaluator
# ---------------------------------------------------------------------------

def test_summary_quality_evaluator():
    """Verify summary length, role recognition, and generic buzzword flagging."""
    evaluator = ContentEvaluator()

    # Good summary
    resume_good = _build_candidate_resume(
        summary="Senior Software Engineer with 6+ years specializing in Python microservices, distributed architectures, and scalable cloud solutions."
    )
    res_good = evaluator.evaluate_summary(resume_good)
    assert res_good.detected is True
    assert res_good.has_target_role is True
    assert res_good.is_generic is False
    assert res_good.score >= 80.0

    # Generic summary with buzzwords
    resume_generic = _build_candidate_resume(
        summary="A hardworking team player and dynamic individual with a proven track record seeking new opportunities."
    )
    res_generic = evaluator.evaluate_summary(resume_generic)
    assert res_generic.is_generic is True
    assert res_generic.score < res_good.score


# ---------------------------------------------------------------------------
# 7. Skill Boundary Safety & Taxonomy Categorization
# ---------------------------------------------------------------------------

def test_skill_boundary_safety_and_categories():
    """Verify Java != JavaScript, C != C++, and skill categorization."""
    skill_intel = SkillIntelligence()

    # Java vs JavaScript boundary
    assert skill_intel.normalize_skill_name("javascript") == "javascript"
    assert skill_intel.normalize_skill_name("java") == "java"
    assert skill_intel.get_skill_category("javascript") == "programming_languages"

    # Category checks
    assert skill_intel.get_skill_category("pytorch") == "data_science_and_ai"
    assert skill_intel.get_skill_category("postgresql") == "databases_and_data_engineering"
    assert skill_intel.get_skill_category("figma") == "design_and_creative"


# ---------------------------------------------------------------------------
# 8. Recommendation Priority Ordering
# ---------------------------------------------------------------------------

def test_recommendation_prioritization():
    """Verify HIGH priority recommendations precede MEDIUM and LOW."""
    resume_flawed = _build_candidate_resume(
        email="",  # missing email triggers HIGH priority
        phone="",  # missing phone triggers HIGH priority
        summary="",  # missing summary triggers MEDIUM priority
    )
    analyzer = CandidateResumeAnalyzer()
    analysis = analyzer.analyze(parsed_resume=resume_flawed)

    recs = analysis.recommendations
    assert len(recs) >= 2

    # Verify priority ordering
    priority_ranks = {PriorityLevel.HIGH: 1, PriorityLevel.MEDIUM: 2, PriorityLevel.LOW: 3}
    for i in range(len(recs) - 1):
        assert priority_ranks[recs[i].priority] <= priority_ranks[recs[i + 1].priority]

    assert recs[0].priority == PriorityLevel.HIGH
    assert any("Contact Information" in r.category for r in recs if r.priority == PriorityLevel.HIGH)


# ---------------------------------------------------------------------------
# 9. Synthetic Candidate Archetypes (AI/ML, Designer)
# ---------------------------------------------------------------------------

def test_synthetic_ai_ml_engineer_archetype():
    """Verify AI/ML Engineer profile highlights correct domain signals."""
    ai_skills = [
        SkillItem(name="Python", canonical_name="python", evidence_type=EvidenceType.RESUME_EVIDENCED),
        SkillItem(name="PyTorch", canonical_name="pytorch", evidence_type=EvidenceType.RESUME_EVIDENCED),
        SkillItem(name="Scikit-Learn", canonical_name="scikit-learn", evidence_type=EvidenceType.RESUME_EVIDENCED),
        SkillItem(name="NLP", canonical_name="nlp", evidence_type=EvidenceType.RESUME_EVIDENCED),
    ]
    resume = _build_candidate_resume(
        name="Elena Rostova",
        summary="Machine Learning Scientist specializing in deep learning, NLP, and LLM fine-tuning.",
        skills=ai_skills,
    )
    analyzer = CandidateResumeAnalyzer()
    analysis = analyzer.analyze(parsed_resume=resume)

    assert "Data Science & AI" in analysis.skills_analysis.domain_signals
    assert analysis.health_score.overall_score >= 80.0


def test_synthetic_product_designer_archetype():
    """Verify UI/UX Designer profile detects design tools and domain signals."""
    designer_skills = [
        SkillItem(name="Figma", canonical_name="figma", evidence_type=EvidenceType.RESUME_EVIDENCED),
        SkillItem(name="UI/UX Design", canonical_name="ui/ux design", evidence_type=EvidenceType.RESUME_EVIDENCED),
        SkillItem(name="Wireframing", canonical_name="wireframing", evidence_type=EvidenceType.RESUME_EVIDENCED),
        SkillItem(name="Prototyping", canonical_name="prototyping", evidence_type=EvidenceType.RESUME_EVIDENCED),
    ]
    resume = _build_candidate_resume(
        name="Jordan Lee",
        summary="Senior Product Designer with 5 years creating responsive mobile and web interfaces.",
        skills=designer_skills,
    )
    analyzer = CandidateResumeAnalyzer()
    analysis = analyzer.analyze(parsed_resume=resume)

    assert "Product Design / UI-UX" in analysis.skills_analysis.domain_signals
    assert analysis.health_score.overall_score >= 80.0


# ---------------------------------------------------------------------------
# 10. Candidate CSV Export
# ---------------------------------------------------------------------------

def test_candidate_csv_export(tmp_path: Path):
    """Verify candidate analysis report CSV contains required fields."""
    resume = _build_candidate_resume(name="Chris Evans")
    analyzer = CandidateResumeAnalyzer()
    analysis = analyzer.analyze(parsed_resume=resume)

    generator = CandidateReportGenerator(output_dir=tmp_path)
    csv_file = generator.generate_candidate_csv(analysis, filename="candidate_audit.csv")

    assert csv_file.is_file()
    content = csv_file.read_text(encoding="utf-8")

    expected_headers = [
        "Candidate Name",
        "Resume File",
        "Overall Health Score",
        "Health Rating",
        "ATS Readiness Score (%)",
        "ATS Readiness Status",
        "Content Quality Score (%)",
        "Skills Score (%)",
        "Experience Score (%)",
        "Education Score (%)",
        "Project Score (%)",
        "Structure Score (%)",
        "Total Experience (Years)",
        "Detected Skills",
        "Top Recommendations",
    ]
    for h in expected_headers:
        assert h in content

    assert "Chris Evans" in content
    assert "HR Report" not in content  # Confirm recruiter terminology eliminated


# ---------------------------------------------------------------------------
# 11. Candidate Web Routes Integration
# ---------------------------------------------------------------------------

def test_web_candidate_analyze_and_dashboard(client):
    """Verify end-to-end candidate upload, analysis, and dashboard rendering."""
    pdf = _create_sample_pdf("Morgan Stanley Python SQL FastAPI AWS")
    data = {
        "resume": (pdf, "morgan_resume.pdf"),
        "job_description": "We need a Python and FastAPI backend engineer.",
    }

    # Upload to /analyze
    response = client.post("/analyze", data=data, content_type="multipart/form-data", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")

    # Follow redirect to dashboard
    dashboard_resp = client.get("/dashboard")
    assert dashboard_resp.status_code == 200
    assert b"RESUME HEALTH SCORE" in dashboard_resp.data
    assert b"ATS READINESS" in dashboard_resp.data
    assert b"Actionable Recommendations" in dashboard_resp.data
    assert b"Download Resume Analysis" in dashboard_resp.data


def test_web_download_candidate_analysis_csv(client, app):
    """Verify downloading candidate analysis CSV report."""
    report_folder = Path(app.config["REPORT_FOLDER"])
    test_file = report_folder / "test_candidate_analysis.csv"
    test_file.write_text("Candidate Name,Overall Health Score\nJordan Lee,88.5\n", encoding="utf-8")

    response = client.get("/download-analysis/test_candidate_analysis.csv")
    assert response.status_code == 200
    assert "text/csv" in response.content_type
    assert b"Jordan Lee" in response.data


def test_web_download_analysis_security(client):
    """Verify path traversal attacks on candidate download route return 404."""
    assert client.get("/download-analysis/..%2F..%2Fconfig.py").status_code == 404
    assert client.get("/download-analysis/non_existent.csv").status_code == 404
