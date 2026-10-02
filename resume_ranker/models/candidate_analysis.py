"""Candidate-facing resume analysis, health score, and ATS readiness models."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class PriorityLevel(str, Enum):
    """Priority ranking for actionable candidate recommendations."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ATSCheckStatus(str, Enum):
    """Validation status for an ATS readiness criterion."""

    PASSED = "PASSED"
    WARNING = "WARNING"
    FAILED = "FAILED"


class ATSCheckItem(BaseModel):
    """Individual ATS readiness audit check result."""

    model_config = ConfigDict(extra="ignore")

    code: str = Field(..., description="Unique check identifier")
    title: str = Field(..., description="Display title for the ATS check")
    status: ATSCheckStatus = Field(..., description="Passed, Warning, or Failed")
    message: str = Field(..., description="Observed assessment details")
    fix_tip: str = Field(..., description="Candidate guidance to fix or optimize this item")


class ATSAnalysis(BaseModel):
    """Consolidated ATS Readiness assessment."""

    model_config = ConfigDict(extra="ignore")

    score: float = Field(..., ge=0.0, le=100.0, description="ATS Readiness percentage score")
    status_label: str = Field(default="Good Readiness", description="Human-friendly ATS status label")
    passed_count: int = Field(default=0, ge=0)
    warning_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    checks: List[ATSCheckItem] = Field(default_factory=list)
    format_summary: str = Field(default="", description="Summary of layout and font/glyph observations")
    warnings: List[str] = Field(default_factory=list)


class HealthScoreComponent(BaseModel):
    """Individual component within the 7-factor Resume Health Score."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Component name")
    score: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Component score if active")
    default_weight: float = Field(..., ge=0.0, le=1.0, description="Nominal weight")
    effective_weight: float = Field(default=0.0, ge=0.0, le=1.0, description="Renormalized active weight")
    is_active: bool = Field(default=True, description="Whether this component was evaluated")
    status: str = Field(default="Active", description="Status (e.g. 'Strong', 'Needs Improvement', 'Not Detected')")
    explanation: str = Field(default="", description="Detailed rationale for this score")


class ResumeHealthScore(BaseModel):
    """Overall explainable Resume Health Score with dynamic weight renormalization."""

    model_config = ConfigDict(extra="ignore")

    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall health score (0-100)")
    label: str = Field(default="Good", description="Score rating (e.g. 'Excellent', 'Good', 'Needs Work')")
    components: Dict[str, HealthScoreComponent] = Field(default_factory=dict)
    explanation: str = Field(default="", description="Overall summary rationale")


class BulletQuality(BaseModel):
    """Evaluation of an individual work experience or project bullet point."""

    model_config = ConfigDict(extra="ignore")

    raw_text: str = Field(..., description="Original bullet text")
    score: float = Field(default=75.0, ge=0.0, le=100.0)
    is_weak: bool = Field(default=False, description="Whether bullet begins with a weak/passive verb")
    weak_pattern: Optional[str] = Field(default=None, description="Matched weak phrase like 'responsible for'")
    has_metrics: bool = Field(default=False, description="Whether quantifiable outcome/metrics were detected")
    detected_metrics: List[str] = Field(default_factory=list, description="Detected numeric or percentage impact")
    suggestion: Optional[str] = Field(default=None, description="Actionable advice to strengthen bullet")
    example: Optional[str] = Field(default=None, description="Sample rewrite illustrating impact")


class ExperienceQualityAnalysis(BaseModel):
    """In-depth quality analysis of work experience and bullet points."""

    model_config = ConfigDict(extra="ignore")

    score: float = Field(..., ge=0.0, le=100.0)
    total_roles: int = Field(default=0, ge=0)
    total_duration_years: float = Field(default=0.0, ge=0.0)
    duration_display: str = Field(default="0 years")
    bullets: List[BulletQuality] = Field(default_factory=list)
    total_bullets: int = Field(default=0, ge=0)
    weak_bullet_count: int = Field(default=0, ge=0)
    measurable_achievement_count: int = Field(default=0, ge=0)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class SummaryQualityAnalysis(BaseModel):
    """Evaluation of the professional summary section."""

    model_config = ConfigDict(extra="ignore")

    detected: bool = Field(default=False)
    score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    word_count: int = Field(default=0, ge=0)
    has_target_role: bool = Field(default=False)
    has_tech_skills: bool = Field(default=False)
    is_generic: bool = Field(default=False)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    suggestion: Optional[str] = Field(default=None)


class ProjectItemAnalysis(BaseModel):
    """Evaluation of an individual project entry."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: Optional[str] = None
    technologies: List[str] = Field(default_factory=list)
    has_metrics: bool = False
    links: List[str] = Field(default_factory=list)
    feedback: str = ""


class ProjectQualityAnalysis(BaseModel):
    """Evaluation of projects section and technical demonstrations."""

    model_config = ConfigDict(extra="ignore")

    detected: bool = Field(default=False)
    score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    project_count: int = Field(default=0, ge=0)
    projects: List[ProjectItemAnalysis] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class SectionQualityDetail(BaseModel):
    """Quality and completeness evaluation of an individual resume section."""

    model_config = ConfigDict(extra="ignore")

    section_key: str = Field(..., description="Canonical section key (e.g. 'EXPERIENCE')")
    display_name: str = Field(..., description="Human-friendly section title (e.g. 'Experience')")
    detected: bool = Field(default=False)
    score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    status: str = Field(default="Not Detected", description="'Strong', 'Needs Improvement', 'Not Detected'")
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    recommendation: Optional[str] = Field(default=None)


class SkillCategoryBreakdown(BaseModel):
    """Skills grouped by technical or professional domain."""

    model_config = ConfigDict(extra="ignore")

    category_key: str
    category_name: str
    skills: List[str] = Field(default_factory=list)


class SkillRelevanceAnalysis(BaseModel):
    """Candidate skills breakdown, evidence verification, and domain signals."""

    model_config = ConfigDict(extra="ignore")

    score: float = Field(..., ge=0.0, le=100.0)
    total_detected: int = Field(default=0, ge=0)
    categories: List[SkillCategoryBreakdown] = Field(default_factory=list)
    evidenced_skills: List[str] = Field(default_factory=list)
    claimed_skills: List[str] = Field(default_factory=list)
    evidenced_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    domain_signals: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class KeywordMatchDetail(BaseModel):
    """Match status for a specific job requirement or role keyword."""

    model_config = ConfigDict(extra="ignore")

    keyword: str
    category: str = "General"
    detected: bool = False
    importance: str = "Required"  # 'Required', 'Preferred', 'Useful'
    evidence_section: Optional[str] = None
    context_tip: str = ""


class JobMatchAnalysis(BaseModel):
    """Optional alignment against a target Job Description."""

    model_config = ConfigDict(extra="ignore")

    is_available: bool = Field(default=False, description="True only if a job description was provided")
    match_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    target_role: Optional[str] = None
    required_skills_matched: List[str] = Field(default_factory=list)
    required_skills_missing: List[str] = Field(default_factory=list)
    preferred_skills_matched: List[str] = Field(default_factory=list)
    preferred_skills_missing: List[str] = Field(default_factory=list)
    experience_meets: Optional[bool] = None
    experience_note: Optional[str] = None
    education_meets: Optional[bool] = None
    education_note: Optional[str] = None
    keyword_coverage_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    keywords: List[KeywordMatchDetail] = Field(default_factory=list)
    tailoring_recommendations: List[str] = Field(default_factory=list)


class CandidateRecommendation(BaseModel):
    """Concrete, prioritized recommendation for resume improvement."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Unique recommendation ID")
    priority: PriorityLevel = Field(..., description="HIGH, MEDIUM, or LOW")
    category: str = Field(..., description="Section or area (e.g. 'Experience', 'ATS', 'Skills')")
    issue: str = Field(..., description="The observed issue or gap")
    why_it_matters: str = Field(..., description="Explanation of why this affects recruiters or ATS")
    suggested_action: str = Field(..., description="Direct guidance on what to change")
    example: Optional[str] = Field(default=None, description="Clear before/after example labeled as an example")


class EducationEntry(BaseModel):
    """Structured education item for candidate display."""

    model_config = ConfigDict(extra="ignore")

    institution: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    dates: Optional[str] = None


class CandidateResumeAnalysis(BaseModel):
    """Complete candidate-facing resume analysis aggregate object."""

    model_config = ConfigDict(extra="ignore")

    document_id: str
    filename: str
    candidate_name: str = "Candidate"
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    links: List[str] = Field(default_factory=list)
    raw_text: str = ""

    # High-level Scores
    health_score: ResumeHealthScore
    ats_readiness: ATSAnalysis

    # Component & Section Breakdowns
    sections: Dict[str, SectionQualityDetail] = Field(default_factory=dict)
    experience_analysis: ExperienceQualityAnalysis
    summary_analysis: SummaryQualityAnalysis
    project_analysis: ProjectQualityAnalysis
    skills_analysis: SkillRelevanceAnalysis

    # Optional Job Description Matching
    job_match: JobMatchAnalysis = Field(default_factory=JobMatchAnalysis)

    # Actionable Improvements & Warnings
    recommendations: List[CandidateRecommendation] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # Profile Details
    education_items: List[EducationEntry] = Field(default_factory=list)
    total_experience_years: float = 0.0
    total_experience_display: str = "0 years"
    created_at: str = ""
