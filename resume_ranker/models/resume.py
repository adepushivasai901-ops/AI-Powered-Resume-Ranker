"""Canonical resume data representation schemas using Pydantic."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from resume_ranker.models.evidence import (
    DocumentBlock,
    EvidenceReference,
    EvidenceType,
    Provenance,
)


class DocumentMetadata(BaseModel):
    """Metadata regarding the ingested physical/digital document."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Unique document ID (UUID or hash)")
    filename: str = Field(..., description="Original filename of the uploaded file")
    source_type: str = Field(default="pdf", description="Document file extension/type ('pdf', 'docx')")
    page_count: int = Field(default=1, ge=0, description="Total number of pages")
    language: str = Field(default="en", description="Primary detected language code")
    parser_path: str = Field(default="pypdf", description="Extraction path or tier used")
    parse_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall parsing confidence")
    warnings: List[str] = Field(default_factory=list, description="Parsing warnings or quality flags")


class CandidateInfo(BaseModel):
    """Candidate identity and contact information."""

    model_config = ConfigDict(extra="ignore")

    name: Optional[str] = Field(default=None, description="Candidate full name")
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    location: Optional[str] = Field(default=None, description="City / state / country location")
    links: List[str] = Field(default_factory=list, description="Portfolio, LinkedIn, or GitHub links")


class ExperienceItem(BaseModel):
    """Individual employment history role with evidence provenance."""

    model_config = ConfigDict(extra="ignore")

    company: Optional[str] = Field(default=None, description="Employer or organization name")
    raw_title: str = Field(..., description="Job title as written in resume")
    normalized_title: Optional[str] = Field(default=None, description="Standardized role title")
    start_date: Optional[str] = Field(default=None, description="Normalized start date (YYYY or YYYY-MM)")
    end_date: Optional[str] = Field(default=None, description="Normalized end date (YYYY, YYYY-MM, or 'Present')")
    is_current: bool = Field(default=False, description="Whether this is marked as candidate's current role")
    duration_months: Optional[int] = Field(default=None, ge=0, description="Computed duration in months")
    employment_type: Optional[str] = Field(default=None, description="Full-time, contract, internship, etc.")
    responsibilities: List[str] = Field(default_factory=list, description="Bullet points of role responsibilities")
    achievements: List[str] = Field(default_factory=list, description="Quantified achievements / impacts")
    technologies_used: List[str] = Field(default_factory=list, description="Technologies referenced in this role")
    evidence_refs: List[EvidenceReference] = Field(default_factory=list, description="Evidence references in resume")


class EducationItem(BaseModel):
    """Academic qualification or educational history."""

    model_config = ConfigDict(extra="ignore")

    institution: Optional[str] = Field(default=None, description="University, college, or school name")
    degree: Optional[str] = Field(default=None, description="Normalized degree level (e.g. 'Bachelor of Science')")
    field_of_study: Optional[str] = Field(default=None, description="Major or specialization")
    start_date: Optional[str] = Field(default=None, description="Start date (YYYY or YYYY-MM)")
    end_date: Optional[str] = Field(default=None, description="Graduation / completion date")
    grade_cgpa: Optional[str] = Field(default=None, description="Reported GPA or honors if available")
    evidence_refs: List[EvidenceReference] = Field(default_factory=list, description="Evidence references")


class ProjectItem(BaseModel):
    """Project portfolio or technical project item."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Project title or name")
    description: Optional[str] = Field(default=None, description="Summary description of project scope")
    technologies: List[str] = Field(default_factory=list, description="Technologies and libraries utilized")
    links: List[str] = Field(default_factory=list, description="Repository or demo links")
    evidence_refs: List[EvidenceReference] = Field(default_factory=list, description="Evidence references")


class CertificationItem(BaseModel):
    """Professional certification or license."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Certification title (e.g. 'AWS Certified Solutions Architect')")
    issuer: Optional[str] = Field(default=None, description="Certifying authority or vendor")
    date: Optional[str] = Field(default=None, description="Issue date or validity period")
    credential_id: Optional[str] = Field(default=None, description="Credential ID or verification URL")
    evidence_refs: List[EvidenceReference] = Field(default_factory=list, description="Evidence references")


class SkillItem(BaseModel):
    """A canonicalized candidate skill linked to evidence and duration."""

    model_config = ConfigDict(extra="ignore")

    canonical_name: str = Field(..., description="Standardized canonical skill name (e.g. 'scikit-learn')")
    aliases_detected: List[str] = Field(default_factory=list, description="Raw aliases seen in text (e.g. ['sklearn'])")
    evidence_type: EvidenceType = Field(default=EvidenceType.EXPLICIT, description="Evidence strength classification")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")
    first_used: Optional[str] = Field(default=None, description="Earliest dated role referencing this skill")
    last_used: Optional[str] = Field(default=None, description="Most recent dated role referencing this skill")
    duration_months: Optional[int] = Field(default=None, ge=0, description="Calculated usage duration in months")
    evidence_refs: List[EvidenceReference] = Field(default_factory=list, description="References where skill appears")


class TimelineGap(BaseModel):
    """Employment timeline gap exceeding a defined threshold."""

    model_config = ConfigDict(extra="ignore")

    start_date: str = Field(..., description="Start of gap period")
    end_date: str = Field(..., description="End of gap period")
    duration_months: int = Field(..., ge=0, description="Duration of gap in months")
    description: Optional[str] = Field(default=None, description="Contextual explanation if detectable")


class TimelineOverlap(BaseModel):
    """Concurrent employment period across multiple organizations."""

    model_config = ConfigDict(extra="ignore")

    role_1: str = Field(..., description="First overlapping role description")
    role_2: str = Field(..., description="Second overlapping role description")
    overlap_months: int = Field(..., ge=0, description="Number of overlapping months")


class TimelineAnalysis(BaseModel):
    """Consolidated career temporal analysis avoiding naive interval summation."""

    model_config = ConfigDict(extra="ignore")

    total_experience_months: int = Field(default=0, ge=0, description="De-duplicated continuous months of work")
    total_experience_years: float = Field(default=0.0, ge=0.0, description="De-duplicated continuous years of work")
    earliest_date: Optional[str] = Field(default=None, description="Earliest recorded work date")
    latest_date: Optional[str] = Field(default=None, description="Latest recorded work date")
    gaps: List[TimelineGap] = Field(default_factory=list, description="Identified employment gaps")
    overlaps: List[TimelineOverlap] = Field(default_factory=list, description="Concurrent employment periods")
    anomalies: List[str] = Field(default_factory=list, description="Chronological anomalies or inconsistencies")


class ParsedResume(BaseModel):
    """Canonical structured representation of a fully ingested and parsed resume."""

    model_config = ConfigDict(extra="ignore")

    document: DocumentMetadata = Field(..., description="Ingested document metadata and quality flags")
    candidate: CandidateInfo = Field(default_factory=CandidateInfo, description="Candidate personal contact details")
    summary: Optional[str] = Field(default=None, description="Professional summary or objective text")
    skills: List[SkillItem] = Field(default_factory=list, description="Extracted canonical skills with evidence")
    experience: List[ExperienceItem] = Field(default_factory=list, description="Employment history items")
    education: List[EducationItem] = Field(default_factory=list, description="Education history items")
    projects: List[ProjectItem] = Field(default_factory=list, description="Project portfolio items")
    certifications: List[CertificationItem] = Field(default_factory=list, description="Certifications and licenses")
    timeline: TimelineAnalysis = Field(default_factory=TimelineAnalysis, description="Career duration and chronology")
    blocks: List[DocumentBlock] = Field(default_factory=list, description="Structured document text blocks")
    sections_detected: Dict[str, List[str]] = Field(
        default_factory=dict, description="Mapping of normalized section name to constituent text blocks/lines"
    )
    provenance: Optional[Provenance] = Field(default=None, description="Pipeline parser and model provenance")
    raw_text: str = Field(default="", description="Complete extracted raw text")
