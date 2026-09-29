"""Matching, anomaly detection, and explainable scoring evaluation models."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from resume_ranker.models.evidence import EvidenceReference
from resume_ranker.models.job_description import RequirementType


class RequirementMatchStatus(str, Enum):
    """Evaluation status for a single requirement tested against candidate evidence."""

    SATISFIED = "SATISFIED"                          # Solid explicit/verified evidence matches requirement
    PARTIALLY_SATISFIED = "PARTIALLY_SATISFIED"      # Inferred match, partial skill overlap, or insufficient duration
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # Mentioned vaguely with uncertain context
    NOT_FOUND = "NOT_FOUND"                          # No evidence of requirement in candidate document
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"    # Contradictory statements or dates detected


class RequirementMatchResult(BaseModel):
    """Detailed matching result for a single job requirement."""

    model_config = ConfigDict(extra="ignore")

    requirement_id: str = Field(..., description="ID of the matched JD requirement")
    canonical_name: str = Field(..., description="Target skill or competency name")
    req_type: RequirementType = Field(default=RequirementType.MUST_HAVE_SKILL, description="Requirement type")
    status: RequirementMatchStatus = Field(default=RequirementMatchStatus.NOT_FOUND, description="Match determination")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalized match score between 0.0 and 1.0")
    candidate_duration_months: Optional[int] = Field(default=None, description="Demonstrated candidate experience in months")
    required_duration_months: Optional[int] = Field(default=None, description="Required experience in months if specified")
    evidence_refs: List[EvidenceReference] = Field(default_factory=list, description="Linked evidence from resume")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in this match result")
    explanation: str = Field(default="", description="Human-readable justification for this match status")


class AnomalySeverity(str, Enum):
    """Severity classification of an anomaly or integrity issue."""

    LOW = "LOW"        # Informational note or minor formatting observation
    MEDIUM = "MEDIUM"  # Potential inconsistency worth human review (e.g., date overlap, unbacked skill)
    HIGH = "HIGH"      # Major anomaly (e.g. impossible chronological dates, prompt injection attempt)


class AnomalyFlag(BaseModel):
    """An anomaly or integrity warning detected in candidate document."""

    model_config = ConfigDict(extra="ignore")

    code: str = Field(..., description="Machine-readable anomaly code (e.g. 'OVERLAPPING_DATES')")
    severity: AnomalySeverity = Field(default=AnomalySeverity.MEDIUM, description="Severity rating")
    message: str = Field(..., description="Clear explanation of the detected anomaly for human review")
    evidence_refs: List[EvidenceReference] = Field(default_factory=list, description="Evidence blocks involved")
    details: Dict[str, Any] = Field(default_factory=dict, description="Structured diagnostics metadata")


class ScoredCandidateProfile(BaseModel):
    """Complete explainable evaluation and ranking profile for a candidate."""

    model_config = ConfigDict(extra="ignore")

    candidate_id: str = Field(..., description="Candidate or resume identifier")
    filename: Optional[str] = Field(default=None, description="Original filename")
    rank: Optional[int] = Field(default=None, ge=1, description="Assigned ranking order")
    final_score: float = Field(..., ge=0.0, le=100.0, description="Overall weighted composite score (0-100)")
    similarity_score: float = Field(default=0.0, ge=0.0, le=100.0, description="TF-IDF / semantic similarity score")
    required_skill_score: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Must-have skill coverage score")
    preferred_skill_score: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Preferred skill bonus score")
    experience_score: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Experience & tenure alignment score")
    education_score: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Educational alignment score")

    requirement_matches: List[RequirementMatchResult] = Field(
        default_factory=list, description="Granular requirement match results"
    )
    matched_required_skills: List[str] = Field(default_factory=list, description="Satisfied required skills")
    missing_required_skills: List[str] = Field(default_factory=list, description="Missing required skills")
    matched_preferred_skills: List[str] = Field(default_factory=list, description="Satisfied preferred skills")
    missing_preferred_skills: List[str] = Field(default_factory=list, description="Missing preferred skills")
    evidenced_skills: List[str] = Field(default_factory=list, description="Skills evidenced in experience or projects")
    claimed_skills: List[str] = Field(default_factory=list, description="Skills claimed in skills lists without experience/project evidence")

    anomalies: List[AnomalyFlag] = Field(default_factory=list, description="Detected resume anomalies")
    total_experience_years: float = Field(default=0.0, ge=0.0, description="Computed continuous experience years")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall system evaluation confidence")
    explanation: str = Field(default="", description="Detailed transparent justification of the score")
