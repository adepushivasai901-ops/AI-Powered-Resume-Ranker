"""Domain schemas and canonical data models for AI-Powered Resume Ranker."""

from resume_ranker.models.evidence import (
    BoundingBox,
    DocumentBlock,
    EvidenceReference,
    EvidenceType,
    Provenance,
)
from resume_ranker.models.job_description import (
    ImportanceLevel,
    JDRequirement,
    ParsedJobDescription,
    RequirementType,
)
from resume_ranker.models.matching import (
    AnomalyFlag,
    AnomalySeverity,
    RequirementMatchResult,
    RequirementMatchStatus,
    ScoredCandidateProfile,
)
from resume_ranker.models.resume import (
    CandidateInfo,
    CertificationItem,
    DocumentMetadata,
    EducationItem,
    ExperienceItem,
    ParsedResume,
    ProjectItem,
    SkillItem,
    TimelineAnalysis,
    TimelineGap,
    TimelineOverlap,
)

__all__ = [
    "EvidenceType",
    "BoundingBox",
    "DocumentBlock",
    "EvidenceReference",
    "Provenance",
    "RequirementType",
    "ImportanceLevel",
    "JDRequirement",
    "ParsedJobDescription",
    "RequirementMatchStatus",
    "RequirementMatchResult",
    "AnomalySeverity",
    "AnomalyFlag",
    "ScoredCandidateProfile",
    "DocumentMetadata",
    "CandidateInfo",
    "ExperienceItem",
    "EducationItem",
    "ProjectItem",
    "CertificationItem",
    "SkillItem",
    "TimelineGap",
    "TimelineOverlap",
    "TimelineAnalysis",
    "ParsedResume",
]
