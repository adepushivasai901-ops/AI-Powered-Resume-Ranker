"""Job description structured representation and requirement schemas."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class RequirementType(str, Enum):
    """Categorization of requirements specified in a job description."""

    MUST_HAVE_SKILL = "MUST_HAVE_SKILL"          # Mandatory technical skill or tool
    PREFERRED_SKILL = "PREFERRED_SKILL"          # Nice-to-have or bonus technical skill
    MINIMUM_EXPERIENCE = "MINIMUM_EXPERIENCE"    # Years of experience required overall or in a discipline
    EDUCATION = "EDUCATION"                      # Degree requirement (e.g. Bachelor's in CS)
    CERTIFICATION = "CERTIFICATION"              # Required professional credential
    RESPONSIBILITY = "RESPONSIBILITY"            # Key expected duty or deliverable
    DOMAIN = "DOMAIN"                            # Industry domain knowledge (e.g. FinTech, Healthcare)


class ImportanceLevel(str, Enum):
    """Relative priority weighting for job description requirements."""

    CRITICAL = "CRITICAL"    # Disqualifying if absent (e.g., core must-have language)
    HIGH = "HIGH"            # Major role competency
    MEDIUM = "MEDIUM"        # Preferred capability
    LOW = "LOW"              # Minor bonus


class JDRequirement(BaseModel):
    """An individual atomic requirement extracted from a job description."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Unique requirement ID (e.g. 'REQ-01')")
    req_type: RequirementType = Field(default=RequirementType.MUST_HAVE_SKILL, description="Requirement category")
    canonical_name: str = Field(..., description="Normalized competency or skill name (e.g. 'python')")
    importance: ImportanceLevel = Field(default=ImportanceLevel.HIGH, description="Priority classification")
    min_experience_months: Optional[int] = Field(default=None, ge=0, description="Minimum months of experience if stated")
    original_text: str = Field(..., description="Exact requirement clause or bullet from the JD")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")
    keywords: List[str] = Field(default_factory=list, description="Associated synonyms and search tokens")


class ParsedJobDescription(BaseModel):
    """Structured representation of a parsed job description."""

    model_config = ConfigDict(extra="ignore")

    raw_text: str = Field(..., description="Original job description text")
    title: Optional[str] = Field(default=None, description="Extracted or inferred target job title")
    seniority: Optional[str] = Field(default=None, description="Seniority level (e.g. 'Junior', 'Mid', 'Senior', 'Lead')")
    min_total_experience_years: Optional[float] = Field(
        default=None, ge=0.0, description="Minimum general career experience in years"
    )
    max_total_experience_years: Optional[float] = Field(
        default=None, ge=0.0, description="Maximum experience years if specified as range (e.g. 2-4 years)"
    )
    relevant_experience_required: bool = Field(
        default=False, description="Whether explicit relevant/hands-on experience was demanded"
    )
    required_degrees: List[str] = Field(default_factory=list, description="Target educational degrees")
    requirements: List[JDRequirement] = Field(
        default_factory=list, description="Atomic extracted requirements"
    )
    detected_skills: List[str] = Field(default_factory=list, description="All normalized skill names found in JD")
    responsibilities: List[str] = Field(default_factory=list, description="Extracted role responsibilities")
    warnings: List[str] = Field(default_factory=list, description="Recruiter warnings such as low skill count")

    @property
    def required_skills(self) -> List[str]:
        """List of canonical skill names marked as must-have requirements."""
        return [
            r.canonical_name
            for r in self.requirements
            if r.req_type == RequirementType.MUST_HAVE_SKILL
        ]

    @property
    def preferred_skills(self) -> List[str]:
        """List of canonical skill names marked as preferred requirements."""
        return [
            r.canonical_name
            for r in self.requirements
            if r.req_type == RequirementType.PREFERRED_SKILL
        ]

    @property
    def education_level(self) -> Optional[str]:
        """First target education degree or level if present."""
        return self.required_degrees[0] if self.required_degrees else None
