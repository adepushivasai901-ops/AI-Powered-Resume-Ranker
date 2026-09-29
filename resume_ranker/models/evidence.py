"""Evidence and provenance schemas for traceable candidate evaluation."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class EvidenceType(str, Enum):
    """Classification of how evidence was established for a claim or skill."""

    EXPLICIT = "EXPLICIT"      # Directly declared in dedicated section (e.g., Skills list)
    VERIFIED = "VERIFIED"      # Directly supported by work experience / project bullet point
    INFERRED = "INFERRED"      # Derived from context or related tech stack, not explicitly stated
    UNCERTAIN = "UNCERTAIN"    # Ambiguous token or weak contextual evidence
    MISSING = "MISSING"        # No supporting evidence found in document


class BoundingBox(BaseModel):
    """Spatial bounding box coordinates on a document page (when supported by parser)."""

    model_config = ConfigDict(frozen=True)

    x0: float = Field(..., description="Left x-coordinate")
    y0: float = Field(..., description="Top y-coordinate")
    x1: float = Field(..., description="Right x-coordinate")
    y1: float = Field(..., description="Bottom y-coordinate")


class DocumentBlock(BaseModel):
    """An atomic text block or segment extracted from a document with structural metadata."""

    model_config = ConfigDict(extra="ignore")

    block_id: str = Field(..., description="Unique identifier for the block within document")
    page: int = Field(default=1, ge=1, description="1-indexed page number")
    text: str = Field(..., description="Raw text content of the block")
    section: Optional[str] = Field(default=None, description="Classified section heading (e.g., 'EXPERIENCE')")
    reading_order: int = Field(default=0, ge=0, description="Sequential reading order index")
    bbox: Optional[BoundingBox] = Field(default=None, description="Page coordinates if available")
    source: str = Field(default="text", description="Extraction channel ('pypdf', 'layout', 'ocr')")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score")


class EvidenceReference(BaseModel):
    """Traceable reference connecting a requirement or skill claim back to document evidence."""

    model_config = ConfigDict(extra="ignore")

    block_id: Optional[str] = Field(default=None, description="Originating DocumentBlock ID if known")
    section: Optional[str] = Field(default=None, description="Resume section where evidence was observed")
    text_snippet: str = Field(..., description="Exact or truncated excerpt from resume text")
    page: Optional[int] = Field(default=None, ge=1, description="Page number where snippet occurs")
    evidence_type: EvidenceType = Field(default=EvidenceType.EXPLICIT, description="Evidence certainty level")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in this evidence extraction")
    note: Optional[str] = Field(default=None, description="Optional reasoning note or context")


class Provenance(BaseModel):
    """Metadata recording the tools and models used to produce an extraction."""

    model_config = ConfigDict(extra="ignore")

    parser: str = Field(..., description="Parser engine name (e.g. 'pypdf', 'spacy')")
    parser_version: Optional[str] = Field(default=None, description="Version string of the parser")
    model_name: Optional[str] = Field(default=None, description="NLP/ML model identifier if applicable")
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp of parsing execution")
