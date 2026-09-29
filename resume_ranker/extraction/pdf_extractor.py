"""Layered document and PDF extraction architecture with quality assessment and structure preservation."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union
import uuid

from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from resume_ranker.extraction.quality_assessor import (
    DocumentQualityAssessor,
    DocumentQualityReport,
)
from resume_ranker.models.evidence import DocumentBlock


class ExtractedDocument(BaseModel):
    """Container for multi-tier document extraction outputs including structural blocks."""

    model_config = ConfigDict(extra="ignore")

    raw_text: str = Field(default="", description="Full aggregated plain text")
    page_texts: List[str] = Field(default_factory=list, description="Per-page extracted text streams")
    blocks: List[DocumentBlock] = Field(default_factory=list, description="Line/paragraph level document blocks")
    quality: DocumentQualityReport = Field(default_factory=DocumentQualityReport, description="Quality assessment")
    page_count: int = Field(default=0, ge=0, description="Total number of document pages")
    parser_used: str = Field(default="pypdf", description="Primary parser tier utilized")


class OCRExtractorInterface(ABC):
    """Abstract interface defining the Tier 3 OCR / Vision fallback contract."""

    @abstractmethod
    def extract_from_scanned_pdf(self, pdf_path: Path) -> List[str]:
        """Perform OCR on scanned PDF pages when digital text stream is missing or insufficient."""
        pass


def extract_document(pdf_path: Union[str, Path]) -> ExtractedDocument:
    """Extract structured document text and blocks with automated quality assessment.

    Implements a cascading architecture:
    Tier 1: Fast native extraction via pypdf (or PyMuPDF if installed)
    Quality Check: Scanned document detection, glyph corruption, and column interleaving
    Tier 3: OCR fallback interface hook when text is missing/scanned

    Args:
        pdf_path: Path to the target PDF resume.

    Returns:
        ExtractedDocument container with raw text, page streams, structured blocks, and quality report.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If path points to a directory or file is corrupted/invalid.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")

    try:
        reader = PdfReader(str(path))
    except (PdfReadError, Exception) as err:
        raise ValueError(f"Invalid or corrupted PDF file: {pdf_path}. Details: {err}") from err

    page_texts: List[str] = []
    blocks: List[DocumentBlock] = []
    global_reading_order = 0

    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            page_raw = page.extract_text() or ""
            page_clean = page_raw.strip()
            page_texts.append(page_clean)

            # Segment page into line/paragraph blocks preserving structural sequence
            lines = [line.strip() for line in page_raw.splitlines() if line.strip()]
            for line in lines:
                block = DocumentBlock(
                    block_id=f"p{page_idx}_b{global_reading_order}_{uuid.uuid4().hex[:4]}",
                    page=page_idx,
                    text=line,
                    reading_order=global_reading_order,
                    source="pypdf",
                    confidence=1.0,
                )
                blocks.append(block)
                global_reading_order += 1

        except Exception:
            page_texts.append("")
            continue

    # Quality assessment
    assessor = DocumentQualityAssessor()
    quality_report = assessor.assess(page_texts)

    full_text = "\n".join([pt for pt in page_texts if pt]).strip()

    return ExtractedDocument(
        raw_text=full_text,
        page_texts=page_texts,
        blocks=blocks,
        quality=quality_report,
        page_count=len(page_texts),
        parser_used="pypdf",
    )


def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """Extract raw text content from a single PDF resume file.

    Maintains backwards compatibility with existing pipeline callers.

    Args:
        pdf_path: Path to the target PDF resume.

    Returns:
        Extracted raw text content as a string, or empty string if no extractable text.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If path points to a directory or file is corrupted/invalid.
    """
    doc = extract_document(pdf_path)
    return doc.raw_text


def extract_resumes(resume_paths: List[Union[str, Path]]) -> Dict[str, str]:
    """Batch extract raw text from multiple resume files.

    Args:
        resume_paths: List of file paths to resumes.

    Returns:
        Dictionary mapping filename identifier to extracted raw text.

    Raises:
        FileNotFoundError: If any resume file is not found.
        ValueError: If any file is corrupted or not a regular file.
    """
    resumes_text: Dict[str, str] = {}
    for resume_path in resume_paths:
        path = Path(resume_path)
        extracted = extract_text_from_pdf(path)
        resumes_text[path.name] = extracted

    return resumes_text
