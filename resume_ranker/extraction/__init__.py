"""Document and PDF extraction module."""

from resume_ranker.extraction.pdf_extractor import (
    ExtractedDocument,
    OCRExtractorInterface,
    extract_document,
    extract_resumes,
    extract_text_from_pdf,
)
from resume_ranker.extraction.quality_assessor import (
    DocumentQualityAssessor,
    DocumentQualityReport,
)
from resume_ranker.extraction.section_detector import SectionDetector

__all__ = [
    "extract_text_from_pdf",
    "extract_resumes",
    "extract_document",
    "ExtractedDocument",
    "OCRExtractorInterface",
    "DocumentQualityAssessor",
    "DocumentQualityReport",
    "SectionDetector",
]
