"""Unit tests for document quality assessment and structured extraction."""

from pathlib import Path
from pypdf import PdfWriter
import pytest

from resume_ranker.extraction import (
    DocumentQualityAssessor,
    extract_document,
    extract_text_from_pdf,
)


def _create_test_pdf(tmp_path: Path, filename: str, pages_text: list) -> Path:
    """Helper to create a multi-page PDF on disk with given page texts."""
    pdf_path = tmp_path / filename
    writer = PdfWriter()
    for text in pages_text:
        # Note: In standard headless testing without canvas fonts, blank pages represent layout/page count
        writer.add_blank_page(width=200, height=200)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_path


def test_quality_assessor_empty():
    """Verify quality report on completely empty text."""
    assessor = DocumentQualityAssessor()
    report = assessor.assess([])
    assert report.is_empty is True
    assert report.confidence == 0.0
    assert len(report.warnings) > 0


def test_quality_assessor_clean_text():
    """Verify quality report on normal high-volume text."""
    assessor = DocumentQualityAssessor()
    pages = [
        "Jane Doe\nSenior AI Engineer\n5 years experience building scalable deep learning systems.",
        "Education: Bachelor of Science in Computer Science.\nSkills: Python, PyTorch, Docker, Kubernetes.",
    ]
    report = assessor.assess(pages)
    assert report.is_empty is False
    assert report.is_very_low_text is False
    assert report.is_scanned_image is False
    assert report.confidence >= 0.9
    assert len(report.warnings) == 0


def test_quality_assessor_low_text_and_scanned():
    """Verify low-text warning on multi-page PDF with minimal characters."""
    assessor = DocumentQualityAssessor()
    pages = ["Page 1", "P2", "End"]  # 12 total chars across 3 pages
    report = assessor.assess(pages)
    assert report.is_very_low_text is True
    assert report.confidence <= 0.35
    assert any("low text" in w.lower() or "scanned" in w.lower() for w in report.warnings)


def test_quality_assessor_suspicious_characters():
    """Verify corruption detection when replacement characters are present."""
    assessor = DocumentQualityAssessor()
    pages = ["Normal title \ufffd\ufffd\ufffd\ufffd\ufffd corrupted font stream \ufffd\ufffd\ufffd"]
    report = assessor.assess(pages)
    assert report.suspicious_chars_ratio > 0.05
    assert any("unprintable" in w.lower() or "corrupt" in w.lower() for w in report.warnings)


def test_extract_document_structure(tmp_path: Path):
    """Verify extract_document returns structured blocks and quality report."""
    pdf_path = _create_test_pdf(tmp_path, "sample.pdf", ["Page 1"])
    doc = extract_document(pdf_path)
    assert doc.page_count == 1
    assert doc.parser_used == "pypdf"
    assert doc.quality is not None
    # Verify backward compatibility
    text = extract_text_from_pdf(pdf_path)
    assert isinstance(text, str)
