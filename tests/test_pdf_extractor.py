"""Unit tests for PDF extraction module."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from pypdf import PdfWriter

from resume_ranker.extraction.pdf_extractor import extract_text_from_pdf, extract_resumes


@pytest.fixture
def blank_pdf_path(tmp_path: Path) -> Path:
    """Fixture that generates a blank, valid PDF file."""
    pdf_file = tmp_path / "blank_sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with open(pdf_file, "wb") as f:
        writer.write(f)
    return pdf_file


def test_extract_text_from_valid_pdf():
    """Verify that extract_text_from_pdf extracts and combines text across pages."""
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Jane Doe\nSoftware Engineer"

    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Skills: Python, SQL, Machine Learning"

    mock_reader = MagicMock()
    mock_reader.pages = [mock_page1, mock_page2]

    with patch("resume_ranker.extraction.pdf_extractor.PdfReader", return_value=mock_reader):
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "is_file", return_value=True):
                result = extract_text_from_pdf("dummy_resume.pdf")

    assert "Jane Doe" in result
    assert "Software Engineer" in result
    assert "Skills: Python, SQL, Machine Learning" in result
    assert isinstance(result, str)


def test_extract_text_from_empty_pdf(blank_pdf_path: Path):
    """Verify that a blank PDF yields an empty string without raising an exception."""
    result = extract_text_from_pdf(blank_pdf_path)
    assert result == ""


def test_extract_text_missing_file(tmp_path: Path):
    """Verify that attempting to extract text from a non-existent file raises FileNotFoundError."""
    missing_file = tmp_path / "non_existent_file.pdf"
    with pytest.raises(FileNotFoundError, match="not found"):
        extract_text_from_pdf(missing_file)


def test_extract_text_directory_path(tmp_path: Path):
    """Verify that passing a directory path raises ValueError."""
    with pytest.raises(ValueError, match="not a file"):
        extract_text_from_pdf(tmp_path)


def test_extract_text_corrupted_pdf(tmp_path: Path):
    """Verify that corrupted PDF files raise ValueError without being silently swallowed."""
    corrupted_file = tmp_path / "corrupted.pdf"
    corrupted_file.write_bytes(b"This is not a valid PDF header")

    with pytest.raises(ValueError, match="Invalid or corrupted PDF"):
        extract_text_from_pdf(corrupted_file)


def test_extract_resumes_batch():
    """Verify batch extraction for multiple resume paths."""
    mock_page_a = MagicMock()
    mock_page_a.extract_text.return_value = "Alice Resume"

    mock_page_b = MagicMock()
    mock_page_b.extract_text.return_value = "Bob Resume"

    def side_effect(path, *args, **kwargs):
        reader = MagicMock()
        if "alice" in str(path).lower():
            reader.pages = [mock_page_a]
        else:
            reader.pages = [mock_page_b]
        return reader

    with patch("resume_ranker.extraction.pdf_extractor.PdfReader", side_effect=side_effect):
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "is_file", return_value=True):
                results = extract_resumes(["alice_resume.pdf", "bob_resume.pdf"])

    assert "alice_resume.pdf" in results
    assert "bob_resume.pdf" in results
    assert results["alice_resume.pdf"] == "Alice Resume"
    assert results["bob_resume.pdf"] == "Bob Resume"


def test_extract_resumes_missing_file_raises():
    """Verify that batch extraction does not silently swallow missing files."""
    with pytest.raises(FileNotFoundError):
        extract_resumes(["missing_resume.pdf"])
