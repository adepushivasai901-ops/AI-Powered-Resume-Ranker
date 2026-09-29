"""Unit tests for Flask routes, UI workflow, and security validations."""

import io
from pathlib import Path
import pytest
from pypdf import PdfWriter

from app import create_app
from config import TestingConfig


@pytest.fixture
def app(tmp_path: Path):
    """Create Flask application instance with temporary runtime folders."""
    app = create_app(config_class=TestingConfig)
    app.config["UPLOAD_FOLDER"] = tmp_path / "uploads"
    app.config["REPORT_FOLDER"] = tmp_path / "reports"
    app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)
    app.config["REPORT_FOLDER"].mkdir(parents=True, exist_ok=True)
    return app


@pytest.fixture
def client(app):
    """Create test client for Flask app."""
    return app.test_client()


def _create_sample_pdf(text: str = "Sample Candidate") -> io.BytesIO:
    """Helper to generate an in-memory PDF file."""
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    stream = io.BytesIO()
    writer.write(stream)
    stream.seek(0)
    return stream


def test_index_get(client):
    """Verify landing page loads with upload form and header."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"AI-Powered Resume Ranker" in response.data
    assert b"Job Description" in response.data
    assert b'name="job_description"' in response.data
    assert b'name="resumes"' in response.data


def test_upload_missing_job_description(client):
    """Verify upload fails gracefully when job description is omitted."""
    pdf_stream = _create_sample_pdf()
    data = {
        "job_description": "   ",
        "resumes": (pdf_stream, "resume.pdf"),
    }
    response = client.post("/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    assert b"Job description is required" in response.data


def test_upload_missing_resumes(client):
    """Verify upload fails gracefully when no resume files are selected."""
    data = {
        "job_description": "Need python developer",
        "resumes": [],
    }
    response = client.post("/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    assert b"Please upload at least one PDF resume" in response.data


def test_upload_invalid_file_extension(client):
    """Verify non-PDF uploads are rejected."""
    txt_stream = io.BytesIO(b"Candidate plain text resume")
    data = {
        "job_description": "Need python developer",
        "resumes": (txt_stream, "resume.txt"),
    }
    response = client.post("/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    assert b"Only PDF resumes are accepted" in response.data


def test_results_get_empty_state(client):
    """Verify GET /results renders graceful empty state when no screening has been run."""
    # Ensure no stored results
    from resume_ranker.web import routes
    routes._results_store.clear()

    response = client.get("/results")
    assert response.status_code == 200
    assert b"No candidate ranking results available yet" in response.data


def test_valid_upload_workflow_and_results(client):
    """Verify end-to-end upload, ranking, and results rendering."""
    pdf1 = _create_sample_pdf("Jane Doe Python SQL AWS")
    pdf2 = _create_sample_pdf("John Smith Pastry Chef")

    data = {
        "job_description": "Looking for Python developer with SQL and AWS experience.",
        "resumes": [
            (pdf1, "jane_doe.pdf"),
            (pdf2, "john_smith.pdf"),
        ],
    }

    # Upload resumes
    response = client.post("/upload", data=data, content_type="multipart/form-data", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/results")

    # Follow redirect to results page
    results_resp = client.get("/results")
    assert results_resp.status_code == 200
    assert b"Candidate Rankings" in results_resp.data
    assert b"jane_doe.pdf" in results_resp.data
    assert b"john_smith.pdf" in results_resp.data
    assert b"Download HR Report" in results_resp.data


def test_download_report_success(client, app):
    """Verify downloading a valid generated report."""
    report_folder = Path(app.config["REPORT_FOLDER"])
    report_file = report_folder / "test_report.csv"
    report_file.write_text("Rank,Candidate,Score\n1,alice.pdf,95.0\n", encoding="utf-8")

    response = client.get("/download-report/test_report.csv")
    assert response.status_code == 200
    assert "text/csv" in response.content_type
    assert b"alice.pdf" in response.data


def test_download_report_path_traversal_prevention(client):
    """Verify path traversal attacks return 404."""
    # Path traversal attempt
    response = client.get("/download-report/..%2F..%2Fconfig.py")
    assert response.status_code in [404, 400]

    response_slash = client.get("/download-report/../../config.py")
    assert response_slash.status_code in [404, 400]


def test_download_report_nonexistent_file(client):
    """Verify requesting non-existent report returns 404."""
    response = client.get("/download-report/non_existent_report.csv")
    assert response.status_code == 404
