"""Tests for HR report generator module."""

import pytest
from pathlib import Path
from resume_ranker.reporting.report_generator import HRReportGenerator


def test_report_generator_initialization():
    """Verify HRReportGenerator directory configuration."""
    generator = HRReportGenerator(output_dir="custom_reports")
    assert generator.output_dir == Path("custom_reports")


def test_generate_csv_report_placeholder():
    """Verify generate_csv_report placeholder raises NotImplementedError on empty list."""
    generator = HRReportGenerator()
    with pytest.raises(NotImplementedError):
        generator.generate_csv_report([])


def test_generate_csv_report_with_candidates(tmp_path: Path):
    """Verify CSV report generation with structured candidate data."""
    generator = HRReportGenerator(output_dir=tmp_path)
    candidates = [
        {
            "rank": 1,
            "candidate_id": "alice.pdf",
            "final_score": 92.5,
            "confidence": 0.95,
            "similarity_score": 85.0,
            "required_skill_score": 100.0,
            "preferred_skill_score": 80.0,
            "experience_score": 90.0,
            "education_score": 100.0,
            "matched_required_skills": ["Python", "SQL"],
            "missing_required_skills": [],
            "total_experience_years": 4.5,
            "anomalies": [],
            "explanation": "Excellent candidate profile.",
        }
    ]
    report_file = generator.generate_csv_report(candidates, filename="test_out.csv")
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "alice.pdf" in content
    assert "92.5" in content
    assert "Evaluation Confidence" in content


def test_generate_summary_with_candidates():
    """Verify summary metrics calculation."""
    generator = HRReportGenerator()
    candidates = [
        {"candidate_id": "c1.pdf", "final_score": 80.0},
        {"candidate_id": "c2.pdf", "final_score": 60.0},
    ]
    summary = generator.generate_summary(candidates)
    assert summary["total_candidates"] == 2
    assert summary["average_score"] == 70.0
    assert summary["top_candidate"] == "c1.pdf"
    assert summary["high_scorers_count"] == 1

