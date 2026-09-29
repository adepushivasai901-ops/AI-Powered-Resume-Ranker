"""Flask routes and UI view handlers for AI-Powered Resume Ranker."""

import csv
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Set
import uuid

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from resume_ranker.extraction import extract_document, extract_text_from_pdf
from resume_ranker.nlp import (
    JDParser,
    ResumeParser,
    SkillIntelligence,
    TemporalEngine,
    TextPreprocessor,
)
from resume_ranker.ranking.ranker import CandidateRanker
from resume_ranker.reporting.report_generator import HRReportGenerator
from resume_ranker.scoring.scorer import CandidateScorer
from resume_ranker.vectorization.tfidf_matcher import TFIDFMatcher

main_bp = Blueprint("main", __name__)

# Server-side in-memory results store for small application demo
_results_store: Dict[str, Any] = {}

# Catalog of technical skills for boundary-aware keyword detection (legacy compatibility)
TECH_SKILLS: Set[str] = {
    "python",
    "sql",
    "java",
    "c++",
    "c#",
    ".net",
    "javascript",
    "typescript",
    "react",
    "angular",
    "vue",
    "node.js",
    "docker",
    "kubernetes",
    "aws",
    "gcp",
    "azure",
    "git",
    "linux",
    "flask",
    "django",
    "fastapi",
    "html",
    "css",
    "machine learning",
    "deep learning",
    "nlp",
    "ai",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "ci/cd",
    "rest apis",
    "graphql",
    "postgresql",
    "mysql",
    "mongodb",
    "redis",
    "spark",
    "hadoop",
    "tableau",
    "power bi",
    "agile",
    "scrum",
    "microservices",
}


def _find_skills(text: str) -> List[str]:
    """Identify recognized technical skills from text using boundary-safe matching.

    Args:
        text: Raw text to scan for skills.

    Returns:
        List of matched normalized skill names.
    """
    if not text or not isinstance(text, str):
        return []

    engine = SkillIntelligence()
    matches = engine.extract_skills_from_text(text)
    return sorted(list({alias.title() if len(alias) > 3 else alias.upper() for _, alias, _ in matches}))


def _generate_csv_report(ranked_candidates: List[Dict[str, Any]], report_path: Path) -> None:
    """Write ranked candidate data to an audit-ready CSV report using HRReportGenerator.

    Args:
        ranked_candidates: Ordered list of candidate scoring dictionaries.
        report_path: Target output file path.
    """
    generator = HRReportGenerator(output_dir=report_path.parent)
    generator.generate_csv_report(ranked_candidates, filename=report_path.name)


@main_bp.route("/", methods=["GET"])
def index():
    """Render landing page with resume upload form and job description input."""
    return render_template("index.html")


@main_bp.route("/upload", methods=["POST"])
def upload_resumes():
    """Handle resume and job description upload and execute full screening pipeline."""
    job_description = request.form.get("job_description", "").strip()
    if not job_description:
        flash("Job description is required.", "danger")
        return redirect(url_for("main.index"))

    files = request.files.getlist("resumes")
    valid_files = [f for f in files if f and f.filename and f.filename.strip()]
    if not valid_files:
        flash("Please upload at least one PDF resume.", "danger")
        return redirect(url_for("main.index"))

    allowed_exts = current_app.config.get("ALLOWED_EXTENSIONS", {"pdf"})
    upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_candidates = []
    for file in valid_files:
        filename = secure_filename(file.filename)
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in allowed_exts:
            flash(f"Invalid file type '{file.filename}'. Only PDF resumes are accepted.", "danger")
            return redirect(url_for("main.index"))

        # Save with unique prefix to avoid naming collisions
        safe_stored_name = f"{uuid.uuid4().hex[:8]}_{filename}"
        dest_path = upload_dir / safe_stored_name
        file.save(str(dest_path))
        saved_candidates.append((filename, dest_path))

    try:
        # Step 1: Parse Job Description into structured requirements
        jd_parser = JDParser()
        parsed_jd = jd_parser.parse_job_description(job_description)

        # Step 2: Extract PDF documents with structural blocks & quality assessment
        resume_parser = ResumeParser()
        raw_resumes: Dict[str, str] = {}
        parsed_resumes = {}
        quality_reports = {}

        for display_name, file_path in saved_candidates:
            try:
                extracted_doc = extract_document(file_path)
            except Exception:
                extracted_doc = extract_document.__wrapped__(file_path) if hasattr(extract_document, "__wrapped__") else None

            if extracted_doc is None:
                continue

            parsed_res = resume_parser.parse_document(extracted_doc, filename=display_name)
            raw_resumes[display_name] = extracted_doc.raw_text
            parsed_resumes[display_name] = parsed_res
            quality_reports[display_name] = extracted_doc.quality

        # Step 3: NLP Preprocessing for TF-IDF Lexical Similarity
        model_name = current_app.config.get("SPACY_MODEL", "en_core_web_sm")
        preprocessor = TextPreprocessor(model_name=model_name)
        proc_job = preprocessor.preprocess(job_description)
        proc_resumes = {cid: preprocessor.preprocess(text) for cid, text in raw_resumes.items()}

        matcher = TFIDFMatcher()
        sim_scores = matcher.compute_batch_similarity(proc_resumes, proc_job)

        # Step 4: Requirement-level Matching, Temporal Reasoning, and Scoring
        scorer = CandidateScorer()
        scored_candidates = []

        for cid, parsed_res in parsed_resumes.items():
            scored_profile = scorer.score_candidate_profile(
                parsed_resume=parsed_res,
                parsed_jd=parsed_jd,
                similarity_score=sim_scores.get(cid, 0.0),
                quality_report=quality_reports.get(cid),
            )
            scored_dict = scored_profile.model_dump()
            scored_candidates.append(scored_dict)

        # Step 5: Ranking
        ranker = CandidateRanker()
        ranked = ranker.rank_candidates(scored_candidates)

        # Step 6: Generate CSV Report
        report_dir = Path(current_app.config["REPORT_FOLDER"])
        report_dir.mkdir(parents=True, exist_ok=True)
        report_filename = f"hr_candidate_report_{int(time.time())}_{uuid.uuid4().hex[:4]}.csv"
        report_path = report_dir / report_filename
        _generate_csv_report(ranked, report_path)

        # Step 7: Store Latest Results
        report_generator = HRReportGenerator(output_dir=report_dir)
        summary_stats = report_generator.generate_summary(ranked) if ranked else {}

        _results_store["latest"] = {
            "candidates": ranked,
            "report_filename": report_filename,
            "job_description": job_description,
            "total_candidates": len(ranked),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parsed_jd": parsed_jd.model_dump(),
            "summary_metrics": summary_stats,
        }

        flash(f"Successfully screened and ranked {len(ranked)} candidates!", "success")
        return redirect(url_for("main.results"))

    except Exception as e:
        flash(f"An error occurred while processing resumes: {str(e)}", "danger")
        return redirect(url_for("main.index"))


@main_bp.route("/results", methods=["GET"])
def results():
    """Display candidate rankings, scores, and matched attributes."""
    latest_data = _results_store.get("latest")
    if not latest_data or not latest_data.get("candidates"):
        return render_template("results.html", candidates=None)

    return render_template(
        "results.html",
        candidates=latest_data["candidates"],
        report_filename=latest_data.get("report_filename"),
        job_description=latest_data.get("job_description"),
        total_candidates=latest_data.get("total_candidates", len(latest_data["candidates"])),
        timestamp=latest_data.get("timestamp"),
        parsed_jd=latest_data.get("parsed_jd"),
        summary_metrics=latest_data.get("summary_metrics", {}),
    )


@main_bp.route("/download-report/<filename>", methods=["GET"])
def download_report(filename: str):
    """Safely serve the generated HR report preventing path traversal.

    Args:
        filename: Name of the report CSV file to download.

    Returns:
        CSV file attachment download response.
    """
    if not filename or ".." in filename or "/" in filename or "\\" in filename:
        abort(404)

    safe_name = secure_filename(filename)
    if not safe_name or not safe_name.lower().endswith(".csv"):
        abort(404)

    report_dir = Path(current_app.config["REPORT_FOLDER"]).resolve()
    target_file = (report_dir / safe_name).resolve()

    if not target_file.is_file() or not target_file.is_relative_to(report_dir):
        abort(404)

    return send_from_directory(str(report_dir), safe_name, as_attachment=True, mimetype="text/csv")
