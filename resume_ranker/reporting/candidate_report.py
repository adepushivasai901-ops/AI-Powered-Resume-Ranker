"""Candidate resume analysis report generator producing structured, exportable CSV audits."""

import csv
from pathlib import Path
from typing import Union

from resume_ranker.models.candidate_analysis import CandidateResumeAnalysis


class CandidateReportGenerator:
    """Generates candidate-facing CSV analysis reports for career optimization and resume improvement."""

    def __init__(self, output_dir: Union[str, Path] = "reports"):
        """Initialize generator with output directory.

        Args:
            output_dir: Destination folder path for generated analysis reports.
        """
        self.output_dir = Path(output_dir)

    def generate_candidate_csv(
        self,
        analysis: CandidateResumeAnalysis,
        filename: str = "resume_analysis_report.csv",
    ) -> Path:
        """Export comprehensive candidate resume analysis into a structured CSV file.

        Args:
            analysis: Complete CandidateResumeAnalysis object.
            filename: Destination file name.

        Returns:
            Path to the written CSV file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / filename

        fieldnames = [
            "Candidate Name",
            "Resume File",
            "Overall Health Score",
            "Health Rating",
            "ATS Readiness Score (%)",
            "ATS Readiness Status",
            "Content Quality Score (%)",
            "Skills Score (%)",
            "Experience Score (%)",
            "Education Score (%)",
            "Project Score (%)",
            "Structure Score (%)",
            "Total Experience (Years)",
            "Detected Skills",
            "Resume-Evidenced Skills",
            "Missing Skills for Target Role",
            "Job Match Score (%)",
            "Target Role",
            "Warnings",
            "Top Recommendations",
        ]

        # Extract sub-scores safely from health score components
        comps = analysis.health_score.components
        ats_score = comps.get("ats").score if "ats" in comps and comps.get("ats") else analysis.ats_readiness.score
        content_score = comps.get("content").score if "content" in comps and comps.get("content") else None
        skills_score = comps.get("skills").score if "skills" in comps and comps.get("skills") else analysis.skills_analysis.score
        exp_score = comps.get("experience").score if "experience" in comps and comps.get("experience") else None
        edu_score = comps.get("education").score if "education" in comps and comps.get("education") else None
        proj_score = comps.get("projects").score if "projects" in comps and comps.get("projects") else None
        struct_score = comps.get("structure").score if "structure" in comps and comps.get("structure") else None

        # Format skills
        all_skills = [
            skill
            for cat in analysis.skills_analysis.categories
            for skill in cat.skills
        ] or analysis.skills_analysis.evidenced_skills + analysis.skills_analysis.claimed_skills

        # Format missing skills & job match
        missing_skills_str = "N/A (No job description provided)"
        job_match_score_str = "N/A"
        target_role_str = "General Evaluation"
        if analysis.job_match and analysis.job_match.is_available:
            missing_skills_str = ", ".join(analysis.job_match.required_skills_missing) if analysis.job_match.required_skills_missing else "None (All required skills satisfied)"
            job_match_score_str = f"{analysis.job_match.match_score:.1f}%" if analysis.job_match.match_score is not None else "N/A"
            target_role_str = analysis.job_match.target_role or "Target Role"

        # Format top recommendations
        rec_texts = [
            f"[{r.priority.value}] {r.issue} -> {r.suggested_action}"
            for r in analysis.recommendations[:5]
        ]

        row = {
            "Candidate Name": analysis.candidate_name,
            "Resume File": analysis.filename,
            "Overall Health Score": f"{analysis.health_score.overall_score:.1f}",
            "Health Rating": analysis.health_score.label,
            "ATS Readiness Score (%)": f"{ats_score:.1f}" if ats_score is not None else "N/A",
            "ATS Readiness Status": analysis.ats_readiness.status_label,
            "Content Quality Score (%)": f"{content_score:.1f}" if content_score is not None else "N/A",
            "Skills Score (%)": f"{skills_score:.1f}" if skills_score is not None else "N/A",
            "Experience Score (%)": f"{exp_score:.1f}" if exp_score is not None else "Not Detected",
            "Education Score (%)": f"{edu_score:.1f}" if edu_score is not None else "Not Detected",
            "Project Score (%)": f"{proj_score:.1f}" if proj_score is not None else "Not Detected",
            "Structure Score (%)": f"{struct_score:.1f}" if struct_score is not None else "N/A",
            "Total Experience (Years)": f"{analysis.total_experience_years:.1f}",
            "Detected Skills": ", ".join(all_skills[:30]),
            "Resume-Evidenced Skills": ", ".join(analysis.skills_analysis.evidenced_skills[:20]),
            "Missing Skills for Target Role": missing_skills_str,
            "Job Match Score (%)": job_match_score_str,
            "Target Role": target_role_str,
            "Warnings": " | ".join(analysis.warnings) if analysis.warnings else "None",
            "Top Recommendations": " || ".join(rec_texts) if rec_texts else "None",
        }

        with open(report_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

        return report_path
