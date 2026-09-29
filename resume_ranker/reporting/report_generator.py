"""HR report generation and export interface for ranking results with evidence and anomaly details."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Union


class HRReportGenerator:
    """Generates structured HR reports (CSV, audit summaries) for candidate screening review."""

    def __init__(self, output_dir: Union[str, Path] = "reports"):
        """Initialize report generator with destination directory.

        Args:
            output_dir: Destination folder path for generated reports.
        """
        self.output_dir = Path(output_dir)

    def generate_csv_report(
        self,
        ranked_candidates: List[Dict[str, Any]],
        filename: str = "hr_candidate_report.csv",
    ) -> Path:
        """Export ranked candidate list with scores, skills, anomalies, and explanations to CSV.

        Args:
            ranked_candidates: List of ranked candidate dictionaries or ScoredCandidateProfile dicts.
            filename: Output CSV filename.

        Returns:
            Path to the generated report file.

        Raises:
            NotImplementedError: If called with empty candidates list in skeleton test mode.
        """
        if not ranked_candidates:
            raise NotImplementedError("Candidate list is empty; provide scored candidates to generate report.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / filename

        fieldnames = [
            "Rank",
            "Candidate",
            "Final Score",
            "Evaluation Confidence (%)",
            "TF-IDF Similarity (%)",
            "Required Skills Score (%)",
            "Preferred Skills Score (%)",
            "Experience Score (%)",
            "Education Score (%)",
            "Matched Skills",
            "Missing Skills",
            "Total Experience (Years)",
            "Anomalies Detected",
            "Explanation",
        ]

        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in ranked_candidates:
                # Handle anomalies if list of dicts or list of strings
                raw_anomalies = c.get("anomalies", [])
                if raw_anomalies:
                    if isinstance(raw_anomalies[0], dict):
                        anomaly_str = "; ".join(f"[{a.get('severity', 'MED')}] {a.get('message', '')}" for a in raw_anomalies)
                    else:
                        anomaly_str = "; ".join(str(a) for a in raw_anomalies)
                else:
                    anomaly_str = "None"

                writer.writerow(
                    {
                        "Rank": c.get("rank", ""),
                        "Candidate": c.get("candidate_id", ""),
                        "Final Score": c.get("final_score", 0.0),
                        "Evaluation Confidence (%)": int(c.get("confidence", 1.0) * 100),
                        "TF-IDF Similarity (%)": c.get("similarity_score", 0.0),
                        "Required Skills Score (%)": c.get("required_skill_score", 0.0),
                        "Preferred Skills Score (%)": c.get("preferred_skill_score", 0.0),
                        "Experience Score (%)": c.get("experience_score", 0.0),
                        "Education Score (%)": c.get("education_score", 100.0),
                        "Matched Skills": ", ".join(c.get("matched_required_skills", [])),
                        "Missing Skills": ", ".join(c.get("missing_required_skills", [])),
                        "Total Experience (Years)": c.get("total_experience_years", 0.0),
                        "Anomalies Detected": anomaly_str,
                        "Explanation": c.get("explanation", ""),
                    }
                )

        return report_path

    def generate_summary(self, ranked_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate high-level metadata summary of ranking run for HR dashboard."""
        if not ranked_candidates:
            raise NotImplementedError("Candidate list is empty; cannot generate summary.")

        scores = [float(c.get("final_score", 0.0)) for c in ranked_candidates]
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0

        return {
            "total_candidates": len(ranked_candidates),
            "top_candidate": ranked_candidates[0].get("candidate_id") if ranked_candidates else None,
            "top_score": ranked_candidates[0].get("final_score") if ranked_candidates else 0.0,
            "average_score": avg_score,
            "high_scorers_count": sum(1 for s in scores if s >= 75.0),
        }
