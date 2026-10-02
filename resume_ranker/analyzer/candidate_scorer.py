"""Candidate Resume Health Scorer with deterministic dynamic weight renormalization."""

from typing import Dict, List, Optional
import numpy as np

from resume_ranker.models.candidate_analysis import (
    ATSAnalysis,
    ExperienceQualityAnalysis,
    HealthScoreComponent,
    ProjectQualityAnalysis,
    ResumeHealthScore,
    SkillRelevanceAnalysis,
    SummaryQualityAnalysis,
)
from resume_ranker.models.resume import ParsedResume


class CandidateResumeScorer:
    """Computes a transparent, explainable 7-component Resume Health Score with dynamic weight renormalization."""

    DEFAULT_WEIGHTS = {
        "ats": 0.20,
        "content": 0.20,
        "skills": 0.20,
        "experience": 0.15,
        "projects": 0.10,
        "education": 0.05,
        "structure": 0.10,
    }

    def compute_health_score(
        self,
        parsed_resume: ParsedResume,
        ats_analysis: ATSAnalysis,
        exp_quality: ExperienceQualityAnalysis,
        summary_quality: SummaryQualityAnalysis,
        project_quality: ProjectQualityAnalysis,
        skill_analysis: SkillRelevanceAnalysis,
    ) -> ResumeHealthScore:
        """Calculate composite Resume Health Score with dynamic weight renormalization for missing sections.

        Args:
            parsed_resume: Structured parsed resume.
            ats_analysis: Evaluated ATS readiness.
            exp_quality: Work experience and bullet quality.
            summary_quality: Professional summary evaluation.
            project_quality: Projects section evaluation.
            skill_analysis: Skill diversity and evidence classification.

        Returns:
            ResumeHealthScore with overall score, rating label, and itemized components.
        """
        components: Dict[str, HealthScoreComponent] = {}

        # 1. ATS Compatibility (20%)
        components["ats"] = HealthScoreComponent(
            name="ATS Readiness",
            score=ats_analysis.score,
            default_weight=self.DEFAULT_WEIGHTS["ats"],
            is_active=True,
            status="Strong" if ats_analysis.score >= 80 else ("Moderate" if ats_analysis.score >= 65 else "Needs Improvement"),
            explanation=f"{ats_analysis.status_label} ({ats_analysis.passed_count} checks passed).",
        )

        # 2. Content Quality (20%) - Evaluates bullet phrasing, active verbs, and summary depth
        content_scores = []
        if exp_quality.bullets:
            content_scores.append(sum(b.score for b in exp_quality.bullets) / len(exp_quality.bullets))
        if summary_quality.detected and summary_quality.score is not None:
            content_scores.append(summary_quality.score)
        
        avg_content = float(np.mean(content_scores)) if content_scores else 60.0
        content_score_val = round(avg_content, 1)
        components["content"] = HealthScoreComponent(
            name="Content Quality",
            score=content_score_val,
            default_weight=self.DEFAULT_WEIGHTS["content"],
            is_active=True,
            status="Strong" if content_score_val >= 80 else ("Good" if content_score_val >= 65 else "Needs Improvement"),
            explanation=(
                f"Evaluated {len(exp_quality.bullets)} experience bullet(s) and summary impact. "
                f"{exp_quality.measurable_achievement_count} quantifiable outcome(s) found."
            ),
        )

        # 3. Skills Relevance & Depth (20%)
        components["skills"] = HealthScoreComponent(
            name="Skills Relevance",
            score=skill_analysis.score,
            default_weight=self.DEFAULT_WEIGHTS["skills"],
            is_active=True,
            status="Strong" if skill_analysis.score >= 80 else ("Moderate" if skill_analysis.score >= 65 else "Needs Improvement"),
            explanation=(
                f"{skill_analysis.total_detected} canonical skill(s) across {len(skill_analysis.categories)} domain(s). "
                f"{len(skill_analysis.evidenced_skills)} substantiated in experience or projects."
            ),
        )

        # 4. Experience Quality (15%)
        if len(parsed_resume.experience) > 0 and exp_quality.total_bullets > 0:
            exp_score = exp_quality.score
            components["experience"] = HealthScoreComponent(
                name="Experience Quality",
                score=exp_score,
                default_weight=self.DEFAULT_WEIGHTS["experience"],
                is_active=True,
                status="Strong" if exp_score >= 80 else ("Good" if exp_score >= 65 else "Needs Improvement"),
                explanation=f"{exp_quality.total_roles} role(s) spanning {exp_quality.duration_display}.",
            )
        else:
            components["experience"] = HealthScoreComponent(
                name="Experience Quality",
                score=None,
                default_weight=self.DEFAULT_WEIGHTS["experience"],
                is_active=False,
                status="Not Detected",
                explanation="No explicit work experience history detected. Excluded from score calculation.",
            )

        # 5. Projects & Achievements (10%)
        if project_quality.detected and project_quality.score is not None:
            proj_score = project_quality.score
            components["projects"] = HealthScoreComponent(
                name="Projects & Achievements",
                score=proj_score,
                default_weight=self.DEFAULT_WEIGHTS["projects"],
                is_active=True,
                status="Strong" if proj_score >= 80 else ("Good" if proj_score >= 65 else "Needs Improvement"),
                explanation=f"{project_quality.project_count} technical project(s) identified with associated tools.",
            )
        else:
            components["projects"] = HealthScoreComponent(
                name="Projects & Achievements",
                score=None,
                default_weight=self.DEFAULT_WEIGHTS["projects"],
                is_active=False,
                status="Not Detected",
                explanation="No dedicated projects section detected. Excluded from score calculation.",
            )

        # 6. Education (5%)
        if len(parsed_resume.education) > 0:
            edu_score = 95.0
            components["education"] = HealthScoreComponent(
                name="Education",
                score=edu_score,
                default_weight=self.DEFAULT_WEIGHTS["education"],
                is_active=True,
                status="Strong",
                explanation=f"{len(parsed_resume.education)} education qualification(s) verified in document.",
            )
        else:
            components["education"] = HealthScoreComponent(
                name="Education",
                score=None,
                default_weight=self.DEFAULT_WEIGHTS["education"],
                is_active=False,
                status="Not Detected",
                explanation="No formal education credentials detected. Excluded from score calculation.",
            )

        # 7. Structure & Completeness (10%)
        detected_sections = set(parsed_resume.sections_detected.keys())
        expected_sections = ["CONTACT", "EXPERIENCE", "SKILLS", "EDUCATION", "PROJECTS", "SUMMARY"]
        present_count = 0
        if parsed_resume.candidate.email or parsed_resume.candidate.phone:
            present_count += 1
        if "EXPERIENCE" in detected_sections or len(parsed_resume.experience) > 0:
            present_count += 1
        if "SKILLS" in detected_sections or len(parsed_resume.skills) > 0:
            present_count += 1
        if "EDUCATION" in detected_sections or len(parsed_resume.education) > 0:
            present_count += 1
        if "PROJECTS" in detected_sections or len(parsed_resume.projects) > 0:
            present_count += 1
        if "SUMMARY" in detected_sections or parsed_resume.summary:
            present_count += 1

        structure_score = round((present_count / len(expected_sections)) * 100.0, 1)
        components["structure"] = HealthScoreComponent(
            name="Structure & Completeness",
            score=structure_score,
            default_weight=self.DEFAULT_WEIGHTS["structure"],
            is_active=True,
            status="Strong" if structure_score >= 80 else ("Good" if structure_score >= 60 else "Needs Improvement"),
            explanation=f"{present_count} of 6 recommended standard resume sections detected.",
        )

        # Dynamic Renormalization
        active_components = [c for c in components.values() if c.is_active and c.score is not None]
        total_active_w = sum(c.default_weight for c in active_components)

        if total_active_w > 0:
            for c in components.values():
                if c.is_active and c.score is not None:
                    c.effective_weight = round(c.default_weight / total_active_w, 4)
                else:
                    c.effective_weight = 0.0

            overall_score = sum(c.score * c.effective_weight for c in active_components)
            final_overall = round(float(np.clip(overall_score, 0.0, 100.0)), 1)
        else:
            final_overall = 0.0

        # Qualitative label
        if final_overall >= 88.0:
            label = "Exceptional — Highly competitive, well-balanced resume"
        elif final_overall >= 78.0:
            label = "Good — Strong foundation, a few key improvements will elevate it"
        elif final_overall >= 65.0:
            label = "Needs Improvement — Several critical areas need strengthening"
        else:
            label = "Needs Significant Work — Key sections and ATS elements require attention"

        active_count = len(active_components)
        explanation = (
            f"Calculated across {active_count} active components renormalized to 100%. "
            f"Overall health rating: {label}."
        )

        return ResumeHealthScore(
            overall_score=final_overall,
            label=label,
            components=components,
            explanation=explanation,
        )
