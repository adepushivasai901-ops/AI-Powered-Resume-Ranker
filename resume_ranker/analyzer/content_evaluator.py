"""Content and section quality evaluator for candidate resumes."""

import re
from typing import Any, Dict, List, Optional, Tuple

from resume_ranker.models.candidate_analysis import (
    BulletQuality,
    ExperienceQualityAnalysis,
    ProjectItemAnalysis,
    ProjectQualityAnalysis,
    SectionQualityDetail,
    SummaryQualityAnalysis,
)
from resume_ranker.models.resume import ParsedResume


class ContentEvaluator:
    """Evaluates the linguistic quality, impact metrics, and depth of resume sections."""

    WEAK_PATTERNS = [
        (re.compile(r"^\s*(responsible for\b|responsibilities included\b)", re.IGNORECASE), "Responsible for"),
        (re.compile(r"^\s*(worked on\b|worked with\b)", re.IGNORECASE), "Worked on/with"),
        (re.compile(r"^\s*(helped with\b|helped to\b|assisted with\b|assisted in\b)", re.IGNORECASE), "Helped/Assisted"),
        (re.compile(r"^\s*(involved in\b|participated in\b)", re.IGNORECASE), "Involved/Participated in"),
        (re.compile(r"^\s*(tasked with\b|duties included\b)", re.IGNORECASE), "Tasked with"),
        (re.compile(r"^\s*(handled\b)", re.IGNORECASE), "Handled"),
    ]

    METRIC_PATTERNS = [
        re.compile(r"\b\d+(?:\.\d+)?%", re.IGNORECASE),  # percentages (e.g. 25%, 3.5%)
        re.compile(r"[\$€£]\s*\d+[\d,]*(?:\.\d+)?(?:\s*[kKmMbB](?:illion)?)?", re.IGNORECASE),  # currency
        re.compile(r"\b\d+x\b", re.IGNORECASE),  # multipliers (e.g. 3x, 10x)
        re.compile(r"\b\d+[\d,]*\+?\s*(?:users|clients|customers|students|downloads|requests|events|queries|models|pipelines)\b", re.IGNORECASE),
        re.compile(r"\b(?:reduced|increased|improved|boosted|saved|scaled|accelerated)\b[^\.\n]*?\b\d+[\d,]*\b", re.IGNORECASE),
        re.compile(r"\b\d+[\d,]*\s*(?:ms|seconds|minutes|hours|days|weeks|months)\b", re.IGNORECASE),
    ]

    GENERIC_BUZZWORDS = [
        "hardworking", "hard-working", "go-getter", "team player", "detail-oriented",
        "self-starter", "results-driven", "fast learner", "think outside the box",
        "motivated professional", "dynamic individual", "proven track record"
    ]

    ROLE_SIGNALS = [
        "engineer", "developer", "architect", "scientist", "analyst", "designer",
        "manager", "lead", "administrator", "consultant", "specialist", "programmer",
        "researcher", "coordinator", "officer", "strategist", "intern"
    ]

    def evaluate_bullet(self, bullet_text: str) -> BulletQuality:
        """Evaluate an individual bullet point for impact, action verbs, and quantifiable metrics."""
        cleaned = bullet_text.strip().lstrip("•-*–— ").strip()
        if not cleaned:
            return BulletQuality(
                raw_text=bullet_text,
                score=50.0,
                is_weak=False,
                has_metrics=False,
                suggestion="Expand bullet point with substantive detail.",
            )

        # Check for weak patterns
        matched_weak = None
        for pattern, label in self.WEAK_PATTERNS:
            if pattern.search(cleaned):
                matched_weak = label
                break

        # Check for metrics
        found_metrics = []
        for pat in self.METRIC_PATTERNS:
            matches = pat.findall(cleaned)
            for m in matches:
                if isinstance(m, tuple):
                    found_metrics.append(m[0])
                elif isinstance(m, str) and m:
                    found_metrics.append(m)

        has_metrics = len(found_metrics) > 0

        # Score & Recommendation Logic
        if matched_weak and not has_metrics:
            score = 42.0
            suggestion = (
                f"Replace the passive opener '{matched_weak}' with an active verb. "
                "Structure bullets as: Action Verb + Technology/Method + Task + Measurable Result."
            )
            example = (
                "Example: 'Engineered scalable background workers using Celery and Redis, "
                "reducing async job latency by 40%' instead of 'Worked on background processing.'"
            )
        elif matched_weak and has_metrics:
            score = 68.0
            suggestion = (
                f"Opening starts with passive phrasing ('{matched_weak}'). "
                "Strengthen by leading with a strong action verb (e.g. 'Optimized', 'Architected', 'Spearheaded')."
            )
            example = "Example: 'Accelerated query execution by 35% through PostgreSQL indexing' instead of 'Responsible for query tuning.'"
        elif not matched_weak and not has_metrics:
            score = 78.0
            suggestion = (
                "Good action orientation. To make it even stronger, add quantified impact where genuine "
                "(e.g. percentage improvement, scale of users, latency, or time saved)."
            )
            example = "Example: 'Designed authentication microservice supporting 50,000+ daily active users.'"
        else:
            score = 96.0
            suggestion = "Strong bullet point combining clear active verbs with measurable business or technical outcomes."
            example = None

        return BulletQuality(
            raw_text=cleaned,
            score=score,
            is_weak=(matched_weak is not None),
            weak_pattern=matched_weak,
            has_metrics=has_metrics,
            detected_metrics=found_metrics[:3],
            suggestion=suggestion,
            example=example,
        )

    def evaluate_experience(self, parsed_resume: ParsedResume) -> ExperienceQualityAnalysis:
        """Evaluate work history depth, bullet impact, and tenure clarity."""
        roles = parsed_resume.experience
        all_bullets: List[BulletQuality] = []

        for role in roles:
            bullets = role.responsibilities + role.achievements
            for b in bullets:
                if len(b.strip()) > 5:
                    all_bullets.append(self.evaluate_bullet(b))

        total_bullets = len(all_bullets)
        weak_count = sum(1 for b in all_bullets if b.is_weak)
        metric_count = sum(1 for b in all_bullets if b.has_metrics)

        total_years = parsed_resume.timeline.total_experience_years if parsed_resume.timeline else 0.0
        duration_display = f"{total_years:.1f} years" if total_years > 0 else "Duration not determined"

        strengths: List[str] = []
        weaknesses: List[str] = []
        recommendations: List[str] = []

        if len(roles) >= 1:
            strengths.append(f"Structured employment history detailing {len(roles)} role(s).")
        else:
            weaknesses.append("No explicit work experience roles detected.")
            recommendations.append("Add employment history or relevant internships with dates, job titles, and organizations.")

        if total_bullets > 0:
            avg_bullet_score = sum(b.score for b in all_bullets) / total_bullets
            if metric_count > 0:
                strengths.append(f"Contains {metric_count} quantified outcome(s) or measurable impact metrics.")
            else:
                weaknesses.append("Bullets lack measurable outcomes (e.g. percentages, scale, speed, revenue, or users).")
                recommendations.append(
                    "Incorporate measurable outcomes into experience bullets. "
                    "Example: 'Decreased page load times by 28%' instead of 'Improved performance.'"
                )

            if weak_count > 0:
                weaknesses.append(f"{weak_count} bullet(s) begin with passive phrasing ('Worked on', 'Responsible for').")
                recommendations.append(
                    "Rewrite passive bullets to begin with strong past-tense action verbs (e.g. 'Built', 'Led', 'Automated', 'Designed')."
                )
            else:
                strengths.append("Bullets consistently employ active, direct action verbs.")

            final_exp_score = round(avg_bullet_score, 1)
        else:
            final_exp_score = 40.0 if len(roles) > 0 else 0.0

        return ExperienceQualityAnalysis(
            score=final_exp_score,
            total_roles=len(roles),
            total_duration_years=total_years,
            duration_display=duration_display,
            bullets=all_bullets,
            total_bullets=total_bullets,
            weak_bullet_count=weak_count,
            measurable_achievement_count=metric_count,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )

    def evaluate_summary(self, parsed_resume: ParsedResume) -> SummaryQualityAnalysis:
        """Evaluate the professional summary for brevity, role clarity, and technical substance."""
        raw_summary = parsed_resume.summary
        if not raw_summary or len(raw_summary.strip()) < 10:
            return SummaryQualityAnalysis(
                detected=False,
                score=None,
                word_count=0,
                has_target_role=False,
                has_tech_skills=False,
                is_generic=False,
                strengths=[],
                weaknesses=["No professional summary or career profile detected."],
                suggestion=(
                    "Add a focused 3-4 sentence professional summary highlighting your target role, "
                    "years of experience, and top core competencies."
                ),
            )

        text = raw_summary.strip()
        words = text.split()
        word_count = len(words)
        lower_text = text.lower()

        has_target_role = any(re.search(rf"\b{re.escape(r)}\b", lower_text) for r in self.ROLE_SIGNALS)
        has_tech = any(s.canonical_name.lower() in lower_text for s in parsed_resume.skills)
        buzzwords_found = [bw for bw in self.GENERIC_BUZZWORDS if bw in lower_text]
        is_generic = len(buzzwords_found) >= 2

        score = 80.0
        strengths: List[str] = []
        weaknesses: List[str] = []

        if 15 <= word_count <= 100:
            strengths.append(f"Ideal summary length ({word_count} words).")
            score += 10.0
        elif word_count < 15:
            weaknesses.append("Summary is very brief (under 15 words).")
            score -= 15.0
        elif word_count > 120:
            weaknesses.append("Summary is lengthy (over 120 words). Recruiters prefer concise overviews.")
            score -= 10.0

        if has_target_role:
            strengths.append("Clearly states a target professional role or title.")
            score += 5.0
        else:
            weaknesses.append("Missing an explicit target role or professional identity.")
            score -= 10.0

        if has_tech:
            strengths.append("Integrates key technical competencies naturally.")
            score += 5.0
        else:
            weaknesses.append("Does not highlight core tools, languages, or methodologies.")
            score -= 10.0

        if is_generic:
            weaknesses.append(f"Contains generic filler buzzwords: {', '.join(buzzwords_found[:3])}.")
            score -= 15.0
        else:
            strengths.append("Avoids overused, generic filler cliches.")

        final_score = round(max(min(score, 100.0), 30.0), 1)

        suggestion = (
            "Focus your summary around: [Target Role] + [Years of Experience / Specialization] + "
            "[3-4 Primary Tech Stack Strengths] + [Key Career Impact]."
        )

        return SummaryQualityAnalysis(
            detected=True,
            score=final_score,
            word_count=word_count,
            has_target_role=has_target_role,
            has_tech_skills=has_tech,
            is_generic=is_generic,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestion=suggestion,
        )

    def evaluate_projects(self, parsed_resume: ParsedResume) -> ProjectQualityAnalysis:
        """Evaluate technical projects, demonstrations, and portfolio links."""
        projects = parsed_resume.projects
        has_project_section = "PROJECTS" in parsed_resume.sections_detected

        if not projects and not has_project_section:
            return ProjectQualityAnalysis(
                detected=False,
                score=None,
                project_count=0,
                projects=[],
                strengths=[],
                weaknesses=["No projects section detected."],
                recommendations=[
                    "Add 2-3 technical or portfolio projects demonstrating practical application of your skills, "
                    "including tools used and GitHub/demo links."
                ],
            )

        items: List[ProjectItemAnalysis] = []
        strengths: List[str] = []
        weaknesses: List[str] = []
        recommendations: List[str] = []

        score_acc = 70.0
        for p in projects:
            desc = p.description or ""
            has_tech = len(p.technologies) > 0
            has_links = len(p.links) > 0

            # check metrics in project description
            has_metric = any(pat.search(desc) for pat in self.METRIC_PATTERNS)

            feedback_parts = []
            if has_tech:
                feedback_parts.append(f"Technologies highlighted: {', '.join(p.technologies[:4])}.")
            else:
                feedback_parts.append("Consider listing the explicit technologies and libraries utilized.")

            if has_links:
                feedback_parts.append("Includes repository or live demonstration link.")

            items.append(
                ProjectItemAnalysis(
                    name=p.name,
                    description=desc,
                    technologies=p.technologies,
                    has_metrics=has_metric,
                    links=p.links,
                    feedback=" ".join(feedback_parts),
                )
            )

        if items:
            strengths.append(f"{len(items)} project(s) identified with technical descriptions.")
            tech_count = sum(1 for it in items if it.technologies)
            if tech_count == len(items):
                strengths.append("All projects explicitly mention associated technologies.")
                score_acc += 15.0
            else:
                weaknesses.append("Some projects do not list the underlying tech stack.")
                score_acc -= 10.0
                recommendations.append("Explicitly state tools and frameworks for each project entry.")

            links_count = sum(1 for it in items if it.links)
            if links_count > 0:
                strengths.append("Includes active links to repositories or live demos.")
                score_acc += 10.0

            final_score = round(max(min(score_acc, 100.0), 40.0), 1)
        else:
            final_score = 65.0
            weaknesses.append("Projects section detected, but individual project titles could not be cleanly segmented.")
            recommendations.append("Format project titles distinctly in bold on their own lines.")

        return ProjectQualityAnalysis(
            detected=True,
            score=final_score,
            project_count=len(items),
            projects=items,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )

    def evaluate_all_sections(
        self,
        parsed_resume: ParsedResume,
        exp_quality: ExperienceQualityAnalysis,
        summary_quality: SummaryQualityAnalysis,
        proj_quality: ProjectQualityAnalysis,
    ) -> Dict[str, SectionQualityDetail]:
        """Produce comprehensive section-by-section health analysis."""
        sections: Dict[str, SectionQualityDetail] = {}

        # 1. Contact
        has_contact = bool(parsed_resume.candidate.email or parsed_resume.candidate.phone)
        contact_strengths = []
        contact_weaknesses = []
        if parsed_resume.candidate.email:
            contact_strengths.append("Email address present")
        else:
            contact_weaknesses.append("Missing email address")
        if parsed_resume.candidate.phone:
            contact_strengths.append("Phone number present")
        else:
            contact_weaknesses.append("Missing phone number")
        if parsed_resume.candidate.links:
            contact_strengths.append("Profile/portfolio links detected")

        contact_score = 100.0 if (parsed_resume.candidate.email and parsed_resume.candidate.phone) else (60.0 if has_contact else 20.0)
        sections["CONTACT"] = SectionQualityDetail(
            section_key="CONTACT",
            display_name="Contact Information",
            detected=has_contact,
            score=contact_score,
            status="Strong" if contact_score >= 80 else ("Needs Improvement" if has_contact else "Missing"),
            strengths=contact_strengths,
            weaknesses=contact_weaknesses,
            recommendation="Place email, phone, location, and LinkedIn/GitHub plainly at the top of the resume." if contact_score < 100 else None,
        )

        # 2. Summary
        sections["SUMMARY"] = SectionQualityDetail(
            section_key="SUMMARY",
            display_name="Professional Summary",
            detected=summary_quality.detected,
            score=summary_quality.score,
            status="Strong" if (summary_quality.score or 0) >= 80 else ("Needs Improvement" if summary_quality.detected else "Not Detected"),
            strengths=summary_quality.strengths,
            weaknesses=summary_quality.weaknesses,
            recommendation=summary_quality.suggestion,
        )

        # 3. Experience
        has_exp = len(parsed_resume.experience) > 0
        sections["EXPERIENCE"] = SectionQualityDetail(
            section_key="EXPERIENCE",
            display_name="Work Experience",
            detected=has_exp,
            score=exp_quality.score if has_exp else None,
            status="Strong" if exp_quality.score >= 80 else ("Needs Improvement" if has_exp else "Not Detected"),
            strengths=exp_quality.strengths,
            weaknesses=exp_quality.weaknesses,
            recommendation=exp_quality.recommendations[0] if exp_quality.recommendations else None,
        )

        # 4. Skills
        has_skills = len(parsed_resume.skills) > 0
        skills_score = 90.0 if len(parsed_resume.skills) >= 6 else (75.0 if len(parsed_resume.skills) >= 3 else 45.0)
        skills_strengths = [f"{len(parsed_resume.skills)} verified canonical skills detected."]
        skills_weaknesses = []
        if len(parsed_resume.skills) < 4:
            skills_weaknesses.append("Few technical skills detected.")
        sections["SKILLS"] = SectionQualityDetail(
            section_key="SKILLS",
            display_name="Skills & Competencies",
            detected=has_skills,
            score=skills_score if has_skills else None,
            status="Strong" if skills_score >= 80 else ("Needs Improvement" if has_skills else "Not Detected"),
            strengths=skills_strengths,
            weaknesses=skills_weaknesses,
            recommendation="Categorize skills (e.g. Languages, Frameworks, Tools) to improve readability." if skills_score < 90 else None,
        )

        # 5. Projects
        sections["PROJECTS"] = SectionQualityDetail(
            section_key="PROJECTS",
            display_name="Projects & Portfolio",
            detected=proj_quality.detected,
            score=proj_quality.score,
            status="Strong" if (proj_quality.score or 0) >= 80 else ("Needs Improvement" if proj_quality.detected else "Not Detected"),
            strengths=proj_quality.strengths,
            weaknesses=proj_quality.weaknesses,
            recommendation=proj_quality.recommendations[0] if proj_quality.recommendations else None,
        )

        # 6. Education
        has_edu = len(parsed_resume.education) > 0
        edu_score = 95.0 if has_edu else None
        edu_strengths = [f"{len(parsed_resume.education)} education entry/entries detected."] if has_edu else []
        edu_weaknesses = ["No education details detected."] if not has_edu else []
        sections["EDUCATION"] = SectionQualityDetail(
            section_key="EDUCATION",
            display_name="Education",
            detected=has_edu,
            score=edu_score,
            status="Strong" if has_edu else "Not Detected",
            strengths=edu_strengths,
            weaknesses=edu_weaknesses,
            recommendation="Include degree, field of study, institution, and completion year." if not has_edu else None,
        )

        return sections
