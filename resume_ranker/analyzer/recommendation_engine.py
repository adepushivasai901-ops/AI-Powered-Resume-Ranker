"""Actionable candidate resume recommendation engine."""

from typing import List, Optional

from resume_ranker.models.candidate_analysis import (
    ATSAnalysis,
    ATSCheckStatus,
    CandidateRecommendation,
    ExperienceQualityAnalysis,
    JobMatchAnalysis,
    PriorityLevel,
    ProjectQualityAnalysis,
    SkillRelevanceAnalysis,
    SummaryQualityAnalysis,
)
from resume_ranker.models.matching import AnomalyFlag, AnomalySeverity
from resume_ranker.models.resume import ParsedResume


class RecommendationEngine:
    """Generates concrete, evidence-grounded recommendations categorized by priority."""

    def generate_recommendations(
        self,
        parsed_resume: ParsedResume,
        ats_analysis: ATSAnalysis,
        exp_quality: ExperienceQualityAnalysis,
        summary_quality: SummaryQualityAnalysis,
        project_quality: ProjectQualityAnalysis,
        skill_analysis: SkillRelevanceAnalysis,
        job_match: Optional[JobMatchAnalysis] = None,
        anomalies: Optional[List[AnomalyFlag]] = None,
    ) -> List[CandidateRecommendation]:
        """Produce prioritized actionable improvements for the candidate.

        Args:
            parsed_resume: Structured parsed resume.
            ats_analysis: ATS readiness results.
            exp_quality: Experience and bullet analysis.
            summary_quality: Professional summary evaluation.
            project_quality: Projects section evaluation.
            skill_analysis: Skill intelligence and domain signals.
            job_match: Optional job description match results.
            anomalies: Detected anomaly flags.

        Returns:
            List of CandidateRecommendation objects sorted by priority (HIGH, MEDIUM, LOW).
        """
        recs: List[CandidateRecommendation] = []
        rec_counter = 1

        def next_id() -> str:
            nonlocal rec_counter
            val = f"REC-{rec_counter:02d}"
            rec_counter += 1
            return val

        # -------------------------------------------------------------
        # HIGH PRIORITY
        # -------------------------------------------------------------

        # 1. Missing Contact Info
        has_email = bool(parsed_resume.candidate.email)
        has_phone = bool(parsed_resume.candidate.phone)
        if not has_email or not has_phone:
            missing_items = []
            if not has_email:
                missing_items.append("email address")
            if not has_phone:
                missing_items.append("phone number")
            recs.append(
                CandidateRecommendation(
                    id=next_id(),
                    priority=PriorityLevel.HIGH,
                    category="Contact Information",
                    issue=f"Missing { ' and '.join(missing_items) } in extractable text.",
                    why_it_matters="Recruiters and automated ATS platforms cannot invite you to interviews if your contact details are unreadable.",
                    suggested_action="Add your active email and phone number plainly at the top of your resume in regular text (avoid placing them inside header/footer boxes or images).",
                    example="Example header format: 'Alex Mercer | Seattle, WA | alex@example.com | (555) 019-2834 | linkedin.com/in/alex'",
                )
            )

        # 2. Critical ATS Issues (Scanned document or severely low text quality)
        for check in ats_analysis.checks:
            if check.code == "TEXT_QUALITY" and check.status == ATSCheckStatus.FAILED:
                recs.append(
                    CandidateRecommendation(
                        id=next_id(),
                        priority=PriorityLevel.HIGH,
                        category="ATS Readiness",
                        issue="Resume appears to be an image scan or flattened graphical file.",
                        why_it_matters="Most commercial ATS parsers cannot read image PDFs, resulting in automated rejection before any human recruiter sees your resume.",
                        suggested_action="Re-export your resume directly as a clean PDF document from Microsoft Word, Google Docs, or LaTeX.",
                        example=None,
                    )
                )

        # 3. Prompt Injections or High Severity Flags
        if anomalies:
            for anom in anomalies:
                if anom.severity == AnomalySeverity.HIGH:
                    recs.append(
                        CandidateRecommendation(
                            id=next_id(),
                            priority=PriorityLevel.HIGH,
                            category="Integrity & Formatting",
                            issue=f"Flagged text pattern detected: {anom.message}",
                            why_it_matters="Unusual hidden instructions, prompt injections, or impossible date ranges trigger compliance flags and lower parsing confidence.",
                            suggested_action="Remove any instructions addressed to automated bots or AI, and ensure all dates follow a verified chronological sequence.",
                            example=None,
                        )
                    )

        # 4. Experience Bullets: Weak phrasing and missing metrics
        if exp_quality.weak_bullet_count >= 1 or (exp_quality.total_bullets >= 1 and exp_quality.measurable_achievement_count == 0):
            recs.append(
                CandidateRecommendation(
                    id=next_id(),
                    priority=PriorityLevel.HIGH,
                    category="Experience & Impact",
                    issue=(
                        f"{exp_quality.weak_bullet_count} bullet(s) start with passive openers ('Responsible for', 'Worked on'), "
                        f"and only {exp_quality.measurable_achievement_count} contain measurable impact metrics."
                    ),
                    why_it_matters="Hiring managers scan for demonstrable business and technical outcomes. Bullets that merely list tasks fail to stand out.",
                    suggested_action=(
                        "Rewrite experience bullets using the Action Verb + Technology/Tool + Task + Measurable Result framework. "
                        "Quantify scale, performance improvements, user counts, or time saved where genuine."
                    ),
                    example=(
                        "Before: 'Responsible for backend Python APIs and database queries.'\n"
                        "After: 'Architected RESTful microservices with FastAPI and PostgreSQL, reducing p95 query latency by 35%.'"
                    ),
                )
            )

        # 5. Missing Required Skills (When JD is provided)
        if job_match and job_match.is_available and job_match.required_skills_missing:
            missing_top = job_match.required_skills_missing[:4]
            recs.append(
                CandidateRecommendation(
                    id=next_id(),
                    priority=PriorityLevel.HIGH,
                    category="Job Alignment",
                    issue=f"Missing {len(job_match.required_skills_missing)} core required skill(s) specified in the job description: {', '.join(missing_top)}.",
                    why_it_matters="Required skills serve as non-negotiable screening filters. Missing them significantly reduces interview invitation rates.",
                    suggested_action=(
                        "If you have authentic experience with these tools, ensure they are explicitly named in your Skills section "
                        "and woven into bullet descriptions. Note: Do not add skills you have never used."
                    ),
                    example=f"Tip: If you've utilized {missing_top[0]} in past positions or projects, describe your practical demonstration.",
                )
            )

        # -------------------------------------------------------------
        # MEDIUM PRIORITY
        # -------------------------------------------------------------

        # 6. Unbacked Skills Ratio (Claimed vs Evidenced)
        if len(skill_analysis.claimed_skills) >= 4 and skill_analysis.evidenced_ratio < 0.40:
            recs.append(
                CandidateRecommendation(
                    id=next_id(),
                    priority=PriorityLevel.MEDIUM,
                    category="Skills Credibility",
                    issue=(
                        f"Only {len(skill_analysis.evidenced_skills)} of {skill_analysis.total_detected} skills "
                        f"({skill_analysis.evidenced_ratio * 100:.0f}%) appear inside experience or project descriptions."
                    ),
                    why_it_matters="Skills listed only in a standalone skills box carry lower credibility with technical interviewers than skills demonstrated in action.",
                    suggested_action="Contextualize your primary skills by mentioning them directly within employment achievements or project writeups.",
                    example="Example: 'Leveraged Docker and GitHub Actions to automate CI/CD pipeline deployments.'",
                )
            )

        # 7. Professional Summary Refinement
        if not summary_quality.detected:
            recs.append(
                CandidateRecommendation(
                    id=next_id(),
                    priority=PriorityLevel.MEDIUM,
                    category="Summary & Profile",
                    issue="No professional summary or career objective detected.",
                    why_it_matters="A concise 3-line summary immediately communicates your professional identity, core domain, and level of experience to recruiters.",
                    suggested_action="Add a 3-4 sentence professional summary at the top of your resume.",
                    example="Example: 'Results-driven Full Stack Engineer with 4+ years of experience designing high-scale React and Python web applications. Expert in PostgreSQL and AWS cloud deployments with a focus on code quality.'",
                )
            )
        elif summary_quality.is_generic or (summary_quality.score and summary_quality.score < 75.0):
            recs.append(
                CandidateRecommendation(
                    id=next_id(),
                    priority=PriorityLevel.MEDIUM,
                    category="Summary & Profile",
                    issue="Professional summary uses generic buzzwords or lacks clear target role definition.",
                    why_it_matters="Generic phrases like 'hardworking team player' waste recruiter attention and do not differentiate your capabilities.",
                    suggested_action="Sharpen your summary by replacing adjectives with specific technical specializations and concrete achievements.",
                    example="Structure: [Professional Title] + [Years of Experience / Domain] + [Top 3 Technologies] + [Core Achievement].",
                )
            )

        # 8. Projects Section Quality
        if not project_quality.detected:
            recs.append(
                CandidateRecommendation(
                    id=next_id(),
                    priority=PriorityLevel.MEDIUM,
                    category="Projects & Portfolio",
                    issue="No dedicated projects section identified.",
                    why_it_matters="Technical projects provide tangible evidence of problem-solving ability and modern stack proficiency, especially for developer roles.",
                    suggested_action="Include 2-3 significant technical projects detailing the problem solved, tech stack, and links to GitHub or live demos.",
                    example="Example Project: 'AI Document Classifier | Python, PyTorch, FastAPI | Built text classifier with 92% accuracy across 10,000 documents; deployed with Docker.'",
                )
            )

        # 9. Preferred Skills for Target Job (When JD is provided)
        if job_match and job_match.is_available and job_match.preferred_skills_missing:
            pref_top = job_match.preferred_skills_missing[:3]
            recs.append(
                CandidateRecommendation(
                    id=next_id(),
                    priority=PriorityLevel.MEDIUM,
                    category="Job Alignment",
                    issue=f"Missing {len(job_match.preferred_skills_missing)} preferred/nice-to-have skill(s): {', '.join(pref_top)}.",
                    why_it_matters="Preferred skills elevate you above other qualified applicants during final candidate selection.",
                    suggested_action="If you have completed courses, certifications, or personal projects with these tools, note them on your resume.",
                    example=None,
                )
            )

        # -------------------------------------------------------------
        # LOW PRIORITY
        # -------------------------------------------------------------

        # 10. Date Format Standardization
        has_date_warn = any(c.code == "DATE_CONSISTENCY" and c.status != ATSCheckStatus.PASSED for c in ats_analysis.checks)
        if has_date_warn:
            recs.append(
                CandidateRecommendation(
                    id=next_id(),
                    priority=PriorityLevel.LOW,
                    category="Formatting & Structure",
                    issue="Some employment or education dates lack consistent month/year intervals.",
                    why_it_matters="Automated timeline parsers calculate tenure more accurately when dates follow a standardized structure.",
                    suggested_action="Standardize all dates to 'Month YYYY – Month YYYY' (or 'Present' for your current role).",
                    example="Example: '06/2021 – 08/2023' or 'Jun 2021 – Aug 2023'.",
                )
            )

        # 11. Section Headings Standardizing
        for check in ats_analysis.checks:
            if check.code == "STANDARD_HEADINGS" and check.status == ATSCheckStatus.WARNING:
                recs.append(
                    CandidateRecommendation(
                        id=next_id(),
                        priority=PriorityLevel.LOW,
                        category="ATS Compatibility",
                        issue=check.message,
                        why_it_matters="Non-standard headings may cause older ATS systems to misclassify experience or skills into the wrong sections.",
                        suggested_action="Use standard, universal section titles: 'Professional Experience', 'Education', 'Skills', 'Projects'.",
                        example=None,
                    )
                )

        # 12. Formatting & Special Characters
        for check in ats_analysis.checks:
            if check.code == "SPECIAL_CHARACTERS" and check.status == ATSCheckStatus.WARNING:
                recs.append(
                    CandidateRecommendation(
                        id=next_id(),
                        priority=PriorityLevel.LOW,
                        category="Visual & Layout",
                        issue="High density of non-standard symbols, decorative dividers, or icons.",
                        why_it_matters="Complex graphic dividers and custom icon fonts can render as unreadable glyphs on legacy applicant platforms.",
                        suggested_action="Use standard bullet points (• or -) and avoid decorative symbols or skill proficiency rating bars.",
                        example=None,
                    )
                )

        # 13. Skills Breadth Recommendation
        if skill_analysis.total_detected < 6:
            recs.append(
                CandidateRecommendation(
                    id=next_id(),
                    priority=PriorityLevel.MEDIUM,
                    category="Skills Inventory",
                    issue=f"Only {skill_analysis.total_detected} technical/domain skill(s) detected in the resume.",
                    why_it_matters="A broader inventory of relevant libraries, developer tools, and frameworks improves keyword matching across ATS filters.",
                    suggested_action="Expand your skills section with modern tools, databases, and libraries relevant to your specialization.",
                    example="Example: Add databases (e.g. PostgreSQL, Redis), cloud tools (AWS, Docker), and testing frameworks (pytest).",
                )
            )

        # 14. Fallback Recommendation
        if not recs:
            recs.append(
                CandidateRecommendation(
                    id=next_id(),
                    priority=PriorityLevel.LOW,
                    category="General Optimization",
                    issue="Resume demonstrates solid baseline structure and impact metrics.",
                    why_it_matters="Continuous tailoring to specific target roles maximizes recruiter response rates.",
                    suggested_action="Before submitting applications, paste the specific job description to audit tailored keyword coverage.",
                    example=None,
                )
            )

        # Sort recommendations: HIGH -> MEDIUM -> LOW
        priority_order = {PriorityLevel.HIGH: 0, PriorityLevel.MEDIUM: 1, PriorityLevel.LOW: 2}
        recs.sort(key=lambda r: priority_order[r.priority])

        return recs
