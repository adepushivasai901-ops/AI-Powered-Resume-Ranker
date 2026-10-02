"""ATS Readiness Assessor for evaluating resume format, headings, contact info, and parser compatibility."""

import re
from typing import Dict, List, Optional

from resume_ranker.extraction.quality_assessor import DocumentQualityReport
from resume_ranker.models.candidate_analysis import (
    ATSAnalysis,
    ATSCheckItem,
    ATSCheckStatus,
)
from resume_ranker.models.resume import ParsedResume


class ATSAssessor:
    """Evaluates candidate resume text against deterministic Applicant Tracking System (ATS) parsing rules."""

    def evaluate_ats_readiness(
        self,
        parsed_resume: ParsedResume,
        quality_report: Optional[DocumentQualityReport] = None,
    ) -> ATSAnalysis:
        """Run comprehensive ATS readiness audit on parsed resume.

        Args:
            parsed_resume: Structured parsed resume document.
            quality_report: Optional PDF extraction quality report.

        Returns:
            ATSAnalysis with score, status label, itemized checks, and warnings.
        """
        checks: List[ATSCheckItem] = []
        scores: Dict[str, float] = {}

        # 1. Contact Information Check
        has_email = bool(parsed_resume.candidate.email)
        has_phone = bool(parsed_resume.candidate.phone)
        if has_email and has_phone:
            checks.append(
                ATSCheckItem(
                    code="CONTACT_INFO",
                    title="Contact Information",
                    status=ATSCheckStatus.PASSED,
                    message="Email address and phone number cleanly detected in plain text.",
                    fix_tip="Keep your contact info in the body header, avoiding header/footer margin zones.",
                )
            )
            scores["contact"] = 15.0
        elif has_email or has_phone:
            detected_part = "email" if has_email else "phone"
            missing_part = "phone number" if has_email else "email address"
            checks.append(
                ATSCheckItem(
                    code="CONTACT_INFO",
                    title="Contact Information",
                    status=ATSCheckStatus.WARNING,
                    message=f"Found {detected_part}, but {missing_part} could not be confidently identified.",
                    fix_tip="Include both an email address and a direct phone number formatted clearly (e.g. +1 555-123-4567).",
                )
            )
            scores["contact"] = 8.0
        else:
            checks.append(
                ATSCheckItem(
                    code="CONTACT_INFO",
                    title="Contact Information",
                    status=ATSCheckStatus.FAILED,
                    message="No email address or phone number could be extracted from the document.",
                    fix_tip="Ensure contact details are typed as selectable plain text and not embedded inside an image or complex graphics.",
                )
            )
            scores["contact"] = 0.0

        # 2. Text Extraction & Font/Glyph Readability
        if quality_report and quality_report.is_scanned_image:
            checks.append(
                ATSCheckItem(
                    code="TEXT_QUALITY",
                    title="Text Extraction Quality",
                    status=ATSCheckStatus.FAILED,
                    message="Resume appears to be a scanned image or non-selectable PDF. ATS parsers cannot read image files.",
                    fix_tip="Export your resume directly as a clean PDF from Google Docs, Microsoft Word, or LaTeX.",
                )
            )
            scores["extraction"] = 0.0
        elif quality_report and (quality_report.suspicious_chars_ratio > 0.05 or quality_report.confidence < 0.60):
            checks.append(
                ATSCheckItem(
                    code="TEXT_QUALITY",
                    title="Text Extraction Quality",
                    status=ATSCheckStatus.WARNING,
                    message="Low text density or non-standard glyph encoding detected. Some text may be garbled by ATS parsers.",
                    fix_tip="Avoid multi-column tables, text boxes, and unusual custom fonts. Use standard fonts like Calibri, Arial, or Georgia.",
                )
            )
            scores["extraction"] = 10.0
        else:
            checks.append(
                ATSCheckItem(
                    code="TEXT_QUALITY",
                    title="Text Extraction Quality",
                    status=ATSCheckStatus.PASSED,
                    message="Clean text encoding with high extraction readability and selectable font glyphs.",
                    fix_tip="Continue using single-column or clean two-column layouts without nested tables.",
                )
            )
            scores["extraction"] = 20.0

        # 3. Standard Section Headings
        detected_sections = set(parsed_resume.sections_detected.keys())
        standard_sections = {"EXPERIENCE", "EDUCATION", "SKILLS"}
        found_standard = standard_sections.intersection(detected_sections)

        if len(found_standard) == 3:
            checks.append(
                ATSCheckItem(
                    code="STANDARD_HEADINGS",
                    title="Standard Section Headings",
                    status=ATSCheckStatus.PASSED,
                    message="All core standard section headers (Experience, Education, Skills) are present and recognized.",
                    fix_tip="Standard headings ensure ATS indexing algorithms place your credentials in the correct profile fields.",
                )
            )
            scores["headings"] = 10.0
        elif len(found_standard) >= 1:
            missing_names = [s.capitalize() for s in (standard_sections - found_standard)]
            checks.append(
                ATSCheckItem(
                    code="STANDARD_HEADINGS",
                    title="Standard Section Headings",
                    status=ATSCheckStatus.WARNING,
                    message=f"Some standard headings were not recognized: {', '.join(missing_names)}.",
                    fix_tip="Use conventional heading titles like 'Work Experience', 'Education', and 'Skills' rather than creative aliases.",
                )
            )
            scores["headings"] = 5.0
        else:
            checks.append(
                ATSCheckItem(
                    code="STANDARD_HEADINGS",
                    title="Standard Section Headings",
                    status=ATSCheckStatus.FAILED,
                    message="Standard section headings could not be identified.",
                    fix_tip="Format section headers in bold uppercase on their own line with standard names.",
                )
            )
            scores["headings"] = 0.0

        # 4. Work Experience Structure
        if len(parsed_resume.experience) >= 1:
            roles_with_dates = [r for r in parsed_resume.experience if r.start_date or r.end_date]
            if len(roles_with_dates) == len(parsed_resume.experience):
                checks.append(
                    ATSCheckItem(
                        code="EXPERIENCE_STRUCTURE",
                        title="Work Experience Section",
                        status=ATSCheckStatus.PASSED,
                        message=f"{len(parsed_resume.experience)} role(s) detected with clear job titles and date ranges.",
                        fix_tip="Maintain chronological or reverse-chronological order with month and year for every position.",
                    )
                )
                scores["experience"] = 15.0
            else:
                checks.append(
                    ATSCheckItem(
                        code="EXPERIENCE_STRUCTURE",
                        title="Work Experience Section",
                        status=ATSCheckStatus.WARNING,
                        message=f"Detected {len(parsed_resume.experience)} role(s), but some entries lack explicit start or end dates.",
                        fix_tip="Add complete dates (e.g. 'Jan 2021 – Present') to every role so ATS engines calculate experience accurately.",
                    )
                )
                scores["experience"] = 9.0
        else:
            checks.append(
                ATSCheckItem(
                    code="EXPERIENCE_STRUCTURE",
                    title="Work Experience Section",
                    status=ATSCheckStatus.FAILED,
                    message="No work experience section identified in the resume.",
                    fix_tip="Add a dedicated 'Experience' or 'Work Experience' section with organization names, titles, and dates.",
                )
            )
            scores["experience"] = 0.0

        # 5. Skills Visibility
        has_skills_section = "SKILLS" in detected_sections
        skill_count = len(parsed_resume.skills)

        if has_skills_section and skill_count >= 5:
            checks.append(
                ATSCheckItem(
                    code="SKILLS_VISIBILITY",
                    title="Skills Section & Keyword Visibility",
                    status=ATSCheckStatus.PASSED,
                    message=f"Dedicated Skills section detected with {skill_count} recognized skills.",
                    fix_tip="Organize skills into logical groups (e.g. Languages, Frameworks, Tools) to aid recruiter scanning.",
                )
            )
            scores["skills"] = 15.0
        elif skill_count >= 3:
            checks.append(
                ATSCheckItem(
                    code="SKILLS_VISIBILITY",
                    title="Skills Section & Keyword Visibility",
                    status=ATSCheckStatus.WARNING,
                    message=f"{skill_count} skill(s) detected, but a dedicated 'Skills' section header was not clearly isolated.",
                    fix_tip="Create an explicit 'Skills' section near the top or bottom of your resume to maximize ATS keyword scoring.",
                )
            )
            scores["skills"] = 9.0
        else:
            checks.append(
                ATSCheckItem(
                    code="SKILLS_VISIBILITY",
                    title="Skills Section & Keyword Visibility",
                    status=ATSCheckStatus.FAILED,
                    message="Very few technical or domain skills detected.",
                    fix_tip="List relevant technical tools, software, languages, and competencies directly in a dedicated Skills section.",
                )
            )
            scores["skills"] = 2.0

        # 6. Education Section
        if len(parsed_resume.education) >= 1:
            checks.append(
                ATSCheckItem(
                    code="EDUCATION_FORMAT",
                    title="Education Credentials",
                    status=ATSCheckStatus.PASSED,
                    message=f"{len(parsed_resume.education)} education qualification(s) detected.",
                    fix_tip="State your degree level, major, institution name, and graduation year clearly.",
                )
            )
            scores["education"] = 5.0
        else:
            checks.append(
                ATSCheckItem(
                    code="EDUCATION_FORMAT",
                    title="Education Credentials",
                    status=ATSCheckStatus.WARNING,
                    message="No formal education credentials detected in the resume text.",
                    fix_tip="If you hold degrees, diplomas, or certifications, ensure they are placed under an 'Education' header.",
                )
            )
            scores["education"] = 2.0

        # 7. Chronological Formatting & Dates
        if parsed_resume.timeline and parsed_resume.timeline.anomalies:
            checks.append(
                ATSCheckItem(
                    code="DATE_CONSISTENCY",
                    title="Date Consistency",
                    status=ATSCheckStatus.WARNING,
                    message="Potential chronological contradictions or date overlap detected in work history.",
                    fix_tip="Verify that start and end dates follow a logical reverse-chronological order without impossible date sequences.",
                )
            )
            scores["dates"] = 5.0
        else:
            checks.append(
                ATSCheckItem(
                    code="DATE_CONSISTENCY",
                    title="Date Consistency",
                    status=ATSCheckStatus.PASSED,
                    message="Career dates are chronologically coherent with parseable intervals.",
                    fix_tip="Standardize date formats throughout (e.g. '05/2022 – 08/2024' or 'May 2022 – Aug 2024').",
                )
            )
            scores["dates"] = 10.0

        # 8. Special Characters & Complex Formatting Risk
        raw_text = parsed_resume.raw_text or ""
        special_chars = re.findall(r"[^\w\s\.,;:!\?\-\(\)\'\"\/@#\+]", raw_text)
        special_ratio = len(special_chars) / max(len(raw_text), 1)

        if special_ratio > 0.04:
            checks.append(
                ATSCheckItem(
                    code="SPECIAL_CHARACTERS",
                    title="Formatting & Special Characters",
                    status=ATSCheckStatus.WARNING,
                    message="High density of non-standard symbols, emojis, or graphic glyphs detected.",
                    fix_tip="Replace fancy bullet icons, geometric symbols, and star rating bars with standard round bullets (- or •).",
                )
            )
            scores["formatting"] = 2.0
        else:
            checks.append(
                ATSCheckItem(
                    code="SPECIAL_CHARACTERS",
                    title="Formatting & Special Characters",
                    status=ATSCheckStatus.PASSED,
                    message="Clean character distribution without problematic symbol artifacts.",
                    fix_tip="Stick to simple typography and bullet characters for maximum ATS reliability.",
                )
            )
            scores["formatting"] = 5.0

        # 9. Projects Section (Bonus signal)
        if len(parsed_resume.projects) >= 1 or "PROJECTS" in detected_sections:
            checks.append(
                ATSCheckItem(
                    code="PROJECTS_SECTION",
                    title="Projects & Practical Demonstrations",
                    status=ATSCheckStatus.PASSED,
                    message=f"Projects section detected ({len(parsed_resume.projects)} project items identified).",
                    fix_tip="Keep including technologies used and measurable results in your project descriptions.",
                )
            )
            scores["projects"] = 5.0
        else:
            checks.append(
                ATSCheckItem(
                    code="PROJECTS_SECTION",
                    title="Projects & Practical Demonstrations",
                    status=ATSCheckStatus.WARNING,
                    message="No clearly identifiable projects section found.",
                    fix_tip="Adding 2-3 technical or portfolio projects highlights practical skill application even without formal job titles.",
                )
            )
            scores["projects"] = 2.0

        # Total ATS Score Calculation
        total_score = round(min(sum(scores.values()), 100.0), 1)

        passed_count = sum(1 for c in checks if c.status == ATSCheckStatus.PASSED)
        warning_count = sum(1 for c in checks if c.status == ATSCheckStatus.WARNING)
        failed_count = sum(1 for c in checks if c.status == ATSCheckStatus.FAILED)

        if total_score >= 85.0:
            status_label = "High ATS Readiness"
        elif total_score >= 65.0:
            status_label = "Moderate ATS Readiness"
        else:
            status_label = "Needs ATS Optimization"

        format_summary = (
            f"Evaluated {len(checks)} ATS criteria. {passed_count} passed, "
            f"{warning_count} warning(s), and {failed_count} failure(s)."
        )

        warnings = [c.message for c in checks if c.status in [ATSCheckStatus.WARNING, ATSCheckStatus.FAILED]]

        return ATSAnalysis(
            score=total_score,
            status_label=status_label,
            passed_count=passed_count,
            warning_count=warning_count,
            failed_count=failed_count,
            checks=checks,
            format_summary=format_summary,
            warnings=warnings,
        )
