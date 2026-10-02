"""Candidate Resume Analyzer orchestrator coordinating ATS, content, skills, scoring, and job matching."""

import datetime
from typing import Dict, List, Optional

from resume_ranker.analyzer.ats_assessor import ATSAssessor
from resume_ranker.analyzer.candidate_scorer import CandidateResumeScorer
from resume_ranker.analyzer.content_evaluator import ContentEvaluator
from resume_ranker.analyzer.recommendation_engine import RecommendationEngine
from resume_ranker.extraction.quality_assessor import DocumentQualityReport
from resume_ranker.matching.requirement_matcher import RequirementMatcher
from resume_ranker.models.candidate_analysis import (
    CandidateResumeAnalysis,
    EducationEntry,
    JobMatchAnalysis,
    KeywordMatchDetail,
    SkillCategoryBreakdown,
    SkillRelevanceAnalysis,
)
from resume_ranker.models.evidence import EvidenceType
from resume_ranker.models.job_description import ParsedJobDescription, RequirementType
from resume_ranker.models.matching import RequirementMatchStatus
from resume_ranker.models.resume import ParsedResume
from resume_ranker.nlp.jd_parser import JDParser
from resume_ranker.nlp.preprocessor import TextPreprocessor
from resume_ranker.nlp.skill_extractor import SkillIntelligence
from resume_ranker.scoring.anomaly_detector import AnomalyDetector
from resume_ranker.vectorization.tfidf_matcher import TFIDFMatcher


class CandidateResumeAnalyzer:
    """End-to-end analyzer transforming parsed resume data into candidate-centric feedback and actionable insights."""

    def __init__(self):
        self.ats_assessor = ATSAssessor()
        self.content_evaluator = ContentEvaluator()
        self.health_scorer = CandidateResumeScorer()
        self.recommendation_engine = RecommendationEngine()
        self.skill_intel = SkillIntelligence()
        self.anomaly_detector = AnomalyDetector()

    def analyze(
        self,
        parsed_resume: ParsedResume,
        quality_report: Optional[DocumentQualityReport] = None,
        job_description: Optional[str] = None,
        parsed_jd: Optional[ParsedJobDescription] = None,
    ) -> CandidateResumeAnalysis:
        """Run complete candidate-centric analysis workflow.

        Args:
            parsed_resume: Structured parsed resume document.
            quality_report: Document parsing quality and character assessment.
            job_description: Optional job description text for job alignment.
            parsed_jd: Optional pre-parsed job description.

        Returns:
            CandidateResumeAnalysis with comprehensive scores, section analysis, ATS checks, and recommendations.
        """
        # 1. ATS Readiness Assessment
        ats_analysis = self.ats_assessor.evaluate_ats_readiness(
            parsed_resume=parsed_resume,
            quality_report=quality_report,
        )

        # 2. Content & Section Evaluations
        exp_quality = self.content_evaluator.evaluate_experience(parsed_resume)
        summary_quality = self.content_evaluator.evaluate_summary(parsed_resume)
        project_quality = self.content_evaluator.evaluate_projects(parsed_resume)
        sections = self.content_evaluator.evaluate_all_sections(
            parsed_resume=parsed_resume,
            exp_quality=exp_quality,
            summary_quality=summary_quality,
            proj_quality=project_quality,
        )

        # 3. Skill Relevance & Categorization
        skill_analysis = self._evaluate_skills(parsed_resume)

        # 4. Optional Job Description Matching
        job_match = self._evaluate_job_match(
            parsed_resume=parsed_resume,
            job_description=job_description,
            parsed_jd=parsed_jd,
        )

        # 5. Composite Resume Health Score (Dynamic Weight Renormalization)
        health_score = self.health_scorer.compute_health_score(
            parsed_resume=parsed_resume,
            ats_analysis=ats_analysis,
            exp_quality=exp_quality,
            summary_quality=summary_quality,
            project_quality=project_quality,
            skill_analysis=skill_analysis,
        )

        # 6. Integrity & Anomaly Checks
        anomalies = self.anomaly_detector.detect_anomalies(parsed_resume, quality_report)

        # 7. Actionable Recommendations
        recommendations = self.recommendation_engine.generate_recommendations(
            parsed_resume=parsed_resume,
            ats_analysis=ats_analysis,
            exp_quality=exp_quality,
            summary_quality=summary_quality,
            project_quality=project_quality,
            skill_analysis=skill_analysis,
            job_match=job_match,
            anomalies=anomalies,
        )

        # 8. Warnings aggregation
        warnings: List[str] = []
        if anomalies:
            warnings.extend([a.message for a in anomalies])
        if not parsed_resume.candidate.email or not parsed_resume.candidate.phone:
            warnings.append("Missing or unreadable contact information in header.")
        if exp_quality.weak_bullet_count >= 2:
            warnings.append(f"{exp_quality.weak_bullet_count} experience bullets begin with passive phrasing ('Worked on', 'Responsible for').")
        if exp_quality.measurable_achievement_count == 0 and exp_quality.total_bullets > 0:
            warnings.append("No quantifiable metrics or measurable achievements detected in experience bullets.")
        if not job_match.is_available:
            warnings.append("Job description match unavailable because no job description was provided.")

        # 9. Format structured Education Items
        edu_entries: List[EducationEntry] = []
        for edu in parsed_resume.education:
            date_str = None
            if edu.start_date or edu.end_date:
                date_str = f"{edu.start_date or ''} – {edu.end_date or 'Completed'}".strip(" –")
            edu_entries.append(
                EducationEntry(
                    institution=edu.institution or "University / College",
                    degree=edu.degree or "Degree / Certificate",
                    field_of_study=edu.field_of_study,
                    dates=date_str,
                )
            )

        # Candidate name resolution
        cand_name = "Candidate"
        if parsed_resume.candidate and parsed_resume.candidate.name:
            if parsed_resume.candidate.name not in ["Unknown Candidate", "curriculum vitae", "resume"]:
                cand_name = parsed_resume.candidate.name
        if cand_name == "Candidate" and parsed_resume.document.filename:
            clean_fname = parsed_resume.document.filename.replace(".pdf", "").replace("_", " ").title()
            cand_name = clean_fname

        total_years = parsed_resume.timeline.total_experience_years if parsed_resume.timeline else 0.0
        duration_display = f"{total_years:.1f} years" if total_years > 0 else "Duration not determined"

        return CandidateResumeAnalysis(
            document_id=parsed_resume.document.id,
            filename=parsed_resume.document.filename,
            candidate_name=cand_name,
            email=parsed_resume.candidate.email,
            phone=parsed_resume.candidate.phone,
            location=parsed_resume.candidate.location,
            links=parsed_resume.candidate.links,
            raw_text=parsed_resume.raw_text,
            health_score=health_score,
            ats_readiness=ats_analysis,
            sections=sections,
            experience_analysis=exp_quality,
            summary_analysis=summary_quality,
            project_analysis=project_quality,
            skills_analysis=skill_analysis,
            job_match=job_match,
            recommendations=recommendations,
            warnings=warnings,
            education_items=edu_entries,
            total_experience_years=total_years,
            total_experience_display=duration_display,
            created_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _evaluate_skills(self, parsed_resume: ParsedResume) -> SkillRelevanceAnalysis:
        """Categorize skills, detect domain signals, and calculate evidence certainty."""
        skills = parsed_resume.skills
        category_map: Dict[str, List[str]] = {}

        # Canonical category names for display
        display_names = {
            "programming_languages": "Programming Languages",
            "frameworks_and_libraries": "Frameworks & Libraries",
            "data_science_and_ai": "AI, ML & Data Science",
            "databases_and_data_engineering": "Databases & Data Engineering",
            "cloud_and_devops": "Cloud & DevOps",
            "tools_and_methods": "Tools & Methodologies",
            "design_and_creative": "Design & UI/UX",
            "product_and_management": "Product & Management",
        }

        evidenced_skills = []
        claimed_skills = []

        for s in skills:
            cat = self.skill_intel.get_skill_category(s.canonical_name) or "other"
            category_map.setdefault(cat, []).append(s.canonical_name.title())

            if s.evidence_type in [EvidenceType.RESUME_EVIDENCED, EvidenceType.VERIFIED]:
                evidenced_skills.append(s.canonical_name.title())
            else:
                claimed_skills.append(s.canonical_name.title())

        categories: List[SkillCategoryBreakdown] = []
        for cat_key, skill_list in category_map.items():
            nice_name = display_names.get(cat_key, cat_key.replace("_", " ").title())
            categories.append(
                SkillCategoryBreakdown(
                    category_key=cat_key,
                    category_name=nice_name,
                    skills=sorted(list(set(skill_list))),
                )
            )

        total_skills = len(skills)
        evidenced_ratio = len(evidenced_skills) / max(total_skills, 1)

        # Domain signals
        domain_signals: List[str] = []
        if "data_science_and_ai" in category_map:
            domain_signals.append("Data Science & AI")
        if "programming_languages" in category_map:
            domain_signals.append("Software Engineering")
        if "cloud_and_devops" in category_map:
            domain_signals.append("Cloud Infrastructure")
        if "design_and_creative" in category_map:
            domain_signals.append("Product Design / UI-UX")
        if "databases_and_data_engineering" in category_map:
            domain_signals.append("Database Management")

        # Scoring
        base_score = min(total_skills * 10.0, 70.0)
        evidence_bonus = evidenced_ratio * 30.0
        final_skill_score = round(min(base_score + evidence_bonus, 100.0), 1)

        recommendations: List[str] = []
        if total_skills < 5:
            recommendations.append("Include more explicit tools and frameworks in your Skills section.")
        if evidenced_ratio < 0.40 and total_skills >= 4:
            recommendations.append("Integrate key skills directly into employment and project descriptions to prove hands-on application.")

        return SkillRelevanceAnalysis(
            score=final_skill_score,
            total_detected=total_skills,
            categories=categories,
            evidenced_skills=evidenced_skills,
            claimed_skills=claimed_skills,
            evidenced_ratio=round(evidenced_ratio, 2),
            domain_signals=domain_signals,
            recommendations=recommendations,
        )

    def _evaluate_job_match(
        self,
        parsed_resume: ParsedResume,
        job_description: Optional[str] = None,
        parsed_jd: Optional[ParsedJobDescription] = None,
    ) -> JobMatchAnalysis:
        """Perform optional requirement matching against target job description."""
        if not job_description and not parsed_jd:
            return JobMatchAnalysis(is_available=False)

        if not parsed_jd and job_description:
            jd_parser = JDParser()
            parsed_jd = jd_parser.parse_job_description(job_description)

        if not parsed_jd:
            return JobMatchAnalysis(is_available=False)

        # Match requirements using existing RequirementMatcher
        matcher = RequirementMatcher()
        match_results = [matcher.match_requirement(req, parsed_resume) for req in parsed_jd.requirements]

        req_matches = [m for m in match_results if m.req_type == RequirementType.MUST_HAVE_SKILL]
        pref_matches = [m for m in match_results if m.req_type == RequirementType.PREFERRED_SKILL]
        exp_matches = [m for m in match_results if m.req_type == RequirementType.MINIMUM_EXPERIENCE]
        edu_matches = [m for m in match_results if m.req_type == RequirementType.EDUCATION]

        matched_req = [
            m.canonical_name.title() for m in req_matches
            if m.status in [RequirementMatchStatus.SATISFIED, RequirementMatchStatus.PARTIALLY_SATISFIED]
        ]
        missing_req = [
            m.canonical_name.title() for m in req_matches
            if m.status in [RequirementMatchStatus.NOT_FOUND, RequirementMatchStatus.INSUFFICIENT_EVIDENCE]
        ]

        matched_pref = [
            m.canonical_name.title() for m in pref_matches
            if m.status in [RequirementMatchStatus.SATISFIED, RequirementMatchStatus.PARTIALLY_SATISFIED]
        ]
        missing_pref = [
            m.canonical_name.title() for m in pref_matches
            if m.status in [RequirementMatchStatus.NOT_FOUND, RequirementMatchStatus.INSUFFICIENT_EVIDENCE]
        ]

        # Experience check
        exp_meets = True
        exp_note = "Meets or aligns with experience expectations."
        if exp_matches:
            top_exp = exp_matches[0]
            if top_exp.status in [RequirementMatchStatus.NOT_FOUND, RequirementMatchStatus.INSUFFICIENT_EVIDENCE]:
                exp_meets = False
                exp_note = f"Job requests {top_exp.required_duration_months // 12 if top_exp.required_duration_months else 'specified'}+ years experience."
            else:
                exp_meets = True
                exp_note = "Demonstrated experience meets stated requirement."

        # Education check
        edu_meets = True
        edu_note = "Education aligns with role requirements."
        if edu_matches:
            top_edu = edu_matches[0]
            if top_edu.status in [RequirementMatchStatus.NOT_FOUND, RequirementMatchStatus.INSUFFICIENT_EVIDENCE]:
                edu_meets = False
                edu_note = "Job requests specific academic degree (e.g. Bachelor's / Master's)."
            else:
                edu_meets = True
                edu_note = "Academic background meets role requirements."

        # TF-IDF lexical overlap
        raw_jd_text = parsed_jd.raw_text or job_description or ""
        tfidf_matcher = TFIDFMatcher()
        proc = TextPreprocessor()
        sim_score = tfidf_matcher.compute_similarity(
            proc.preprocess(parsed_resume.raw_text),
            proc.preprocess(raw_jd_text),
        )

        # Keyword Coverage Details
        keywords: List[KeywordMatchDetail] = []
        for m in req_matches:
            is_found = m.status in [RequirementMatchStatus.SATISFIED, RequirementMatchStatus.PARTIALLY_SATISFIED]
            evidence_sec = m.evidence_refs[0].section if m.evidence_refs else None
            tip = (
                "Verified in resume."
                if is_found
                else f"Consider adding {m.canonical_name.title()} only if you have genuine practical experience."
            )
            keywords.append(
                KeywordMatchDetail(
                    keyword=m.canonical_name.title(),
                    category="Required Skills",
                    detected=is_found,
                    importance="Required",
                    evidence_section=evidence_sec,
                    context_tip=tip,
                )
            )

        for m in pref_matches:
            is_found = m.status in [RequirementMatchStatus.SATISFIED, RequirementMatchStatus.PARTIALLY_SATISFIED]
            evidence_sec = m.evidence_refs[0].section if m.evidence_refs else None
            tip = (
                "Verified in resume."
                if is_found
                else f"Preferred skill. Mention {m.canonical_name.title()} if you have relevant project experience."
            )
            keywords.append(
                KeywordMatchDetail(
                    keyword=m.canonical_name.title(),
                    category="Preferred Skills",
                    detected=is_found,
                    importance="Preferred",
                    evidence_section=evidence_sec,
                    context_tip=tip,
                )
            )

        total_kw = len(keywords)
        matched_kw = sum(1 for k in keywords if k.detected)
        kw_coverage_pct = round((matched_kw / total_kw * 100.0), 1) if total_kw > 0 else round(sim_score * 100.0, 1)

        # Weighted Job Match Score
        # Required Skills: 50%, Preferred: 20%, Similarity: 20%, Experience: 10%
        req_score = (len(matched_req) / len(req_matches) * 100.0) if req_matches else 80.0
        pref_score = (len(matched_pref) / len(pref_matches) * 100.0) if pref_matches else 70.0
        sim_val = round(sim_score * 100.0, 1)
        exp_val = 100.0 if exp_meets else 40.0

        match_score = round(
            (req_score * 0.50) + (pref_score * 0.20) + (sim_val * 0.20) + (exp_val * 0.10),
            1,
        )

        tailoring_recs: List[str] = []
        if missing_req:
            tailoring_recs.append(
                f"Highlight required skills ({', '.join(missing_req[:3])}) if you have authentic experience with them."
            )
        if missing_pref:
            tailoring_recs.append(
                f"Add preferred competencies ({', '.join(missing_pref[:3])}) to elevate your profile above other applicants."
            )
        if not exp_meets:
            tailoring_recs.append("Clarify career duration and relevant project timelines to satisfy stated tenure requirements.")

        target_role = "Target Position"
        for line in raw_jd_text.splitlines():
            clean_l = line.strip()
            if clean_l and len(clean_l) < 50:
                target_role = clean_l
                break

        return JobMatchAnalysis(
            is_available=True,
            match_score=min(match_score, 100.0),
            target_role=target_role,
            required_skills_matched=matched_req,
            required_skills_missing=missing_req,
            preferred_skills_matched=matched_pref,
            preferred_skills_missing=missing_pref,
            experience_meets=exp_meets,
            experience_note=exp_note,
            education_meets=edu_meets,
            education_note=edu_note,
            keyword_coverage_pct=kw_coverage_pct,
            keywords=keywords,
            tailoring_recommendations=tailoring_recs,
        )
