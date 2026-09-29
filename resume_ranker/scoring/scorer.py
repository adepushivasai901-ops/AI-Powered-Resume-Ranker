from typing import Any, Dict, List, Optional
import numpy as np

from resume_ranker.extraction.quality_assessor import DocumentQualityReport
from resume_ranker.matching.requirement_matcher import RequirementMatcher
from resume_ranker.models.evidence import EvidenceType
from resume_ranker.models.job_description import ParsedJobDescription, RequirementType
from resume_ranker.models.matching import (
    AnomalyFlag,
    AnomalySeverity,
    RequirementMatchStatus,
    ScoredCandidateProfile,
)
from resume_ranker.models.resume import ParsedResume
from resume_ranker.scoring.anomaly_detector import AnomalyDetector


class CandidateScorer:
    """Evaluates candidate relevance scores based on configurable weighted components."""

    # Common technology skill synonyms and formatting variations
    SKILL_SYNONYMS: Dict[str, str] = {
        "react.js": "react",
        "reactjs": "react",
        "node.js": "node.js",
        "nodejs": "node.js",
        "vue.js": "vue",
        "vuejs": "vue",
        "postgres": "postgresql",
        "postgresql": "postgresql",
        "scikit-learn": "scikit-learn",
        "sklearn": "scikit-learn",
        "k8s": "kubernetes",
        "aws": "aws",
        "amazon web services": "aws",
        "gcp": "gcp",
        "google cloud platform": "gcp",
        "ml": "machine learning",
        "nlp": "nlp",
        "ai": "ai",
        "ci/cd": "ci/cd",
        "ci-cd": "ci/cd",
        "rest api": "rest apis",
        "restful": "rest apis",
        "rest apis": "rest apis",
    }

    def __init__(
        self,
        similarity_weight: float = 0.50,
        required_skills_weight: float = 0.30,
        preferred_skills_weight: float = 0.10,
        experience_weight: float = 0.10,
        education_weight: float = 0.10,
        skills_weight: Optional[float] = None,
    ):
        """Initialize candidate scorer with configurable weighting parameters.

        Args:
            similarity_weight: Weight for TF-IDF lexical similarity (default: 0.50).
            required_skills_weight: Weight for required skill coverage (default: 0.30).
            preferred_skills_weight: Weight for preferred skill coverage (default: 0.10).
            experience_weight: Weight for experience/qualification relevance (default: 0.10).
            education_weight: Weight for education qualification relevance (default: 0.10).
            skills_weight: Legacy parameter alias mapping to required_skills_weight.
        """
        self.similarity_weight = similarity_weight
        if skills_weight is not None:
            self.required_skills_weight = skills_weight
            self.preferred_skills_weight = 0.0
            self.experience_weight = 0.0
            self.education_weight = 0.0
        else:
            self.required_skills_weight = required_skills_weight
            self.preferred_skills_weight = preferred_skills_weight
            self.experience_weight = experience_weight
            self.education_weight = education_weight

    def normalize_skill(self, skill: str) -> str:
        """Normalize a skill name for robust, case-insensitive comparison.

        Preserves special technical names like C++, C#, .NET, SQL, AWS, React, Python.

        Args:
            skill: Raw skill string.

        Returns:
            Normalized skill name.
        """
        if not skill or not isinstance(skill, str):
            return ""
        cleaned = skill.strip().lower()
        from resume_ranker.nlp.skill_extractor import SkillIntelligence
        normalized = SkillIntelligence().normalize_skill_name(cleaned)
        return self.SKILL_SYNONYMS.get(normalized, normalized)

    def score_candidate(
        self,
        candidate_id: str,
        similarity_score: float,
        matched_skills: Optional[List[str]] = None,
        missing_skills: Optional[List[str]] = None,
        matched_preferred_skills: Optional[List[str]] = None,
        missing_preferred_skills: Optional[List[str]] = None,
        experience_score: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compute deterministic relevance score profile and explanation for a candidate.

        Args:
            candidate_id: Unique candidate or resume identifier.
            similarity_score: TF-IDF lexical similarity score (0.0 to 1.0 or 0 to 100).
            matched_skills: List of matched required skills.
            missing_skills: List of missing required skills.
            matched_preferred_skills: List of matched preferred skills.
            missing_preferred_skills: List of missing preferred skills.
            experience_score: Component score for experience relevance (0 to 100).
            **kwargs: Additional metadata parameters.

        Returns:
            Dictionary containing final score, component breakdown, and structured explanation.
        """
        matched_req = matched_skills if matched_skills is not None else []
        missing_req = missing_skills if missing_skills is not None else []
        matched_pref = matched_preferred_skills if matched_preferred_skills is not None else []
        missing_pref = missing_preferred_skills if missing_preferred_skills is not None else []

        # Scale similarity score to 0-100 range if provided in [0.0, 1.0]
        if similarity_score <= 1.0:
            scaled_sim = float(similarity_score * 100.0)
        else:
            scaled_sim = float(similarity_score)
        scaled_sim = float(np.clip(scaled_sim, 0.0, 100.0))

        # Calculate required skill coverage (missing skills negatively affect score)
        total_req = len(matched_req) + len(missing_req)
        if total_req > 0:
            required_skill_score = (len(matched_req) / total_req) * 100.0
        else:
            required_skill_score = 100.0

        # Calculate preferred skill coverage
        total_pref = len(matched_pref) + len(missing_pref)
        if total_pref > 0:
            preferred_skill_score = (len(matched_pref) / total_pref) * 100.0
        else:
            preferred_skill_score = 0.0

        # Normalize experience score to 0-100
        if experience_score <= 1.0 and experience_score > 0.0:
            exp_score = float(experience_score * 100.0)
        else:
            exp_score = float(experience_score)
        exp_score = float(np.clip(exp_score, 0.0, 100.0))

        # Compute weighted sum
        total_weight = (
            self.similarity_weight
            + self.required_skills_weight
            + self.preferred_skills_weight
            + self.experience_weight
        )

        weighted_total = (
            (scaled_sim * self.similarity_weight)
            + (required_skill_score * self.required_skills_weight)
            + (preferred_skill_score * self.preferred_skills_weight)
            + (exp_score * self.experience_weight)
        )

        if total_weight > 0:
            raw_final = weighted_total / total_weight
        else:
            raw_final = 0.0

        final_score = round(float(np.clip(raw_final, 0.0, 100.0)), 2)

        # Generate structured explanation
        explanation = (
            f"Final score: {final_score:.2f}/100. "
            f"Component breakdown: TF-IDF lexical similarity ({scaled_sim:.1f}%) "
            f"weighted at {self.similarity_weight * 100:.0f}%, "
            f"required skills ({required_skill_score:.1f}% "
            f"[{len(matched_req)}/{total_req} matched]) "
            f"weighted at {self.required_skills_weight * 100:.0f}%, "
            f"preferred skills ({preferred_skill_score:.1f}% "
            f"[{len(matched_pref)}/{max(1, total_pref)} matched]) "
            f"weighted at {self.preferred_skills_weight * 100:.0f}%, "
            f"and experience relevance ({exp_score:.1f}%) "
            f"weighted at {self.experience_weight * 100:.0f}%."
        )

        return {
            "candidate_id": str(candidate_id),
            "final_score": final_score,
            "similarity_score": round(scaled_sim, 2),
            "required_skill_score": round(required_skill_score, 2),
            "preferred_skill_score": round(preferred_skill_score, 2),
            "experience_score": round(exp_score, 2),
            "matched_required_skills": list(matched_req),
            "missing_required_skills": list(missing_req),
            "matched_preferred_skills": list(matched_pref),
            "missing_preferred_skills": list(missing_pref),
            "explanation": explanation,
        }

    def score_batch(self, candidate_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch score multiple candidates.

        Args:
            candidate_data: List of candidate feature dictionaries.

        Returns:
            List of scored candidate profile dictionaries.
        """
        if not candidate_data:
            return []
        return [self.score_candidate(**item) for item in candidate_data]

    def score_candidate_profile(
        self,
        parsed_resume: ParsedResume,
        parsed_jd: ParsedJobDescription,
        similarity_score: float,
        quality_report: Optional[DocumentQualityReport] = None,
    ) -> ScoredCandidateProfile:
        """Evaluate candidate resume against structured job description using evidence-linked composite scoring.

        Args:
            parsed_resume: Fully structured canonical parsed resume.
            parsed_jd: Structured target job description.
            similarity_score: Lexical/semantic similarity score (0.0 to 1.0 or 0 to 100).
            quality_report: Optional extraction quality assessment metrics.

        Returns:
            ScoredCandidateProfile with explainable subscores, evidence links, and anomaly flags.
        """
        # 1. Match all JD requirements against resume
        matcher = RequirementMatcher()
        match_results = matcher.match_all_requirements(parsed_jd.requirements, parsed_resume)

        # Scale similarity score to 0-100 range
        if similarity_score <= 1.0:
            scaled_sim = float(similarity_score * 100.0)
        else:
            scaled_sim = float(similarity_score)
        scaled_sim = float(np.clip(scaled_sim, 0.0, 100.0))

        # 2. Partition matches by requirement type
        req_skill_matches = [m for m in match_results if m.req_type == RequirementType.MUST_HAVE_SKILL]
        pref_skill_matches = [m for m in match_results if m.req_type == RequirementType.PREFERRED_SKILL]
        exp_matches = [m for m in match_results if m.req_type == RequirementType.MINIMUM_EXPERIENCE]
        edu_matches = [m for m in match_results if m.req_type == RequirementType.EDUCATION]

        # 3. Dynamic Renormalization & Subcomponent Scoring
        active_weights: Dict[str, float] = {}
        active_scores: Dict[str, float] = {}

        # Component: Similarity (always active)
        active_weights["similarity"] = self.similarity_weight
        active_scores["similarity"] = scaled_sim

        # Component: Required Skills
        if req_skill_matches:
            req_skill_score: Optional[float] = (
                sum(m.score for m in req_skill_matches) / len(req_skill_matches)
            ) * 100.0
            active_weights["required_skills"] = self.required_skills_weight
            active_scores["required_skills"] = req_skill_score
        else:
            req_skill_score = None

        # Component: Preferred Skills
        if pref_skill_matches:
            pref_skill_score: Optional[float] = (
                sum(m.score for m in pref_skill_matches) / len(pref_skill_matches)
            ) * 100.0
            active_weights["preferred_skills"] = self.preferred_skills_weight
            active_scores["preferred_skills"] = pref_skill_score
        else:
            pref_skill_score = None

        # Component: Career Experience
        if exp_matches:
            exp_score: Optional[float] = (
                sum(m.score for m in exp_matches) / len(exp_matches)
            ) * 100.0
            active_weights["experience"] = self.experience_weight
            active_scores["experience"] = exp_score
        elif parsed_resume.timeline.total_experience_years > 0:
            exp_score = min(100.0, parsed_resume.timeline.total_experience_years * 20.0)
            active_weights["experience"] = self.experience_weight
            active_scores["experience"] = exp_score
        else:
            exp_score = None

        # Component: Education
        if edu_matches:
            edu_score: Optional[float] = (
                sum(m.score for m in edu_matches) / len(edu_matches)
            ) * 100.0
            active_weights["education"] = self.education_weight
            active_scores["education"] = edu_score
        else:
            edu_score = None

        # 4. Dynamic Renormalization: Weighted Composite Calculation
        total_active_w = sum(active_weights.values())
        if total_active_w > 0:
            weighted_sum = sum(active_scores[k] * active_weights[k] for k in active_weights)
            final_score = round(float(np.clip(weighted_sum / total_active_w, 0.0, 100.0)), 2)
        else:
            final_score = 0.0

        # 5. Extract matched and missing skill lists
        matched_req = [
            m.canonical_name
            for m in req_skill_matches
            if m.status in [RequirementMatchStatus.SATISFIED, RequirementMatchStatus.PARTIALLY_SATISFIED]
        ]
        missing_req = [
            m.canonical_name
            for m in req_skill_matches
            if m.status in [RequirementMatchStatus.NOT_FOUND, RequirementMatchStatus.INSUFFICIENT_EVIDENCE]
        ]

        matched_pref = [
            m.canonical_name
            for m in pref_skill_matches
            if m.status in [RequirementMatchStatus.SATISFIED, RequirementMatchStatus.PARTIALLY_SATISFIED]
        ]
        missing_pref = [
            m.canonical_name
            for m in pref_skill_matches
            if m.status in [RequirementMatchStatus.NOT_FOUND, RequirementMatchStatus.INSUFFICIENT_EVIDENCE]
        ]

        # 6. Extract evidenced vs claimed skills distinction
        evidenced_skills = [
            s.canonical_name for s in parsed_resume.skills
            if s.evidence_type in [EvidenceType.RESUME_EVIDENCED, EvidenceType.VERIFIED]
        ]
        claimed_skills = [
            s.canonical_name for s in parsed_resume.skills
            if s.evidence_type in [EvidenceType.CLAIMED, EvidenceType.EXPLICIT, EvidenceType.INFERRED]
        ]

        # 7. Anomaly checks
        anomaly_detector = AnomalyDetector()
        anomalies = anomaly_detector.detect_anomalies(parsed_resume, quality_report)

        # 8. Analysis Confidence estimation
        confidence = self.compute_analysis_confidence(
            parsed_resume=parsed_resume,
            quality_report=quality_report,
            anomalies=anomalies,
        )

        # 9. Human-interpretable explanation
        active_breakdown_parts = []
        for name, w in active_weights.items():
            eff_pct = (w / total_active_w) * 100.0 if total_active_w > 0 else 0.0
            score_val = active_scores.get(name, 0.0)
            label = {
                "similarity": "TF-IDF lexical similarity",
                "required_skills": "Required skills",
                "preferred_skills": "Preferred skills",
                "experience": "Career experience",
                "education": "Education alignment",
            }.get(name, name)
            active_breakdown_parts.append(f"{label}: {score_val:.1f}% (weight: {eff_pct:.0f}%)")

        missing_noted = []
        if exp_score is None:
            missing_noted.append("Experience: Not found (excluded from scoring)")
        if edu_score is None:
            missing_noted.append("Education: Not specified in JD (excluded from scoring)")

        breakdown_text = "; ".join(active_breakdown_parts)
        if missing_noted:
            breakdown_text += f". Notes: {'; '.join(missing_noted)}"

        explanation = (
            f"Final score: {final_score:.2f}/100 (Analysis Confidence: {confidence * 100:.0f}%). "
            f"Active components: {breakdown_text}."
        )
        if anomalies:
            high_anoms = [a.message for a in anomalies if a.severity in [AnomalySeverity.HIGH, AnomalySeverity.MEDIUM]]
            if high_anoms:
                explanation += f" Attention: {'; '.join(high_anoms)}."

        cand_name = (
            parsed_resume.candidate.name
            if parsed_resume.candidate and parsed_resume.candidate.name and parsed_resume.candidate.name != "Unknown Candidate"
            else parsed_resume.document.filename
        )
        edu_summary = (
            ", ".join(
                f"{e.degree or 'Degree'}{(' in ' + e.field_of_study) if e.field_of_study else ''}"
                for e in parsed_resume.education
            )
            if parsed_resume.education
            else "Not found"
        )

        return ScoredCandidateProfile(
            candidate_id=parsed_resume.document.filename,
            candidate_name=cand_name,
            filename=parsed_resume.document.filename,
            final_score=final_score,
            similarity_score=round(scaled_sim, 2),
            required_skill_score=round(req_skill_score, 2) if req_skill_score is not None else None,
            preferred_skill_score=round(pref_skill_score, 2) if pref_skill_score is not None else None,
            experience_score=round(exp_score, 2) if exp_score is not None else None,
            education_score=round(edu_score, 2) if edu_score is not None else None,
            education_summary=edu_summary,
            requirement_matches=match_results,
            matched_required_skills=matched_req,
            missing_required_skills=missing_req,
            matched_preferred_skills=matched_pref,
            missing_preferred_skills=missing_pref,
            evidenced_skills=evidenced_skills,
            claimed_skills=claimed_skills,
            anomalies=anomalies,
            total_experience_years=parsed_resume.timeline.total_experience_years,
            confidence=confidence,
            explanation=explanation,
        )

    def compute_analysis_confidence(
        self,
        parsed_resume: ParsedResume,
        quality_report: Optional[DocumentQualityReport] = None,
        anomalies: Optional[List[AnomalyFlag]] = None,
    ) -> float:
        """Compute realistic, evidence-grounded Analysis Confidence (capped at 0.88 for self-reported resumes).

        Factors:
        1. Document Quality & Text Density (0.35)
        2. Section Completeness (0.35)
        3. Evidence Verification Quality (0.30)
        4. Deductions for detected anomalies
        """
        # 1. Document Quality & Text Density Factor
        if quality_report:
            words = getattr(quality_report, "total_words", getattr(quality_report, "word_count", 0))
            if quality_report.is_scanned_image:
                doc_quality_factor = 0.25
            elif words < 100:
                doc_quality_factor = 0.45
            elif words < 250:
                doc_quality_factor = 0.65
            else:
                doc_quality_factor = min(1.0, quality_report.confidence)
            doc_quality_factor -= min(0.30, quality_report.suspicious_chars_ratio * 3.0)
            doc_quality_factor = max(0.15, doc_quality_factor)
        else:
            word_count = len(parsed_resume.raw_text.split())
            if word_count < 100:
                doc_quality_factor = 0.45
            elif word_count < 250:
                doc_quality_factor = 0.70
            else:
                doc_quality_factor = 0.90

        # 2. Section Structure Completeness Factor
        sections_found = set(parsed_resume.sections_detected.keys())
        struct_score = 0.20  # baseline
        if "EXPERIENCE" in sections_found:
            struct_score += 0.30
        if "SKILLS" in sections_found:
            struct_score += 0.25
        if "EDUCATION" in sections_found:
            struct_score += 0.15
        if "HEADER" in sections_found or "SUMMARY" in sections_found:
            struct_score += 0.10
        struct_factor = min(1.0, struct_score)

        # 3. Contextual Evidence Quality Factor
        total_skills = len(parsed_resume.skills)
        if total_skills > 0:
            evidenced_count = sum(
                1 for s in parsed_resume.skills
                if s.evidence_type in [EvidenceType.RESUME_EVIDENCED, EvidenceType.VERIFIED]
            )
            evidenced_ratio = evidenced_count / total_skills
            evidence_factor = 0.60 + (0.35 * evidenced_ratio)
        else:
            evidence_factor = 0.50

        # Weighted combination
        raw_conf = (
            (0.35 * doc_quality_factor)
            + (0.35 * struct_factor)
            + (0.30 * evidence_factor)
        )

        # 4. Anomaly Deductions
        if anomalies:
            for a in anomalies:
                if a.severity == AnomalySeverity.HIGH:
                    raw_conf -= 0.30
                elif a.severity == AnomalySeverity.MEDIUM:
                    raw_conf -= 0.12
                elif a.severity == AnomalySeverity.LOW:
                    raw_conf -= 0.04
            if any(a.severity == AnomalySeverity.HIGH for a in anomalies):
                raw_conf = min(raw_conf, 0.60)

        # Maximum ceiling: 0.88 for self-reported resumes; floor: 0.20
        analysis_confidence = round(float(np.clip(raw_conf, 0.20, 0.88)), 2)
        return analysis_confidence

