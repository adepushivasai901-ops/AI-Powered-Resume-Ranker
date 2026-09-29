from typing import Any, Dict, List, Optional
import numpy as np

from resume_ranker.extraction.quality_assessor import DocumentQualityReport
from resume_ranker.matching.requirement_matcher import RequirementMatcher
from resume_ranker.models.job_description import ParsedJobDescription, RequirementType
from resume_ranker.models.matching import (
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
        skills_weight: Optional[float] = None,
    ):
        """Initialize candidate scorer with configurable weighting parameters.

        Args:
            similarity_weight: Weight for TF-IDF lexical similarity (default: 0.50).
            required_skills_weight: Weight for required skill coverage (default: 0.30).
            preferred_skills_weight: Weight for preferred skill coverage (default: 0.10).
            experience_weight: Weight for experience/qualification relevance (default: 0.10).
            skills_weight: Legacy parameter alias mapping to required_skills_weight.
        """
        self.similarity_weight = similarity_weight
        if skills_weight is not None:
            self.required_skills_weight = skills_weight
            self.preferred_skills_weight = 0.0
            self.experience_weight = 0.0
        else:
            self.required_skills_weight = required_skills_weight
            self.preferred_skills_weight = preferred_skills_weight
            self.experience_weight = experience_weight

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
        return self.SKILL_SYNONYMS.get(cleaned, cleaned)

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

        # 3. Compute subcomponent percentage scores (0-100)
        if req_skill_matches:
            req_skill_score = (sum(m.score for m in req_skill_matches) / len(req_skill_matches)) * 100.0
        else:
            req_skill_score = 100.0

        if pref_skill_matches:
            pref_skill_score = (sum(m.score for m in pref_skill_matches) / len(pref_skill_matches)) * 100.0
        else:
            pref_skill_score = 0.0

        if exp_matches:
            exp_score = (sum(m.score for m in exp_matches) / len(exp_matches)) * 100.0
        elif parsed_resume.timeline.total_experience_years > 0:
            exp_score = min(100.0, parsed_resume.timeline.total_experience_years * 20.0)
        else:
            exp_score = 50.0

        if edu_matches:
            edu_score = (sum(m.score for m in edu_matches) / len(edu_matches)) * 100.0
        else:
            edu_score = 100.0

        # 4. Weighted total score
        total_w = (
            self.similarity_weight
            + self.required_skills_weight
            + self.preferred_skills_weight
            + self.experience_weight
        )
        weighted_sum = (
            (scaled_sim * self.similarity_weight)
            + (req_skill_score * self.required_skills_weight)
            + (pref_skill_score * self.preferred_skills_weight)
            + (exp_score * self.experience_weight)
        )
        final_score = round(float(np.clip(weighted_sum / total_w if total_w > 0 else 0.0, 0.0, 100.0)), 2)

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

        # 6. Anomaly checks
        anomaly_detector = AnomalyDetector()
        anomalies = anomaly_detector.detect_anomalies(parsed_resume, quality_report)

        # 7. Confidence estimation
        confidence = 1.0
        if quality_report:
            confidence = min(confidence, quality_report.confidence)
        if any(a.severity == AnomalySeverity.HIGH for a in anomalies):
            confidence = min(confidence, 0.60)
        confidence = round(confidence, 2)

        # 8. Human-interpretable explanation
        matched_req_count = len([m for m in req_skill_matches if m.status == RequirementMatchStatus.SATISFIED])
        total_req_count = len(req_skill_matches)
        explanation = (
            f"Final score: {final_score:.2f}/100 (Confidence: {confidence * 100:.0f}%). "
            f"Component breakdown: Required skills {req_skill_score:.1f}% "
            f"({matched_req_count}/{max(1, total_req_count)} verified); "
            f"Preferred skills: {pref_skill_score:.1f}%; "
            f"Career experience: {exp_score:.1f}% ({parsed_resume.timeline.total_experience_years:.1f} continuous years); "
            f"TF-IDF lexical similarity: {scaled_sim:.1f}%; "
            f"Education alignment: {edu_score:.1f}%."
        )
        if anomalies:
            high_anoms = [a.message for a in anomalies if a.severity == AnomalySeverity.HIGH]
            if high_anoms:
                explanation += f" Attention: {'; '.join(high_anoms)}."

        return ScoredCandidateProfile(
            candidate_id=parsed_resume.document.filename,
            filename=parsed_resume.document.filename,
            final_score=final_score,
            similarity_score=round(scaled_sim, 2),
            required_skill_score=round(req_skill_score, 2),
            preferred_skill_score=round(pref_skill_score, 2),
            experience_score=round(exp_score, 2),
            education_score=round(edu_score, 2),
            requirement_matches=match_results,
            matched_required_skills=matched_req,
            missing_required_skills=missing_req,
            matched_preferred_skills=matched_pref,
            missing_preferred_skills=missing_pref,
            anomalies=anomalies,
            total_experience_years=parsed_resume.timeline.total_experience_years,
            confidence=confidence,
            explanation=explanation,
        )

