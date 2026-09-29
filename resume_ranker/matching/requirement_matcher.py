"""Requirement-level matching engine: evaluating candidate evidence against structured JD requirements."""

from typing import Dict, List, Optional

from resume_ranker.models.evidence import EvidenceReference, EvidenceType
from resume_ranker.models.job_description import JDRequirement, RequirementType
from resume_ranker.models.matching import RequirementMatchResult, RequirementMatchStatus
from resume_ranker.models.resume import ParsedResume, SkillItem


class RequirementMatcher:
    """Evaluates granular JD requirements against candidate resume evidence blocks."""

    def match_requirement(
        self, requirement: JDRequirement, resume: ParsedResume
    ) -> RequirementMatchResult:
        """Evaluate a single requirement against a parsed candidate resume.

        Pipeline:
        JD Requirement -> Candidate Evidence Retrieval -> Evidence Classification -> Duration/Recency Check -> Status
        """
        # 1. Skill requirement matching
        if requirement.req_type in [RequirementType.MUST_HAVE_SKILL, RequirementType.PREFERRED_SKILL]:
            return self._match_skill_requirement(requirement, resume)

        # 2. Minimum career experience requirement matching
        elif requirement.req_type == RequirementType.MINIMUM_EXPERIENCE:
            return self._match_experience_requirement(requirement, resume)

        # 3. Education requirement matching
        elif requirement.req_type == RequirementType.EDUCATION:
            return self._match_education_requirement(requirement, resume)

        # 4. Fallback general requirement matching
        return self._match_generic_requirement(requirement, resume)

    def _match_skill_requirement(
        self, requirement: JDRequirement, resume: ParsedResume
    ) -> RequirementMatchResult:
        """Match skill requirement against candidate's extracted canonical skills and sections."""
        canonical_target = requirement.canonical_name.lower().strip()

        # Find matching skill item in candidate profile
        skill_item: Optional[SkillItem] = None
        for s in resume.skills:
            if s.canonical_name.lower() == canonical_target:
                skill_item = s
                break

        if not skill_item:
            # Check if skill token appears in raw text or any section
            return RequirementMatchResult(
                requirement_id=requirement.id,
                canonical_name=requirement.canonical_name,
                req_type=requirement.req_type,
                status=RequirementMatchStatus.NOT_FOUND,
                score=0.0,
                confidence=1.0,
                explanation=f"Required skill '{requirement.canonical_name}' was not detected in candidate profile.",
            )

        # Evidence exists: classify by strength and duration
        evidence_type = skill_item.evidence_type
        evidence_refs = list(skill_item.evidence_refs)
        sections_found = {ref.section for ref in evidence_refs if ref.section}

        # Check required duration if specified
        min_months = requirement.min_experience_months

        if evidence_type == EvidenceType.VERIFIED:
            status = RequirementMatchStatus.SATISFIED
            score = 1.0
            explanation = (
                f"Skill '{requirement.canonical_name}' is verified with practical experience in "
                f"{', '.join(sections_found) or 'experience records'}."
            )
        elif evidence_type == EvidenceType.EXPLICIT:
            if min_months and min_months > 12:
                # Required multiple years but only explicitly listed in skills without role details
                status = RequirementMatchStatus.PARTIALLY_SATISFIED
                score = 0.75
                explanation = (
                    f"Skill '{requirement.canonical_name}' is listed in Technical Skills, but requires "
                    f"{min_months // 12} years verified in work experience."
                )
            else:
                status = RequirementMatchStatus.SATISFIED
                score = 0.90
                explanation = f"Skill '{requirement.canonical_name}' is explicitly listed in candidate skills."
        elif evidence_type == EvidenceType.INFERRED:
            status = RequirementMatchStatus.PARTIALLY_SATISFIED
            score = 0.60
            explanation = f"Skill '{requirement.canonical_name}' was inferred from summary or adjacent tools."
        else:
            status = RequirementMatchStatus.INSUFFICIENT_EVIDENCE
            score = 0.40
            explanation = f"Vague or uncertain mention of '{requirement.canonical_name}' in resume."

        return RequirementMatchResult(
            requirement_id=requirement.id,
            canonical_name=requirement.canonical_name,
            req_type=requirement.req_type,
            status=status,
            score=score,
            candidate_duration_months=skill_item.duration_months,
            required_duration_months=min_months,
            evidence_refs=evidence_refs,
            confidence=skill_item.confidence,
            explanation=explanation,
        )

    def _match_experience_requirement(
        self, requirement: JDRequirement, resume: ParsedResume
    ) -> RequirementMatchResult:
        """Match minimum career experience against candidate's de-duplicated timeline."""
        cand_months = resume.timeline.total_experience_months
        cand_years = resume.timeline.total_experience_years
        req_months = requirement.min_experience_months or 0
        req_years = round(req_months / 12.0, 1)

        refs: List[EvidenceReference] = []
        if resume.experience:
            for exp in resume.experience[:2]:
                refs.append(
                    EvidenceReference(
                        section="EXPERIENCE",
                        text_snippet=f"{exp.raw_title} at {exp.company or 'Company'} ({exp.start_date or ''} - {exp.end_date or ''})",
                        evidence_type=EvidenceType.VERIFIED,
                    )
                )

        if cand_months >= req_months:
            return RequirementMatchResult(
                requirement_id=requirement.id,
                canonical_name=requirement.canonical_name,
                req_type=requirement.req_type,
                status=RequirementMatchStatus.SATISFIED,
                score=1.0,
                candidate_duration_months=cand_months,
                required_duration_months=req_months,
                evidence_refs=refs,
                confidence=1.0,
                explanation=f"Demonstrates {cand_years:.1f} years of continuous experience, exceeding requirement of {req_years:.1f} years.",
            )
        elif cand_months > 0:
            ratio = min(0.9, cand_months / max(1, req_months))
            return RequirementMatchResult(
                requirement_id=requirement.id,
                canonical_name=requirement.canonical_name,
                req_type=requirement.req_type,
                status=RequirementMatchStatus.PARTIALLY_SATISFIED,
                score=round(ratio, 2),
                candidate_duration_months=cand_months,
                required_duration_months=req_months,
                evidence_refs=refs,
                confidence=0.85,
                explanation=f"Candidate has {cand_years:.1f} years of continuous experience vs {req_years:.1f} years required.",
            )
        else:
            return RequirementMatchResult(
                requirement_id=requirement.id,
                canonical_name=requirement.canonical_name,
                req_type=requirement.req_type,
                status=RequirementMatchStatus.NOT_FOUND,
                score=0.0,
                candidate_duration_months=0,
                required_duration_months=req_months,
                evidence_refs=[],
                confidence=0.7,
                explanation=f"No measurable career timeline detected to satisfy {req_years:.1f} years required.",
            )

    def _match_education_requirement(
        self, requirement: JDRequirement, resume: ParsedResume
    ) -> RequirementMatchResult:
        """Match degree requirements against candidate education entries."""
        req_name = requirement.canonical_name.lower()
        refs: List[EvidenceReference] = []

        for edu in resume.education:
            deg_str = f"{edu.degree or ''} in {edu.field_of_study or ''} from {edu.institution or ''}".strip()
            refs.append(
                EvidenceReference(
                    section="EDUCATION",
                    text_snippet=deg_str,
                    evidence_type=EvidenceType.EXPLICIT,
                )
            )

        if not resume.education:
            return RequirementMatchResult(
                requirement_id=requirement.id,
                canonical_name=requirement.canonical_name,
                req_type=requirement.req_type,
                status=RequirementMatchStatus.INSUFFICIENT_EVIDENCE,
                score=0.5,
                evidence_refs=[],
                confidence=0.6,
                explanation="No structured education entries found in resume.",
            )

        # Check degree alignment
        degree_keywords = ["bachelor", "master", "phd", "doctorate"]
        target_keyword = next((kw for kw in degree_keywords if kw in req_name), None)

        if target_keyword:
            has_degree = any(target_keyword in (edu.degree or "").lower() for edu in resume.education)
        else:
            has_degree = any(req_name in (edu.degree or "").lower() for edu in resume.education)

        if has_degree or any("degree" in (edu.degree or "").lower() for edu in resume.education):
            return RequirementMatchResult(
                requirement_id=requirement.id,
                canonical_name=requirement.canonical_name,
                req_type=requirement.req_type,
                status=RequirementMatchStatus.SATISFIED,
                score=1.0,
                evidence_refs=refs,
                confidence=0.9,
                explanation=f"Candidate satisfies education requirement ({requirement.canonical_name}).",
            )

        return RequirementMatchResult(
            requirement_id=requirement.id,
            canonical_name=requirement.canonical_name,
            req_type=requirement.req_type,
            status=RequirementMatchStatus.PARTIALLY_SATISFIED,
            score=0.7,
            evidence_refs=refs,
            confidence=0.8,
            explanation=f"Candidate has academic history, but target degree ({requirement.canonical_name}) was not explicitly matched.",
        )

    def _match_generic_requirement(
        self, requirement: JDRequirement, resume: ParsedResume
    ) -> RequirementMatchResult:
        """Fallback matcher for generic responsibilities or domain keywords."""
        token = requirement.canonical_name.lower()
        if token in resume.raw_text.lower():
            return RequirementMatchResult(
                requirement_id=requirement.id,
                canonical_name=requirement.canonical_name,
                req_type=requirement.req_type,
                status=RequirementMatchStatus.PARTIALLY_SATISFIED,
                score=0.75,
                confidence=0.7,
                explanation=f"Keyword '{token}' observed in candidate text.",
            )
        return RequirementMatchResult(
            requirement_id=requirement.id,
            canonical_name=requirement.canonical_name,
            req_type=requirement.req_type,
            status=RequirementMatchStatus.NOT_FOUND,
            score=0.0,
            confidence=0.8,
            explanation=f"No direct mention of '{token}' found.",
        )

    def match_all_requirements(
        self, requirements: List[JDRequirement], resume: ParsedResume
    ) -> List[RequirementMatchResult]:
        """Evaluate all JD requirements against a candidate's parsed resume."""
        return [self.match_requirement(req, resume) for req in requirements]
