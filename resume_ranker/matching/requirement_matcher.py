"""Requirement-level matching engine: evaluating candidate evidence against structured JD requirements."""

from typing import Dict, List, Optional, Set, Tuple

from resume_ranker.models.evidence import EvidenceReference, EvidenceType
from resume_ranker.models.job_description import JDRequirement, RequirementType
from resume_ranker.models.matching import RequirementMatchResult, RequirementMatchStatus
from resume_ranker.models.resume import ParsedResume, SkillItem
from resume_ranker.nlp.skill_extractor import SkillIntelligence


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
        skill_engine = SkillIntelligence()
        canonical_target = skill_engine.normalize_skill_name(requirement.canonical_name.lower().strip())

        # Find matching skill item in candidate profile (matching either canonical or alias)
        skill_item: Optional[SkillItem] = None
        for s in resume.skills:
            if s.canonical_name.lower() == canonical_target or skill_engine.normalize_skill_name(s.canonical_name) == canonical_target:
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

        if evidence_type in [EvidenceType.RESUME_EVIDENCED, EvidenceType.VERIFIED]:
            status = RequirementMatchStatus.SATISFIED
            score = 1.0
            explanation = (
                f"Skill '{requirement.canonical_name}' is evidenced with practical application in "
                f"{', '.join(sections_found) or 'experience records'}."
            )
        elif evidence_type in [EvidenceType.CLAIMED, EvidenceType.EXPLICIT]:
            if min_months and min_months > 12:
                # Required multiple years but only claimed in skills without role details
                status = RequirementMatchStatus.PARTIALLY_SATISFIED
                score = 0.50
                explanation = (
                    f"Skill '{requirement.canonical_name}' is claimed in Technical Skills, but lacks "
                    f"required {min_months // 12}+ years of demonstrated experience in work history."
                )
            else:
                status = RequirementMatchStatus.PARTIALLY_SATISFIED
                score = 0.70
                explanation = (
                    f"Skill '{requirement.canonical_name}' is claimed in candidate skills list, but not "
                    "substantiated by work experience or project evidence."
                )
        elif evidence_type == EvidenceType.INFERRED:
            status = RequirementMatchStatus.PARTIALLY_SATISFIED
            score = 0.45
            explanation = (
                f"Skill '{requirement.canonical_name}' was inferred from summary or adjacent tools; "
                "lacking demonstrated work evidence."
            )
        else:
            status = RequirementMatchStatus.INSUFFICIENT_EVIDENCE
            score = 0.20
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
        req_name = f"{requirement.canonical_name} {requirement.original_text} {' '.join(requirement.keywords)}".lower()
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
                status=RequirementMatchStatus.NOT_FOUND,
                score=0.0,
                evidence_refs=[],
                confidence=0.8,
                explanation=f"No structured education entries found in resume to satisfy {requirement.canonical_name}.",
            )

        # Check degree alignment with robust alias and level reasoning
        satisfied = False
        partial = False
        for edu in resume.education:
            deg = (edu.degree or "").lower()
            field = (edu.field_of_study or "").lower()
            cand_edu_text = f"{deg} {field}".strip()

            is_sat, is_part = self._degree_satisfies_requirement(req_name, cand_edu_text)
            if is_sat:
                satisfied = True
                break
            elif is_part:
                partial = True

        if satisfied or any("degree" in (edu.degree or "").lower() and "bachelor" in req_name for edu in resume.education):
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
        elif partial:
            return RequirementMatchResult(
                requirement_id=requirement.id,
                canonical_name=requirement.canonical_name,
                req_type=requirement.req_type,
                status=RequirementMatchStatus.PARTIALLY_SATISFIED,
                score=0.6,
                evidence_refs=refs,
                confidence=0.8,
                explanation=f"Candidate has related academic history, but target degree ({requirement.canonical_name}) was not fully satisfied.",
            )

        return RequirementMatchResult(
            requirement_id=requirement.id,
            canonical_name=requirement.canonical_name,
            req_type=requirement.req_type,
            status=RequirementMatchStatus.PARTIALLY_SATISFIED,
            score=0.4,
            evidence_refs=refs,
            confidence=0.75,
            explanation=f"Candidate has academic history, but target degree ({requirement.canonical_name}) was not matched.",
        )

    @classmethod
    def _degree_satisfies_requirement(cls, req_name: str, cand_deg: str) -> Tuple[bool, bool]:
        """Check if candidate degree satisfies or partially satisfies required degree."""
        r = req_name.lower().strip()
        c = cand_deg.lower().strip()

        if not c:
            return False, False

        # Direct string containment
        if r in c or c in r:
            return True, False

        bachelor_aliases = {"bachelor", "bachelors", "btech", "b.tech", "be", "b.e", "bs", "b.s", "bsc", "b.sc", "bca", "bba", "undergraduate"}
        master_aliases = {"master", "masters", "mtech", "m.tech", "me", "m.e", "ms", "m.s", "msc", "m.sc", "mca", "mba", "postgraduate"}
        phd_aliases = {"phd", "ph.d", "doctorate", "doctor of philosophy"}

        # Acronym equivalences: B.Tech <-> B.E.
        if any(term in r for term in ["b.tech", "btech", "bachelor of technology"]):
            if any(term in c for term in ["b.tech", "btech", "bachelor of technology", "b.e", "be", "bachelor of engineering"]):
                return True, False
        if any(term in r for term in ["m.tech", "mtech", "master of technology"]):
            if any(term in c for term in ["m.tech", "mtech", "master of technology", "m.e", "me", "master of engineering"]):
                return True, False
        if "mca" in r or "master of computer applications" in r:
            if "mca" in c or "master of computer applications" in c:
                return True, False
        if "mba" in r or "master of business administration" in r:
            if "mba" in c or "master of business administration" in c:
                return True, False

        r_is_bach = any(a in r for a in bachelor_aliases)
        c_is_bach = any(a in c for a in bachelor_aliases)
        r_is_mast = any(a in r for a in master_aliases)
        c_is_mast = any(a in c for a in master_aliases)
        r_is_phd = any(a in r for a in phd_aliases)
        c_is_phd = any(a in c for a in phd_aliases)

        if r_is_bach and (c_is_bach or c_is_mast or c_is_phd):
            return True, False
        if r_is_mast and (c_is_mast or c_is_phd):
            return True, False
        if r_is_phd and c_is_phd:
            return True, False

        if r_is_mast and c_is_bach:
            return False, True

        return False, False

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
        self,
        requirements: Optional[List[JDRequirement]] = None,
        resume: Optional[ParsedResume] = None,
        parsed_resume: Optional[ParsedResume] = None,
    ) -> List[RequirementMatchResult]:
        """Evaluate all JD requirements against a candidate's parsed resume."""
        target_resume = parsed_resume or resume
        if target_resume is None or requirements is None:
            return []
        return [self.match_requirement(req, target_resume) for req in requirements]

    match_requirements = match_all_requirements
