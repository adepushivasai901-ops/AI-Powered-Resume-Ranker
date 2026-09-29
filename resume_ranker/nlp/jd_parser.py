"""Job Description Intelligence: structured parsing of requirements, skills, experience, and seniority."""

import re
from typing import Dict, List, Optional, Set, Tuple

from resume_ranker.models.job_description import (
    ImportanceLevel,
    JDRequirement,
    ParsedJobDescription,
    RequirementType,
)
from resume_ranker.nlp.skill_extractor import SkillIntelligence


class JDParser:
    """Parses unstructured Job Description text into structured, typed atomic requirements."""

    SENIORITY_KEYWORDS: List[Tuple[str, str]] = [
        ("principal", "Principal"),
        ("staff", "Staff"),
        ("lead", "Lead"),
        ("senior", "Senior"),
        ("sr", "Senior"),
        ("mid-level", "Mid-Level"),
        ("mid", "Mid-Level"),
        ("junior", "Junior"),
        ("jr", "Junior"),
        ("intern", "Intern"),
        ("entry-level", "Entry-Level"),
        ("entry", "Entry-Level"),
    ]

    EDUCATION_PATTERNS: List[Tuple[str, str]] = [
        (r"(?i)\bph\.?d\b", "Ph.D."),
        (r"(?i)\bmaster(?:'s)?\b|\bm\.?s\.?\b", "Master's Degree"),
        (r"(?i)\bbachelor(?:'s)?\b|\bb\.?s\.?\b|\bb\.?tech\b", "Bachelor's Degree"),
    ]

    PREFERRED_SECTION_MARKERS: Set[str] = {
        "preferred",
        "nice to have",
        "nice-to-have",
        "preferred qualifications",
        "bonus",
        "bonus points",
        "pluses",
        "plus",
        "good to have",
        "desirable",
        "additional skills",
    }

    REQUIRED_SECTION_MARKERS: Set[str] = {
        "requirements",
        "required qualifications",
        "required",
        "must have",
        "must-have",
        "minimum qualifications",
        "what you need",
        "qualifications",
        "skills & requirements",
    }

    RESPONSIBILITY_SECTION_MARKERS: Set[str] = {
        "responsibilities",
        "what you'll do",
        "duties",
        "role responsibilities",
        "day to day",
    }

    def __init__(self, skill_engine: Optional[SkillIntelligence] = None):
        """Initialize JDParser with skill intelligence engine."""
        self.skill_engine = skill_engine or SkillIntelligence()

    def extract_job_title_and_seniority(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract job title and seniority level from JD header or text lines."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None, None

        title: Optional[str] = None
        # Check explicit title labels
        for line in lines[:8]:
            m = re.match(r"(?i)^(?:position|title|job\s*title|role)\s*:\s*(.+)$", line)
            if m:
                title = m.group(1).strip()
                break

        # If no explicit label, check if first line looks like a title
        if not title and len(lines[0].split()) <= 8:
            title = lines[0]

        # Seniority extraction
        seniority: Optional[str] = None
        search_target = f"{title or ''} {lines[0] if lines else ''}".lower()
        for kw, canonical_sen in self.SENIORITY_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", search_target):
                seniority = canonical_sen
                break

        # Fallback search across first 500 chars if not in title
        if not seniority:
            for kw, canonical_sen in self.SENIORITY_KEYWORDS:
                if re.search(r"\b" + re.escape(kw) + r"\b", text[:500].lower()):
                    seniority = canonical_sen
                    break

        return title, seniority

    def extract_minimum_experience(self, text: str) -> Optional[float]:
        """Extract general overall career experience requirement in years."""
        pattern = r"(?i)(\d+)(?:\s*-\s*\d+)?\+?\s*(?:years?|yrs?)(?:\s*of)?\s*(?:professional|software|relevant|work|industry)?\s*experience"
        matches = re.findall(pattern, text)
        if matches:
            try:
                # Return maximum stated requirement as representative minimum
                years = max([float(m) for m in matches if float(m) <= 20])
                return years
            except ValueError:
                return None
        return None

    def extract_education_requirements(self, text: str) -> List[str]:
        """Extract degree qualifications specified in the JD."""
        found_degrees: List[str] = []
        for pattern, degree_name in self.EDUCATION_PATTERNS:
            if re.search(pattern, text):
                found_degrees.append(degree_name)
        return found_degrees

    def parse_job_description(self, raw_text: str) -> ParsedJobDescription:
        """Parse complete job description into structured requirements."""
        if not raw_text or not raw_text.strip():
            return ParsedJobDescription(raw_text="")

        title, seniority = self.extract_job_title_and_seniority(raw_text)
        min_total_exp = self.extract_minimum_experience(raw_text)
        education_reqs = self.extract_education_requirements(raw_text)

        # Segment JD lines into sections
        lines = raw_text.splitlines()
        current_section = "GENERAL"
        section_lines: Dict[str, List[str]] = {"GENERAL": []}

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            cleaned_header = re.sub(r"^[#*>\-=_~]+\s*", "", line_str)
            cleaned_header = re.sub(r"[\-=_~:#*]+$", "", cleaned_header).lower().strip()

            if any(cleaned_header == marker for marker in self.PREFERRED_SECTION_MARKERS):
                current_section = "PREFERRED"
                section_lines[current_section] = []
            elif any(cleaned_header == marker for marker in self.REQUIRED_SECTION_MARKERS):
                current_section = "REQUIRED"
                section_lines[current_section] = []
            elif any(cleaned_header == marker for marker in self.RESPONSIBILITY_SECTION_MARKERS):
                current_section = "RESPONSIBILITIES"
                section_lines[current_section] = []
            else:
                section_lines.setdefault(current_section, []).append(line_str)

        requirements: List[JDRequirement] = []
        req_counter = 1
        all_detected_skills: List[str] = []
        seen_skills: Set[str] = set()

        # 1. Experience requirement as a structured item
        if min_total_exp is not None:
            req_id = f"REQ-{req_counter:02d}"
            req_counter += 1
            requirements.append(
                JDRequirement(
                    id=req_id,
                    req_type=RequirementType.MINIMUM_EXPERIENCE,
                    canonical_name=f"{min_total_exp:g}+ years experience",
                    importance=ImportanceLevel.HIGH,
                    min_experience_months=int(min_total_exp * 12),
                    original_text=f"Minimum {min_total_exp:g}+ years experience required.",
                    confidence=1.0,
                )
            )

        # 2. Education requirement
        for deg in education_reqs:
            req_id = f"REQ-{req_counter:02d}"
            req_counter += 1
            requirements.append(
                JDRequirement(
                    id=req_id,
                    req_type=RequirementType.EDUCATION,
                    canonical_name=deg.lower(),
                    importance=ImportanceLevel.MEDIUM,
                    original_text=f"Degree requirement: {deg}",
                    confidence=0.9,
                )
            )

        # 3. Process section by section for skills
        for sec_name, sec_content in section_lines.items():
            sec_text = "\n".join(sec_content)
            skills_found = self.skill_engine.extract_skills_from_text(sec_text)

            is_preferred = sec_name == "PREFERRED"

            for canonical, matched_token, ref in skills_found:
                if canonical in seen_skills:
                    continue
                seen_skills.add(canonical)
                all_detected_skills.append(canonical)

                req_id = f"REQ-{req_counter:02d}"
                req_counter += 1

                # Check if specific duration was mentioned for this skill
                skill_dur_match = re.search(
                    r"(\d+)\+?\s*(?:years?|yrs?)(?:.*)" + re.escape(matched_token),
                    ref.text_snippet,
                    re.IGNORECASE,
                )
                min_months = None
                if skill_dur_match:
                    try:
                        min_months = int(float(skill_dur_match.group(1)) * 12)
                    except ValueError:
                        pass

                req_type = RequirementType.PREFERRED_SKILL if is_preferred else RequirementType.MUST_HAVE_SKILL
                importance = (
                    ImportanceLevel.MEDIUM if is_preferred
                    else (ImportanceLevel.CRITICAL if canonical in ["python", "sql", "java", "c++", "c#"] else ImportanceLevel.HIGH)
                )

                requirements.append(
                    JDRequirement(
                        id=req_id,
                        req_type=req_type,
                        canonical_name=canonical,
                        importance=importance,
                        min_experience_months=min_months,
                        original_text=ref.text_snippet,
                        confidence=1.0,
                        keywords=[canonical, matched_token],
                    )
                )

        responsibilities = section_lines.get("RESPONSIBILITIES", [])

        return ParsedJobDescription(
            raw_text=raw_text,
            title=title,
            seniority=seniority,
            min_total_experience_years=min_total_exp,
            required_degrees=education_reqs,
            requirements=requirements,
            detected_skills=all_detected_skills,
            responsibilities=responsibilities,
        )
