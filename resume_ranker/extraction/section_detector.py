"""Deterministic resume section detection and boundary segmentation."""

import re
from typing import Dict, List, Optional, Tuple

from resume_ranker.models.evidence import DocumentBlock


class SectionDetector:
    """Classifies resume headings and segments raw text or blocks into structured sections."""

    # Canonical section definitions and their common variations/aliases
    SECTION_ALIASES: Dict[str, List[str]] = {
        "SUMMARY": [
            "summary",
            "professional summary",
            "career summary",
            "executive summary",
            "objective",
            "career objective",
            "professional objective",
            "about me",
            "profile",
            "personal profile",
            "overview",
        ],
        "EXPERIENCE": [
            "experience",
            "work experience",
            "professional experience",
            "employment history",
            "work history",
            "employment",
            "career history",
            "relevant experience",
            "practical experience",
            "internships",
            "work",
        ],
        "EDUCATION": [
            "education",
            "academic background",
            "educational background",
            "academic history",
            "academics",
            "academic qualifications",
            "degrees",
            "university",
            "qualifications",
        ],
        "SKILLS": [
            "skills",
            "technical skills",
            "skills & competencies",
            "core competencies",
            "technical proficiencies",
            "technical expertise",
            "technologies",
            "tech stack",
            "tools & technologies",
            "programming skills",
            "key skills",
            "competencies",
            "areas of expertise",
        ],
        "PROJECTS": [
            "projects",
            "personal projects",
            "key projects",
            "notable projects",
            "academic projects",
            "technical projects",
            "project work",
            "portfolio",
        ],
        "CERTIFICATIONS": [
            "certifications",
            "certificates",
            "licenses",
            "professional certifications",
            "credentials",
            "accreditations",
            "licenses & certifications",
        ],
        "ACHIEVEMENTS": [
            "achievements",
            "awards",
            "honors",
            "accomplishments",
            "awards & achievements",
            "honors & awards",
            "key achievements",
        ],
        "PUBLICATIONS": [
            "publications",
            "research papers",
            "papers",
            "patents",
            "research",
        ],
        "LANGUAGES": [
            "languages",
            "language proficiency",
            "languages known",
            "spoken languages",
        ],
        "INTERESTS": [
            "interests",
            "hobbies",
            "activities",
            "extracurricular activities",
            "volunteer work",
            "volunteering",
        ],
    }

    def __init__(self):
        """Build normalized lookup table for fast deterministic matching."""
        self._lookup: Dict[str, str] = {}
        for canonical, aliases in self.SECTION_ALIASES.items():
            for alias in aliases:
                self._lookup[alias.lower().strip()] = canonical

    def clean_heading_candidate(self, line: str) -> str:
        """Strip markdown, numbering, punctuation, and decorative artifacts from heading line candidate."""
        if not line or not isinstance(line, str):
            return ""
        # Remove markdown headers (###), bullet symbols, borders, underlines
        cleaned = re.sub(r"^[#*>\-=_~]+\s*", "", line)
        cleaned = re.sub(r"[\-=_~:#*]+$", "", cleaned)
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", cleaned)  # e.g., "1. Experience"
        # Collapse multiple spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def classify_line_as_heading(self, line: str) -> Optional[str]:
        """Determine if a single line represents a recognized resume section header.

        Employs strict length and structure filters to prevent ordinary sentences from
        generating false positives (e.g. 'I have 5 years experience with Python').

        Args:
            line: Raw line candidate string.

        Returns:
            Canonical section name (e.g., 'EXPERIENCE') or None if not a recognized heading.
        """
        if not line:
            return None

        stripped = line.strip()
        if not stripped:
            return None

        # Guard: Headings are concise titles. Sentences with > 5 words or > 45 characters are not headings.
        words = stripped.split()
        if len(words) > 5 or len(stripped) > 45:
            return None

        # Clean decorative symbols
        cleaned = self.clean_heading_candidate(stripped).lower()
        if not cleaned:
            return None

        # Direct alias match
        if cleaned in self._lookup:
            return self._lookup[cleaned]

        return None

    def segment_text(self, text: str) -> Dict[str, List[str]]:
        """Segment raw text lines into categorized section blocks.

        Args:
            text: Full raw resume text.

        Returns:
            Dictionary mapping canonical section name to lines of content in that section.
            Lines preceding any section heading are stored under 'HEADER'.
        """
        sections: Dict[str, List[str]] = {"HEADER": []}
        current_section = "HEADER"

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            detected_section = self.classify_line_as_heading(line)
            if detected_section:
                current_section = detected_section
                if current_section not in sections:
                    sections[current_section] = []
            else:
                sections[current_section].append(line)

        return sections

    def tag_blocks(self, blocks: List[DocumentBlock]) -> List[DocumentBlock]:
        """Assign detected section labels to sequential DocumentBlocks in-place.

        Args:
            blocks: Sequential list of DocumentBlock instances from document extraction.

        Returns:
            Updated blocks with section metadata populated.
        """
        current_section = "HEADER"
        for block in blocks:
            detected = self.classify_line_as_heading(block.text)
            if detected:
                current_section = detected
                block.section = current_section
            else:
                block.section = current_section
        return blocks
