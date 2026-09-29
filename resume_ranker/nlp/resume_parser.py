"""Full resume parsing pipeline: integrating extraction, sectioning, skills, timeline, and education."""

import re
from typing import Dict, List, Optional
import uuid

from resume_ranker.extraction.pdf_extractor import ExtractedDocument
from resume_ranker.extraction.section_detector import SectionDetector
from resume_ranker.models.evidence import DocumentBlock, EvidenceReference, EvidenceType, Provenance
from resume_ranker.models.resume import (
    CandidateInfo,
    DocumentMetadata,
    EducationItem,
    ExperienceItem,
    ParsedResume,
    SkillItem,
    TimelineAnalysis,
)
from resume_ranker.nlp.skill_extractor import SkillIntelligence
from resume_ranker.nlp.temporal_engine import TemporalEngine


class ResumeParser:
    """Orchestrates structured resume ingestion, sectioning, skill verification, and timeline analysis."""

    def __init__(
        self,
        section_detector: Optional[SectionDetector] = None,
        skill_engine: Optional[SkillIntelligence] = None,
        temporal_engine: Optional[TemporalEngine] = None,
    ):
        """Initialize parser components."""
        self.section_detector = section_detector or SectionDetector()
        self.skill_engine = skill_engine or SkillIntelligence()
        self.temporal_engine = temporal_engine or TemporalEngine()

    NON_NAME_TOKENS = {
        "resume", "curriculum", "vitae", "cv", "portfolio", "profile", "summary",
        "contact", "experience", "education", "skills", "projects", "certifications",
        "engineer", "developer", "designer", "manager", "architect", "lead", "specialist",
        "analyst", "consultant", "scientist", "programmer", "administrator", "director",
        "intern", "phone", "email", "address", "github", "linkedin", "page", "university",
        "college", "institute", "school", "technologies", "languages", "tools",
    }

    DEGREE_PATTERNS = [
        (r"(?i)\b(?:ph\.?d|doctorate|doctor of philosophy)\b", "Ph.D."),
        (r"(?i)\b(?:m\.?tech|master of technology)\b", "M.Tech"),
        (r"(?i)\b(?:m\.?e\.?|master of engineering)\b", "M.E."),
        (r"(?i)\b(?:mca|master of computer applications)\b", "MCA"),
        (r"(?i)\b(?:mba|master of business administration)\b", "MBA"),
        (r"(?i)\b(?:master(?:'s)?(?:\s+degree)?|\bm\.?s\.?|\bm\.?sc)\b", "Master's Degree"),
        (r"(?i)\b(?:b\.?tech|bachelor of technology)\b", "B.Tech"),
        (r"(?i)\b(?:b\.?e\.?|bachelor of engineering)\b", "B.E."),
        (r"(?i)\b(?:bca|bachelor of computer applications)\b", "BCA"),
        (r"(?i)\b(?:bba|bachelor of business administration)\b", "BBA"),
        (r"(?i)\b(?:bachelor(?:'s)?(?:\s+degree)?|\bb\.?s\.?|\bb\.?sc)\b", "Bachelor's Degree"),
        (r"(?i)\b(?:associate(?:'s)?(?:\s+degree)?|\ba\.?s\.?|\ba\.?a\.?)\b", "Associate Degree"),
        (r"(?i)\bdiploma\b", "Diploma"),
    ]

    def parse_candidate_info(
        self, raw_text: str, header_lines: Optional[List[str]] = None
    ) -> CandidateInfo:
        """Extract candidate email, phone, links, and reliable display name without fabrication."""
        email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", raw_text)
        email = email_match.group(0) if email_match else None

        phone_match = re.search(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", raw_text)
        phone = phone_match.group(0) if phone_match else None

        # Extract links
        links = re.findall(
            r"(?i)\b(?:https?://|www\.)[^\s,;()]+|(?:linkedin\.com/in/[^\s,;()]+)|(?:github\.com/[^\s,;()]+)",
            raw_text,
        )

        if header_lines is None:
            header_lines = [l.strip() for l in raw_text.splitlines()[:10] if l.strip()]

        name = None
        for line in header_lines[:5]:
            cleaned = line.strip()
            if not cleaned:
                continue

            # Skip lines with contact information or links
            if "@" in cleaned or re.search(r"(?i)https?://|www\.|\.com|\.io|\.org|\.net|github|linkedin", cleaned):
                continue
            if re.search(r"\d", cleaned):
                continue
            if re.search(r"[-–—|/]", cleaned) and len(cleaned.split()) > 2:
                continue

            words = [w.strip(".,;:()") for w in cleaned.split() if w.strip(".,;:()")]
            if len(words) in [2, 3, 4]:
                lower_words = [w.lower() for w in words]
                if any(w in self.NON_NAME_TOKENS for w in lower_words):
                    continue
                # Ensure each word starts with a capital letter
                if all(w[0].isupper() and w.isalpha() for w in words):
                    name = " ".join(words)
                    break

        if not name:
            name = "Unknown Candidate"

        return CandidateInfo(name=name, email=email, phone=phone, links=links)

    def parse_experience_items(self, experience_lines: List[str]) -> List[ExperienceItem]:
        """Extract job roles, companies, and date intervals from experience section lines."""
        experiences: List[ExperienceItem] = []
        current_exp: Optional[ExperienceItem] = None

        for line in experience_lines:
            line_str = line.strip()
            if not line_str:
                continue

            date_pair = self.temporal_engine.extract_date_range_from_text(line_str)
            if date_pair:
                start_tup, end_tup = date_pair
                start_str = self.temporal_engine.format_date_tuple(start_tup)
                end_str = self.temporal_engine.format_date_tuple(end_tup)

                # Strip date pattern to isolate role title/company
                title_company = re.sub(
                    r"(?i)\b(?:[a-z]+[,\.\s]+)?\d{4}\s*(?:-|–|—|to)\s*(?:(?:[a-z]+[,\.\s]+)?\d{4}|present|current|now)\b",
                    "",
                    line_str,
                ).strip(" -–—()|,")

                parts = [p.strip() for p in re.split(r"[-–—|@,]\s*", title_company) if p.strip()]
                raw_title = parts[0] if parts else line_str
                company = parts[1] if len(parts) > 1 else None

                current_exp = ExperienceItem(
                    raw_title=raw_title,
                    company=company,
                    start_date=start_str,
                    end_date=end_str,
                    is_current="present" in line_str.lower() or "current" in line_str.lower(),
                    evidence_refs=[
                        EvidenceReference(
                            section="EXPERIENCE",
                            text_snippet=line_str,
                            evidence_type=EvidenceType.RESUME_EVIDENCED,
                        )
                    ],
                )
                experiences.append(current_exp)
            elif current_exp:
                # Append responsibility bullet
                current_exp.responsibilities.append(line_str)

        return experiences

    def parse_education_items(self, education_lines: List[str]) -> List[EducationItem]:
        """Extract academic qualifications, degrees, institutions, and fields of study."""
        items: List[EducationItem] = []
        for line in education_lines:
            line_str = line.strip()
            if not line_str:
                continue

            degree = None
            for pattern, canonical_deg in self.DEGREE_PATTERNS:
                if re.search(pattern, line_str):
                    degree = canonical_deg
                    break

            if degree:
                # Extract field of study (e.g. "in Computer Science")
                field_match = re.search(r"(?i)\bin\s+([A-Za-z\s/&-]+?)(?:,|\(|$|\d{4})", line_str)
                field_of_study = field_match.group(1).strip() if field_match else None

                # Extract graduation year
                year_match = re.search(r"\b(19\d{2}|20\d{2})\b", line_str)
                grad_year = int(year_match.group(1)) if year_match else None

                # Extract institution if separated by from / at / comma
                inst_match = re.search(r"(?i)(?:from|at)\s+([A-Za-z\s/&.-]+?)(?:,|\(|$|\d{4})", line_str)
                institution = inst_match.group(1).strip() if inst_match else line_str

                items.append(
                    EducationItem(
                        degree=degree,
                        field_of_study=field_of_study,
                        institution=institution,
                        graduation_year=grad_year,
                        evidence_refs=[
                            EvidenceReference(
                                section="EDUCATION",
                                text_snippet=line_str,
                                evidence_type=EvidenceType.CLAIMED,
                            )
                        ],
                    )
                )

        return items

    def parse_document(self, extracted: ExtractedDocument, filename: str) -> ParsedResume:
        """Process an ExtractedDocument into a canonical ParsedResume representation."""
        # 1. Segment text and tag blocks with sections
        sections = self.section_detector.segment_text(extracted.raw_text)
        tagged_blocks = self.section_detector.tag_blocks(extracted.blocks)

        # 2. Extract and verify skills against evidence sections
        skills = self.skill_engine.extract_and_verify_resume_skills(sections, tagged_blocks)

        # 3. Extract experience history and compute career timeline
        experience_lines = sections.get("EXPERIENCE", [])
        experiences = self.parse_experience_items(experience_lines)
        timeline = self.temporal_engine.analyze_timeline(experiences)

        # 4. Extract education
        education_lines = sections.get("EDUCATION", [])
        education = self.parse_education_items(education_lines)

        # 5. Extract candidate info
        header_lines = sections.get("HEADER", [])
        candidate = self.parse_candidate_info(extracted.raw_text, header_lines)

        # 6. Summary text if present
        summary_lines = sections.get("SUMMARY", [])
        summary = " ".join(summary_lines) if summary_lines else None

        # 7. Document metadata
        doc_metadata = DocumentMetadata(
            id=str(uuid.uuid4().hex[:8]),
            filename=filename,
            source_type="pdf",
            page_count=extracted.page_count,
            parser_path=extracted.parser_used,
            parse_confidence=extracted.quality.confidence,
            warnings=extracted.quality.warnings,
        )

        provenance = Provenance(
            parser="pypdf",
            parser_version="6.19.0",
            model_name="en_core_web_sm",
        )

        return ParsedResume(
            document=doc_metadata,
            candidate=candidate,
            summary=summary,
            skills=skills,
            experience=experiences,
            education=education,
            timeline=timeline,
            blocks=tagged_blocks,
            sections_detected=sections,
            provenance=provenance,
            raw_text=extracted.raw_text,
        )
