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

    def parse_candidate_info(self, raw_text: str, header_lines: List[str]) -> CandidateInfo:
        """Extract candidate email, phone, and plausible display name."""
        email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", raw_text)
        email = email_match.group(0) if email_match else None

        phone_match = re.search(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", raw_text)
        phone = phone_match.group(0) if phone_match else None

        name = None
        for line in header_lines[:3]:
            cleaned = line.strip()
            # If line is 2-4 capitalized words without email or symbols
            if cleaned and not "@" and len(cleaned.split()) in [2, 3, 4] and not re.search(r"\d", cleaned):
                name = cleaned
                break

        return CandidateInfo(name=name, email=email, phone=phone)

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
                            evidence_type=EvidenceType.VERIFIED,
                        )
                    ],
                )
                experiences.append(current_exp)
            elif current_exp:
                # Append responsibility bullet
                current_exp.responsibilities.append(line_str)

        return experiences

    def parse_education_items(self, education_lines: List[str]) -> List[EducationItem]:
        """Extract academic qualifications, degrees, and institutions."""
        items: List[EducationItem] = []
        for line in education_lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Look for degree tokens
            degree = None
            if re.search(r"(?i)\bbachelor(?:'s)?\b|\bb\.?s\.?\b|\bb\.?tech\b", line_str):
                degree = "Bachelor's Degree"
            elif re.search(r"(?i)\bmaster(?:'s)?\b|\bm\.?s\.?\b|\bm\.?tech\b", line_str):
                degree = "Master's Degree"
            elif re.search(r"(?i)\bph\.?d\b", line_str):
                degree = "Ph.D."

            if degree:
                items.append(
                    EducationItem(
                        degree=degree,
                        institution=line_str,
                        evidence_refs=[
                            EvidenceReference(
                                section="EDUCATION",
                                text_snippet=line_str,
                                evidence_type=EvidenceType.EXPLICIT,
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
