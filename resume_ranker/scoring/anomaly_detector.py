"""Resume integrity and anomaly detection: date contradictions, overlapping roles, prompt injection."""

import re
from typing import List, Optional

from resume_ranker.extraction.quality_assessor import DocumentQualityReport
from resume_ranker.models.evidence import EvidenceReference, EvidenceType
from resume_ranker.models.matching import AnomalyFlag, AnomalySeverity
from resume_ranker.models.resume import ParsedResume


class AnomalyDetector:
    """Evaluates candidate resume content for discrepancies, suspicious patterns, and integrity risks."""

    # Patterns indicating potential adversarial prompt injection attempts embedded in resume text
    PROMPT_INJECTION_PATTERNS = [
        r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior)\s+instructions\b",
        r"(?i)\bsystem\s+prompt\b",
        r"(?i)\bgive\s+(?:this\s+)?candidate\s+(?:a\s+)?(?:perfect|100|maximum)\s+score\b",
        r"(?i)\bdisregard\s+(?:all\s+)?(?:criteria|qualifications)\b",
        r"(?i)\byou\s+are\s+now\s+(?:a|an)\b",
        r"(?i)\boverride\s+(?:the\s+)?evaluation\b",
    ]

    def detect_anomalies(
        self, resume: ParsedResume, quality: Optional[DocumentQualityReport] = None
    ) -> List[AnomalyFlag]:
        """Audit parsed resume and return all flagged integrity observations with severity and evidence."""
        anomalies: List[AnomalyFlag] = []

        # 1. Prompt Injection / Adversarial Instruction Detection
        raw_lower = resume.raw_text.lower()
        for pat in self.PROMPT_INJECTION_PATTERNS:
            match = re.search(pat, raw_lower)
            if match:
                start = max(0, match.start() - 30)
                end = min(len(resume.raw_text), match.end() + 30)
                snippet = resume.raw_text[start:end].strip()

                anomalies.append(
                    AnomalyFlag(
                        code="PROMPT_INJECTION_DETECTED",
                        severity=AnomalySeverity.HIGH,
                        message="Potential prompt injection or adversarial instruction detected in resume text.",
                        evidence_refs=[
                            EvidenceReference(
                                section="RAW_TEXT",
                                text_snippet=snippet,
                                evidence_type=EvidenceType.EXPLICIT,
                            )
                        ],
                        details={"matched_pattern": pat},
                    )
                )

        # 2. Chronological Date Contradictions
        for anomaly_msg in resume.timeline.anomalies:
            anomalies.append(
                AnomalyFlag(
                    code="CHRONOLOGICAL_CONTRADICTION",
                    severity=AnomalySeverity.HIGH,
                    message=anomaly_msg,
                    evidence_refs=[],
                    details={"timeline_anomaly": anomaly_msg},
                )
            )

        # 3. Concurrent Employment Overlaps
        for overlap in resume.timeline.overlaps:
            if overlap.overlap_months > 2:
                anomalies.append(
                    AnomalyFlag(
                        code="CONCURRENT_EMPLOYMENT_OVERLAP",
                        severity=AnomalySeverity.MEDIUM,
                        message=(
                            f"Overlapping employment detected: '{overlap.role_1}' and '{overlap.role_2}' "
                            f"overlap by {overlap.overlap_months} months."
                        ),
                        details={"overlap_months": overlap.overlap_months},
                    )
                )

        # 4. Unbacked / Unsupported Skill Claims
        total_skills = len(resume.skills)
        if total_skills >= 5:
            unbacked_count = sum(1 for s in resume.skills if s.evidence_type == EvidenceType.EXPLICIT)
            if (unbacked_count / total_skills) > 0.70:
                anomalies.append(
                    AnomalyFlag(
                        code="HIGH_UNBACKED_SKILLS_RATIO",
                        severity=AnomalySeverity.LOW,
                        message=(
                            f"{unbacked_count} of {total_skills} skills are listed without corresponding "
                            "practical application evidence in work experience or projects."
                        ),
                        details={"unbacked_skills_count": unbacked_count, "total_skills": total_skills},
                    )
                )

        # 5. Document Quality & Corruption Flags
        if quality:
            if quality.is_scanned_image:
                anomalies.append(
                    AnomalyFlag(
                        code="SCANNED_DOCUMENT_LOW_TEXT",
                        severity=AnomalySeverity.HIGH,
                        message="Resume appears to be a scanned image or photo with minimal extractable digital text.",
                    )
                )
            if quality.suspicious_chars_ratio > 0.05:
                anomalies.append(
                    AnomalyFlag(
                        code="HIGH_FONT_CORRUPTION",
                        severity=AnomalySeverity.MEDIUM,
                        message="High proportion of unprintable/corrupt glyphs detected in PDF text stream.",
                        details={"suspicious_ratio": quality.suspicious_chars_ratio},
                    )
                )

        return anomalies
