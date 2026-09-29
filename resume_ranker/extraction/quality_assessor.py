"""Document quality assessment module for evaluating extraction integrity."""

import re
from typing import List
from pydantic import BaseModel, ConfigDict, Field


class DocumentQualityReport(BaseModel):
    """Quality metrics and diagnostic warnings for an extracted document."""

    model_config = ConfigDict(extra="ignore")

    total_chars: int = Field(default=0, ge=0, description="Total characters extracted")
    total_words: int = Field(default=0, ge=0, description="Total whitespace-delimited words")
    page_count: int = Field(default=0, ge=0, description="Total number of document pages")
    avg_chars_per_page: float = Field(default=0.0, ge=0.0, description="Average extracted characters per page")
    is_empty: bool = Field(default=False, description="Whether document text is completely blank")
    is_very_low_text: bool = Field(default=False, description="Whether total text volume is unusually low (< 50 chars)")
    is_scanned_image: bool = Field(default=False, description="Whether document is suspected to be a scanned image/photo")
    suspicious_chars_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Ratio of unprintable or corrupt glyphs")
    has_column_interleaving: bool = Field(default=False, description="Whether multi-column text interleaving was detected")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score")
    warnings: List[str] = Field(default_factory=list, description="Diagnostic warnings for recruiters/auditors")


class DocumentQualityAssessor:
    """Evaluates the structural and lexical quality of raw text extracted from documents."""

    def __init__(self, low_text_threshold: int = 50, scanned_page_avg_threshold: int = 40):
        """Initialize assessor thresholds.

        Args:
            low_text_threshold: Minimum expected characters for a valid text resume.
            scanned_page_avg_threshold: Minimum avg characters per page to consider digital text.
        """
        self.low_text_threshold = low_text_threshold
        self.scanned_page_avg_threshold = scanned_page_avg_threshold

    def assess(self, page_texts: List[str]) -> DocumentQualityReport:
        """Run quality checks across extracted document pages.

        Args:
            page_texts: List of extracted text strings per page.

        Returns:
            DocumentQualityReport containing flags, confidence, and warnings.
        """
        page_count = len(page_texts)
        full_text = "\n".join(page_texts).strip()
        total_chars = len(full_text)
        words = full_text.split()
        total_words = len(words)

        avg_chars = (total_chars / page_count) if page_count > 0 else 0.0
        warnings: List[str] = []

        is_empty = total_chars == 0
        is_very_low = 0 < total_chars < self.low_text_threshold
        is_scanned = (page_count > 0 and avg_chars < self.scanned_page_avg_threshold and not is_empty)

        if is_empty:
            warnings.append("Document text is empty; no extractable text stream found.")
        elif is_very_low:
            warnings.append(
                f"Extremely low text volume ({total_chars} characters). Resume may be corrupted or mostly graphical."
            )
        elif is_scanned:
            warnings.append(
                f"Average characters per page ({avg_chars:.1f}) is very low; PDF may be a scanned image or photograph."
            )

        # Check for unprintable/corrupted character ratios
        suspicious_chars = len(re.findall(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]", full_text))
        suspicious_ratio = (suspicious_chars / total_chars) if total_chars > 0 else 0.0

        if suspicious_ratio > 0.05:
            warnings.append(
                f"High proportion of unprintable/corrupt characters ({suspicious_ratio * 100:.1f}%). Possible font encoding issue."
            )

        # Heuristic for multi-column text interleaving:
        # Detect patterns where short capitalized headings/words repeat abruptly with multiple mid-sentence breaks
        has_column_interleaving = False
        lines = [line.strip() for line in full_text.splitlines() if line.strip()]
        if len(lines) > 10:
            short_fragments = sum(1 for line in lines if len(line.split()) <= 2 and not line.isupper())
            if short_fragments / len(lines) > 0.45:
                has_column_interleaving = True
                warnings.append("Suspicious line fragment density detected; PDF may have multi-column reading order issues.")

        # Calculate confidence
        if is_empty:
            confidence = 0.0
        elif is_very_low:
            confidence = 0.2
        elif is_scanned:
            confidence = 0.35
        else:
            confidence = 1.0
            if suspicious_ratio > 0.02:
                confidence -= min(0.3, suspicious_ratio * 5)
            if has_column_interleaving:
                confidence -= 0.15
            confidence = max(0.2, round(confidence, 2))

        return DocumentQualityReport(
            total_chars=total_chars,
            total_words=total_words,
            page_count=page_count,
            avg_chars_per_page=round(avg_chars, 1),
            is_empty=is_empty,
            is_very_low_text=is_very_low,
            is_scanned_image=is_scanned,
            suspicious_chars_ratio=round(suspicious_ratio, 4),
            has_column_interleaving=has_column_interleaving,
            confidence=confidence,
            warnings=warnings,
        )
