"""Temporal reasoning engine: date normalization, non-overlapping interval math, and career timeline analysis."""

from datetime import datetime
import re
from typing import Dict, List, Optional, Tuple

from resume_ranker.models.resume import (
    ExperienceItem,
    TimelineAnalysis,
    TimelineGap,
    TimelineOverlap,
)

# Reference current year and month for 'Present' dates
CURRENT_YEAR = 2026
CURRENT_MONTH = 9

MONTH_NAMES: Dict[str, int] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


class TemporalEngine:
    """Parses dates from employment records and computes non-overlapping career timelines."""

    def normalize_date_token(self, token: str) -> Optional[Tuple[int, int]]:
        """Normalize a date expression into (year, month) tuple.

        Supports YYYY, YYYY-MM, Month YYYY, and 'Present'.
        """
        if not token:
            return None
        cleaned = token.strip().lower()

        if cleaned in ["present", "current", "now", "till date", "ongoing"]:
            return (CURRENT_YEAR, CURRENT_MONTH)

        # Match MM/YYYY or MM-YYYY
        m_slash = re.match(r"^(\d{1,2})[/-](\d{4})$", cleaned)
        if m_slash:
            month = int(m_slash.group(1))
            year = int(m_slash.group(2))
            if 1 <= month <= 12 and 1960 <= year <= CURRENT_YEAR + 1:
                return (year, month)

        # Match YYYY-MM
        m_iso = re.match(r"^(\d{4})[/-](\d{1,2})$", cleaned)
        if m_iso:
            year = int(m_iso.group(1))
            month = int(m_iso.group(2))
            if 1 <= month <= 12 and 1960 <= year <= CURRENT_YEAR + 1:
                return (year, month)

        # Match 'Month YYYY' (e.g. 'March 2021' or 'Mar, 2021')
        m_text = re.match(r"^([a-z]+)[,\.\s]+(\d{4})$", cleaned)
        if m_text:
            m_str = m_text.group(1)
            year = int(m_text.group(2))
            month = MONTH_NAMES.get(m_str, 1)
            if 1960 <= year <= CURRENT_YEAR + 1:
                return (year, month)

        # Match standalone 4-digit YYYY
        m_year = re.match(r"^(\d{4})$", cleaned)
        if m_year:
            year = int(m_year.group(1))
            if 1960 <= year <= CURRENT_YEAR + 1:
                return (year, 1)

        return None

    def format_date_tuple(self, dt: Optional[Tuple[int, int]]) -> Optional[str]:
        """Format (year, month) to standard YYYY-MM string."""
        if not dt:
            return None
        return f"{dt[0]:04d}-{dt[1]:02d}"

    def date_to_absolute_months(self, year: int, month: int) -> int:
        """Convert year and month to an absolute integer count of months since year 0."""
        return (year * 12) + month

    def extract_date_range_from_text(self, text: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Extract start and end date pair from a line of text (e.g., 'Mar 2020 - Dec 2022')."""
        pattern = (
            r"(?i)\b(?:([a-z]+)[,\.\s]+)?(\d{4})\s*(?:-|–|—|to)\s*"
            r"(?:(?:([a-z]+)[,\.\s]+)?(\d{4})|(present|current|now))\b"
        )
        match = re.search(pattern, text)
        if not match:
            return None

        m1_name, y1_str, m2_name, y2_str, present_kw = match.groups()

        try:
            y1 = int(y1_str)
            m1 = MONTH_NAMES.get(m1_name.lower(), 1) if m1_name else 1
            start_date = (y1, m1)

            if present_kw:
                end_date = (CURRENT_YEAR, CURRENT_MONTH)
            else:
                y2 = int(y2_str)
                m2 = MONTH_NAMES.get(m2_name.lower(), 12) if m2_name else 12
                end_date = (y2, m2)

            return start_date, end_date
        except (ValueError, AttributeError):
            return None

    def merge_intervals(self, intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping or continuous intervals of months to prevent double counting."""
        if not intervals:
            return []

        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged: List[Tuple[int, int]] = [sorted_intervals[0]]

        for current in sorted_intervals[1:]:
            prev_start, prev_end = merged[-1]
            curr_start, curr_end = current

            # Overlapping or adjacent
            if curr_start <= prev_end + 1:
                merged[-1] = (prev_start, max(prev_end, curr_end))
            else:
                merged.append(current)

        return merged

    def analyze_timeline(self, experiences: List[ExperienceItem]) -> TimelineAnalysis:
        """Analyze complete candidate employment history to compute real career duration and gaps.

        Strictly avoids double-counting overlapping roles.
        """
        raw_intervals: List[Tuple[int, int, str]] = []
        anomalies: List[str] = []

        for exp in experiences:
            # Parse dates if not already structured
            start_tup = self.normalize_date_token(exp.start_date) if exp.start_date else None
            end_tup = self.normalize_date_token(exp.end_date) if exp.end_date else None

            if start_tup and end_tup:
                start_abs = self.date_to_absolute_months(*start_tup)
                end_abs = self.date_to_absolute_months(*end_tup)

                if end_abs < start_abs:
                    anomalies.append(
                        f"Chronological anomaly in role '{exp.raw_title}': End date ({exp.end_date}) precedes start date ({exp.start_date})."
                    )
                    continue

                duration = max(1, end_abs - start_abs + 1)
                exp.duration_months = duration
                raw_intervals.append((start_abs, end_abs, exp.company or exp.raw_title))

        if not raw_intervals:
            return TimelineAnalysis(anomalies=anomalies)

        # Find overlaps
        overlaps: List[TimelineOverlap] = []
        for i in range(len(raw_intervals)):
            for j in range(i + 1, len(raw_intervals)):
                s1, e1, name1 = raw_intervals[i]
                s2, e2, name2 = raw_intervals[j]

                overlap_start = max(s1, s2)
                overlap_end = min(e1, e2)

                if overlap_start <= overlap_end:
                    overlap_duration = (overlap_end - overlap_start) + 1
                    if overlap_duration > 1:
                        overlaps.append(
                            TimelineOverlap(
                                role_1=name1,
                                role_2=name2,
                                overlap_months=overlap_duration,
                            )
                        )

        # Merge intervals to prevent double-counting
        month_intervals = [(s, e) for s, e, _ in raw_intervals]
        merged = self.merge_intervals(month_intervals)

        total_months = sum((end - start + 1) for start, end in merged)
        total_years = round(total_months / 12.0, 1)

        # Find employment gaps (> 3 months between merged blocks)
        gaps: List[TimelineGap] = []
        for i in range(len(merged) - 1):
            curr_end = merged[i][1]
            next_start = merged[i + 1][0]
            gap_len = (next_start - curr_end) - 1
            if gap_len >= 3:
                start_yr, start_mo = divmod(curr_end, 12)
                end_yr, end_mo = divmod(next_start, 12)
                gaps.append(
                    TimelineGap(
                        start_date=f"{start_yr:04d}-{max(1, start_mo):02d}",
                        end_date=f"{end_yr:04d}-{max(1, end_mo):02d}",
                        duration_months=gap_len,
                        description=f"{gap_len}-month gap in continuous employment",
                    )
                )

        earliest_abs = merged[0][0]
        latest_abs = merged[-1][1]
        earliest_yr, earliest_mo = divmod(earliest_abs, 12)
        latest_yr, latest_mo = divmod(latest_abs, 12)

        return TimelineAnalysis(
            total_experience_months=total_months,
            total_experience_years=total_years,
            earliest_date=f"{earliest_yr:04d}-{max(1, earliest_mo):02d}",
            latest_date=f"{latest_yr:04d}-{max(1, latest_mo):02d}",
            gaps=gaps,
            overlaps=overlaps,
            anomalies=anomalies,
        )
