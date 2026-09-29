"""Unit tests for TemporalEngine career timeline reasoning."""

import pytest
from resume_ranker.models.resume import ExperienceItem
from resume_ranker.nlp.temporal_engine import TemporalEngine


@pytest.fixture
def temporal_engine() -> TemporalEngine:
    """Fixture providing initialized TemporalEngine."""
    return TemporalEngine()


def test_date_token_normalization(temporal_engine: TemporalEngine):
    """Verify normalization of various date formats."""
    assert temporal_engine.normalize_date_token("2022") == (2022, 1)
    assert temporal_engine.normalize_date_token("2021-04") == (2021, 4)
    assert temporal_engine.normalize_date_token("March 2020") == (2020, 3)
    assert temporal_engine.normalize_date_token("05/2019") == (2019, 5)

    pres = temporal_engine.normalize_date_token("Present")
    assert pres is not None
    assert pres[0] >= 2024


def test_extract_date_range_from_text(temporal_engine: TemporalEngine):
    """Verify extracting start/end pairs from employment text."""
    res = temporal_engine.extract_date_range_from_text("Senior Engineer (Jan 2020 - Dec 2022)")
    assert res is not None
    start, end = res
    assert start == (2020, 1)
    assert end == (2022, 12)


def test_overlapping_employment_not_double_counted(temporal_engine: TemporalEngine):
    """Verify overlapping jobs do not falsely inflate total career duration."""
    # Role 1: Jan 2020 - Dec 2021 (24 months)
    # Role 2: Jun 2020 - Dec 2021 (19 months overlapping with Role 1)
    exp1 = ExperienceItem(
        company="Company A",
        raw_title="Developer",
        start_date="2020-01",
        end_date="2021-12",
    )
    exp2 = ExperienceItem(
        company="Startup B",
        raw_title="Consultant",
        start_date="2020-06",
        end_date="2021-12",
    )

    analysis = temporal_engine.analyze_timeline([exp1, exp2])

    # If naively summed, it would be 24 + 19 = 43 months.
    # Merged continuous duration must be exactly 24 months (2.0 years).
    assert analysis.total_experience_months == 24
    assert analysis.total_experience_years == 2.0
    assert len(analysis.overlaps) == 1
    assert analysis.overlaps[0].overlap_months == 19


def test_gap_detection(temporal_engine: TemporalEngine):
    """Verify detection of employment gaps >= 3 months."""
    # Role 1 ends Dec 2019
    # Role 2 starts Jun 2020 (5 month gap: Jan - May 2020)
    exp1 = ExperienceItem(
        company="Co 1",
        raw_title="Engineer",
        start_date="2018-01",
        end_date="2019-12",
    )
    exp2 = ExperienceItem(
        company="Co 2",
        raw_title="Senior Engineer",
        start_date="2020-06",
        end_date="2022-12",
    )

    analysis = temporal_engine.analyze_timeline([exp1, exp2])
    assert len(analysis.gaps) == 1
    assert analysis.gaps[0].duration_months >= 3


def test_chronological_anomaly_end_before_start(temporal_engine: TemporalEngine):
    """Verify anomaly flagged when end date precedes start date."""
    exp = ExperienceItem(
        company="Bad Dates Corp",
        raw_title="Time Traveler",
        start_date="2023-01",
        end_date="2021-01",
    )
    analysis = temporal_engine.analyze_timeline([exp])
    assert len(analysis.anomalies) > 0
    assert "precedes" in analysis.anomalies[0].lower()
