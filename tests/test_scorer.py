"""Unit tests for candidate scoring module."""

import pytest
from resume_ranker.scoring.scorer import CandidateScorer


@pytest.fixture
def default_scorer() -> CandidateScorer:
    """Fixture providing CandidateScorer with default weights."""
    return CandidateScorer()


def test_scorer_initialization():
    """Verify CandidateScorer initialization with default and custom weights."""
    scorer = CandidateScorer()
    assert scorer.similarity_weight == 0.50
    assert scorer.required_skills_weight == 0.30
    assert scorer.preferred_skills_weight == 0.10
    assert scorer.experience_weight == 0.10

    # Test backwards compatibility with legacy skills_weight parameter
    legacy_scorer = CandidateScorer(similarity_weight=0.6, skills_weight=0.4)
    assert legacy_scorer.similarity_weight == 0.6
    assert legacy_scorer.required_skills_weight == 0.4


def test_score_identical_high_similarity_candidate(default_scorer: CandidateScorer):
    """Verify that a candidate with 1.0 similarity and all skills matched scores near 100."""
    result = default_scorer.score_candidate(
        candidate_id="cand_top",
        similarity_score=1.0,
        matched_skills=["Python", "SQL", "Docker"],
        missing_skills=[],
        matched_preferred_skills=["AWS"],
        missing_preferred_skills=[],
        experience_score=100.0,
    )

    assert result["candidate_id"] == "cand_top"
    assert result["final_score"] == 100.0
    assert result["similarity_score"] == 100.0
    assert result["required_skill_score"] == 100.0
    assert result["preferred_skill_score"] == 100.0
    assert result["experience_score"] == 100.0
    assert "explanation" in result
    assert "100.00" in result["explanation"]


def test_score_required_skills_fully_matched(default_scorer: CandidateScorer):
    """Verify scoring when all required skills are matched."""
    result = default_scorer.score_candidate(
        candidate_id="cand_full_req",
        similarity_score=0.8,
        matched_skills=["Python", "SQL"],
        missing_skills=[],
    )

    assert result["required_skill_score"] == 100.0
    assert result["matched_required_skills"] == ["Python", "SQL"]
    assert result["missing_required_skills"] == []
    # 50% * 80 + 30% * 100 + 10% * 0 + 10% * 0 = 40 + 30 = 70.0
    assert result["final_score"] == 70.0


def test_score_partial_required_skill_match(default_scorer: CandidateScorer):
    """Verify scoring when half of required skills are matched."""
    result = default_scorer.score_candidate(
        candidate_id="cand_partial",
        similarity_score=0.6,
        matched_skills=["Python"],
        missing_skills=["Kubernetes"],
    )

    assert result["required_skill_score"] == 50.0
    assert result["matched_required_skills"] == ["Python"]
    assert result["missing_required_skills"] == ["Kubernetes"]
    # 50% * 60 + 30% * 50 = 30 + 15 = 45.0
    assert result["final_score"] == 45.0


def test_score_missing_required_skills(default_scorer: CandidateScorer):
    """Verify that completely missing required skills yields 0 for required_skill_score."""
    result = default_scorer.score_candidate(
        candidate_id="cand_missing",
        similarity_score=0.5,
        matched_skills=[],
        missing_skills=["Python", "SQL", "Docker"],
    )

    assert result["required_skill_score"] == 0.0
    assert result["missing_required_skills"] == ["Python", "SQL", "Docker"]
    # 50% * 50 = 25.0
    assert result["final_score"] == 25.0


def test_score_with_preferred_skills(default_scorer: CandidateScorer):
    """Verify preferred skills contribute positively to preferred_skill_score and final_score."""
    without_pref = default_scorer.score_candidate(
        candidate_id="c1",
        similarity_score=0.5,
        matched_skills=["Python"],
        missing_skills=[],
        matched_preferred_skills=[],
        missing_preferred_skills=["AWS"],
    )

    with_pref = default_scorer.score_candidate(
        candidate_id="c2",
        similarity_score=0.5,
        matched_skills=["Python"],
        missing_skills=[],
        matched_preferred_skills=["AWS"],
        missing_preferred_skills=[],
    )

    assert with_pref["preferred_skill_score"] == 100.0
    assert without_pref["preferred_skill_score"] == 0.0
    assert with_pref["final_score"] > without_pref["final_score"]
    # Difference should equal preferred weight: 10% * 100 = 10.0 points
    assert pytest.approx(with_pref["final_score"] - without_pref["final_score"]) == 10.0


def test_custom_score_weighting():
    """Verify that custom weights alter final score calculations accordingly."""
    custom_scorer = CandidateScorer(
        similarity_weight=0.20,
        required_skills_weight=0.60,
        preferred_skills_weight=0.10,
        experience_weight=0.10,
    )
    result = custom_scorer.score_candidate(
        candidate_id="c_custom",
        similarity_score=0.50,  # 50%
        matched_skills=["Python"],
        missing_skills=[],  # 100% required skills
    )
    # 20% * 50 + 60% * 100 = 10 + 60 = 70.0
    assert result["final_score"] == 70.0


def test_score_bounds_and_range(default_scorer: CandidateScorer):
    """Verify that score always remains strictly bounded within [0.0, 100.0]."""
    zero_result = default_scorer.score_candidate(
        candidate_id="c_zero",
        similarity_score=0.0,
        matched_skills=[],
        missing_skills=["Python"],
        matched_preferred_skills=[],
        missing_preferred_skills=["AWS"],
        experience_score=0.0,
    )
    assert zero_result["final_score"] == 0.0

    max_result = default_scorer.score_candidate(
        candidate_id="c_max",
        similarity_score=1.0,
        matched_skills=["Python"],
        missing_skills=[],
        matched_preferred_skills=["AWS"],
        missing_preferred_skills=[],
        experience_score=100.0,
    )
    assert max_result["final_score"] == 100.0


def test_deterministic_scoring(default_scorer: CandidateScorer):
    """Verify that repeated scoring calls with identical data return identical results."""
    args = {
        "candidate_id": "cand_det",
        "similarity_score": 0.73,
        "matched_skills": ["Python", "Flask"],
        "missing_skills": ["PostgreSQL"],
        "matched_preferred_skills": ["Docker"],
        "missing_preferred_skills": ["Kubernetes"],
        "experience_score": 75.0,
    }
    run1 = default_scorer.score_candidate(**args)
    run2 = default_scorer.score_candidate(**args)
    assert run1 == run2


def test_explanation_field_contains_breakdown(default_scorer: CandidateScorer):
    """Verify that the explanation field is informative and contains component weights."""
    result = default_scorer.score_candidate(
        candidate_id="c_exp",
        similarity_score=0.7,
        matched_skills=["Python"],
        missing_skills=["Docker"],
    )
    explanation = result["explanation"]
    assert "TF-IDF lexical similarity" in explanation
    assert "required skills" in explanation
    assert "preferred skills" in explanation
    assert "experience" in explanation
    assert str(result["final_score"]) in explanation


def test_skill_normalization():
    """Verify that technical skill variations and formatting are normalized cleanly."""
    scorer = CandidateScorer()
    assert scorer.normalize_skill("  React.js  ") == "react"
    assert scorer.normalize_skill("NodeJS") == "node.js"
    assert scorer.normalize_skill("PostgreSQL") == "postgresql"
    assert scorer.normalize_skill("C++") == "c++"
    assert scorer.normalize_skill("C#") == "c#"
    assert scorer.normalize_skill(".NET") == ".net"
    assert scorer.normalize_skill("SQL") == "sql"


def test_score_batch(default_scorer: CandidateScorer):
    """Verify batch candidate scoring."""
    batch_input = [
        {
            "candidate_id": "c1",
            "similarity_score": 0.8,
            "matched_skills": ["Python"],
            "missing_skills": [],
        },
        {
            "candidate_id": "c2",
            "similarity_score": 0.4,
            "matched_skills": [],
            "missing_skills": ["Python"],
        },
    ]
    scored = default_scorer.score_batch(batch_input)
    assert len(scored) == 2
    assert scored[0]["candidate_id"] == "c1"
    assert scored[1]["candidate_id"] == "c2"
    assert scored[0]["final_score"] > scored[1]["final_score"]
