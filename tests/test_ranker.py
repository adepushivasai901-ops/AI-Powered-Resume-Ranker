"""Unit tests for candidate ranking module."""

import pytest
from resume_ranker.ranking.ranker import CandidateRanker


@pytest.fixture
def ranker() -> CandidateRanker:
    """Fixture providing CandidateRanker."""
    return CandidateRanker()


def test_ranker_initialization():
    """Verify CandidateRanker score key assignment and custom score keys."""
    default_r = CandidateRanker()
    assert default_r.score_key == "final_score"

    custom_r = CandidateRanker(score_key="custom_score")
    assert custom_r.score_key == "custom_score"


def test_rank_candidates_descending_order(ranker: CandidateRanker):
    """Verify that candidates are ordered by final_score descending."""
    candidates = [
        {"candidate_id": "c_low", "final_score": 45.0},
        {"candidate_id": "c_high", "final_score": 92.5},
        {"candidate_id": "c_mid", "final_score": 70.0},
    ]

    ranked = ranker.rank_candidates(candidates)

    assert len(ranked) == 3
    assert ranked[0]["candidate_id"] == "c_high"
    assert ranked[1]["candidate_id"] == "c_mid"
    assert ranked[2]["candidate_id"] == "c_low"


def test_rank_assignment(ranker: CandidateRanker):
    """Verify that rank numbers start at 1 and increment sequentially."""
    candidates = [
        {"candidate_id": "c1", "final_score": 88.0},
        {"candidate_id": "c2", "final_score": 75.0},
        {"candidate_id": "c3", "final_score": 60.0},
    ]

    ranked = ranker.rank_candidates(candidates)

    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2
    assert ranked[2]["rank"] == 3


def test_deterministic_tie_breaking(ranker: CandidateRanker):
    """Verify that equal scores are broken deterministically using candidate_id."""
    candidates = [
        {"candidate_id": "charlie", "final_score": 80.0},
        {"candidate_id": "alice", "final_score": 80.0},
        {"candidate_id": "bob", "final_score": 80.0},
    ]

    ranked = ranker.rank_candidates(candidates)

    assert [c["candidate_id"] for c in ranked] == ["alice", "bob", "charlie"]
    assert [c["rank"] for c in ranked] == [1, 2, 3]


def test_rank_candidates_empty_input(ranker: CandidateRanker):
    """Verify that ranking an empty candidate list returns an empty list safely."""
    assert ranker.rank_candidates([]) == []
    assert ranker.rank_candidates(None) == []


def test_caller_list_not_mutated(ranker: CandidateRanker):
    """Verify that the caller's input list and original dictionaries are not mutated."""
    original_dict_a = {"candidate_id": "a", "final_score": 50.0}
    original_dict_b = {"candidate_id": "b", "final_score": 90.0}
    input_list = [original_dict_a, original_dict_b]

    # Preserve copies to verify immutability
    original_order = list(input_list)
    original_dict_a_copy = dict(original_dict_a)

    ranked = ranker.rank_candidates(input_list)

    # Input list order is unchanged
    assert input_list[0] == original_dict_a
    assert input_list[1] == original_dict_b
    assert input_list == original_order

    # Original candidate dictionary was not mutated (no 'rank' injected into caller's object)
    assert "rank" not in original_dict_a
    assert original_dict_a == original_dict_a_copy

    # Output list contains rank
    assert ranked[0]["rank"] == 1
    assert ranked[0]["candidate_id"] == "b"


def test_filter_top_candidates(ranker: CandidateRanker):
    """Verify top-N candidate filtering."""
    candidates = [
        {"candidate_id": f"c_{i}", "final_score": float(i)}
        for i in range(15)
    ]
    ranked = ranker.rank_candidates(candidates)

    top_5 = ranker.filter_top_candidates(ranked, top_n=5)
    assert len(top_5) == 5
    assert top_5[0]["rank"] == 1
    assert top_5[4]["rank"] == 5

    # Edge cases for top_n
    assert ranker.filter_top_candidates(ranked, top_n=0) == []
    assert ranker.filter_top_candidates(ranked, top_n=-1) == []
    assert len(ranker.filter_top_candidates(ranked, top_n=100)) == 15
    assert ranker.filter_top_candidates([], top_n=5) == []
