"""Candidate ranking interface for ordering candidates by relevance deterministically."""

from typing import Any, Dict, List, Optional, Tuple


class CandidateRanker:
    """Ranks candidates based on computed relevance scores and breaks ties deterministically."""

    def __init__(self, score_key: str = "final_score"):
        """Initialize ranker with target score metric.

        Args:
            score_key: Dictionary key representing candidate total score (default: 'final_score').
        """
        self.score_key = score_key

    def rank_candidates(
        self, candidates: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Sort candidates in descending order of score and assign ranks.

        Ties are broken deterministically by sorting candidate_id in ascending alphabetical order.
        Does not mutate the caller's input list or candidate dictionary objects.

        Args:
            candidates: List of scored candidate dictionaries.

        Returns:
            New list of candidate dictionaries ordered by rank with 'rank' assigned.
        """
        if not candidates:
            return []

        # Create shallow copies of dictionaries to avoid mutating caller's original objects
        copied_candidates = [dict(c) for c in candidates]

        # Sort key: (-score, candidate_id) ensures descending score, ascending ID on ties
        def sort_key(c: Dict[str, Any]) -> Tuple[float, str]:
            score_val = float(c.get(self.score_key, 0.0))
            cid = str(c.get("candidate_id", ""))
            return (-score_val, cid)

        sorted_candidates = sorted(copied_candidates, key=sort_key)

        # Assign rank ordinals starting from 1
        for idx, candidate in enumerate(sorted_candidates, start=1):
            candidate["rank"] = idx

        return sorted_candidates

    def filter_top_candidates(
        self, ranked_candidates: Optional[List[Dict[str, Any]]], top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Filter top N ranked candidates.

        Args:
            ranked_candidates: Ordered list of candidate records.
            top_n: Maximum number of candidate profiles to return.

        Returns:
            Top N candidate profiles (or empty list if top_n <= 0 or input is empty).
        """
        if not ranked_candidates or top_n <= 0:
            return []
        return ranked_candidates[:top_n]
