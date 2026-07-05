"""
features/elo.py
================
Standard Elo rating update, used to keep each team's Elo current as real
match results come in via ``ingestion/live_updates.py``.
"""

from __future__ import annotations

from config import ELO_K_FACTOR


def expected_score(elo_a: float, elo_b: float) -> float:
    """Probability that side A beats side B, per the standard Elo formula."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def update_elo(
    elo_a: float, elo_b: float, actual_score_a: float, k: float = ELO_K_FACTOR
) -> tuple[float, float]:
    """Return updated (elo_a, elo_b) after a single match.

    Args:
        elo_a: Team A's Elo rating before the match.
        elo_b: Team B's Elo rating before the match.
        actual_score_a: 1.0 if A won, 0.5 if a draw, 0.0 if A lost.
        k: Sensitivity factor -- higher reacts more strongly to a single result.
    """
    exp_a = expected_score(elo_a, elo_b)
    exp_b = 1.0 - exp_a
    actual_score_b = 1.0 - actual_score_a

    new_elo_a = elo_a + k * (actual_score_a - exp_a)
    new_elo_b = elo_b + k * (actual_score_b - exp_b)
    return new_elo_a, new_elo_b


def result_to_score(goals_a: int, goals_b: int) -> float:
    """Convert a scoreline into A's Elo actual-score (1 / 0.5 / 0)."""
    if goals_a > goals_b:
        return 1.0
    if goals_a < goals_b:
        return 0.0
    return 0.5
