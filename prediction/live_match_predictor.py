"""
prediction/live_match_predictor.py
===================================
Adjusts a pre-match win probability using the CURRENT scoreline and match
minute, so "who wins" can be answered for a match that's already underway,
not just before kickoff.

Honesty note: this is a transparent, hand-built heuristic, NOT a second
trained model. There is no free, ready-made in-play (minute-by-minute)
training dataset, so rather than pretend otherwise, this module combines the
pre-match model probability with a simple, clearly-documented rule: a lead
matters more as less time remains, and the chance of the scoreline still
changing shrinks as the clock runs down. It is meant to be directionally
sensible and explainable, not a precisely calibrated in-play model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LiveMatchState:
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    minute: int  # 0-90+ (stoppage time is clamped to 90 for this heuristic)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def live_adjusted_probabilities(
    pre_p_home: float, pre_p_draw: float, pre_p_away: float,
    home_goals: int, away_goals: int, minute: int,
) -> tuple[float, float, float]:
    """Return (P(home win), P(draw), P(away win)) adjusted for the live scoreline.

    Behaviour, by construction:
    * At minute=0 with the scoreline still 0-0, this returns the pre-match
      probabilities unchanged (no new information yet).
    * A goal lead is weighted more heavily as the clock runs down (a 1-goal
      lead in the 88th minute is worth far more than one in the 5th).
    * If the scoreline is level, the draw probability grows toward a high
      ceiling as full time approaches (fewer chances left to break the tie).
    * If one side leads, the draw probability shrinks toward zero as full
      time approaches (an equalizer becomes less likely the less time is left).
    """
    minute = min(max(minute, 0), 90)
    time_remaining_frac = (90 - minute) / 90.0
    goal_diff = home_goals - away_goals

    denom = pre_p_home + pre_p_away
    p_home_norm = pre_p_home / denom if denom > 0 else 0.5
    baseline_logit = _logit(p_home_norm)

    # Urgency: a goal matters ~1x at kickoff, up to ~3.5x as full time nears.
    urgency = 1.0 + 2.5 * (1.0 - time_remaining_frac)
    adjusted_logit = baseline_logit + goal_diff * urgency * 1.15
    p_home_norm_adj = _sigmoid(adjusted_logit)

    if goal_diff == 0:
        # Still level: draw becomes more likely as time runs out.
        p_draw_adj = pre_p_draw + (0.92 - pre_p_draw) * (1.0 - time_remaining_frac)
    else:
        # Someone's ahead: draw probability shrinks as time runs out.
        p_draw_adj = pre_p_draw * time_remaining_frac

    p_draw_adj = min(max(p_draw_adj, 0.0), 0.97)
    remaining = 1.0 - p_draw_adj
    p_home_adj = remaining * p_home_norm_adj
    p_away_adj = remaining * (1.0 - p_home_norm_adj)

    return p_home_adj, p_draw_adj, p_away_adj


def explain_live_state(state: LiveMatchState, p_home: float, p_draw: float, p_away: float) -> str:
    """Short, plain-language explanation of the live win probability."""
    minute = min(max(state.minute, 0), 90)
    time_left = 90 - minute
    diff = state.home_goals - state.away_goals

    if diff == 0:
        score_desc = f"level at {state.home_goals}-{state.away_goals}"
    elif diff > 0:
        score_desc = f"{state.home_team} leading {state.home_goals}-{state.away_goals}"
    else:
        score_desc = f"{state.away_team} leading {state.away_goals}-{state.home_goals}"

    favorite, favorite_prob = (
        (state.home_team, p_home) if p_home >= p_away else (state.away_team, p_away)
    )

    return (
        f"At minute {minute}' ({score_desc}, {time_left}' remaining), "
        f"**{favorite}** is favored to win with a **{favorite_prob * 100:.1f}%** probability, "
        f"factoring in the current scoreline, time remaining, and each side's pre-match strength."
    )
