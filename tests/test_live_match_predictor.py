"""Unit tests for prediction/live_match_predictor.py."""

from __future__ import annotations

import pytest

from prediction.live_match_predictor import LiveMatchState, explain_live_state, live_adjusted_probabilities


def test_kickoff_scoreless_matches_pre_match_probabilities():
    pre = (0.5, 0.25, 0.25)
    adj = live_adjusted_probabilities(*pre, home_goals=0, away_goals=0, minute=0)
    assert adj == pytest.approx(pre, abs=1e-6)


def test_probabilities_always_sum_to_one():
    pre = (0.4, 0.3, 0.3)
    for goals in [(0, 0), (1, 0), (0, 1), (2, 1), (0, 3)]:
        for minute in [0, 15, 45, 60, 89, 90]:
            adj = live_adjusted_probabilities(*pre, home_goals=goals[0], away_goals=goals[1], minute=minute)
            assert sum(adj) == pytest.approx(1.0, abs=1e-6)


def test_late_lead_is_more_decisive_than_early_lead():
    pre = (0.4, 0.3, 0.3)
    early = live_adjusted_probabilities(*pre, home_goals=1, away_goals=0, minute=10)
    late = live_adjusted_probabilities(*pre, home_goals=1, away_goals=0, minute=85)
    # Home win probability should be higher when the same lead happens later.
    assert late[0] > early[0]


def test_level_score_draw_probability_rises_near_full_time():
    pre = (0.4, 0.3, 0.3)
    early = live_adjusted_probabilities(*pre, home_goals=0, away_goals=0, minute=10)
    late = live_adjusted_probabilities(*pre, home_goals=0, away_goals=0, minute=88)
    assert late[1] > early[1]


def test_trailing_side_draw_chance_shrinks_near_full_time():
    pre = (0.4, 0.3, 0.3)
    early = live_adjusted_probabilities(*pre, home_goals=1, away_goals=0, minute=10)
    late = live_adjusted_probabilities(*pre, home_goals=1, away_goals=0, minute=88)
    assert late[1] < early[1]


def test_explain_live_state_mentions_leader():
    state = LiveMatchState("Alpha", "Beta", 2, 0, 70)
    text = explain_live_state(state, p_home=0.9, p_draw=0.05, p_away=0.05)
    assert "Alpha" in text
    assert "70" in text
