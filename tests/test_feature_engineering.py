"""Unit tests for features/feature_engineering.py -- focus on correctness and
the no-leakage guarantee (features for a match must only reflect information
available strictly before that match's date)."""

from __future__ import annotations

import pandas as pd
import pytest

from features.feature_engineering import FeatureBuilder


def _toy_matches() -> pd.DataFrame:
    """Three matches for two teams, designed so features are hand-verifiable."""
    return pd.DataFrame(
        [
            # date, home, away, home_goals, away_goals, tournament, outcome,
            # home_elo_rating, away_elo_rating, home_fifa_rank, away_fifa_rank
            {"date": "2020-01-01", "home_team": "Alpha", "away_team": "Beta",
             "home_goals": 2, "away_goals": 0, "tournament": "Friendly", "outcome": 0,
             "home_elo_rating": 1800, "away_elo_rating": 1700,
             "home_fifa_rank": 5, "away_fifa_rank": 20},
            {"date": "2020-02-01", "home_team": "Beta", "away_team": "Alpha",
             "home_goals": 1, "away_goals": 1, "tournament": "Friendly", "outcome": 1,
             "home_elo_rating": 1700, "away_elo_rating": 1810,
             "home_fifa_rank": 20, "away_fifa_rank": 4},
            {"date": "2020-03-01", "home_team": "Alpha", "away_team": "Beta",
             "home_goals": 0, "away_goals": 3, "tournament": "World Cup", "outcome": 2,
             "home_elo_rating": 1810, "away_elo_rating": 1705,
             "home_fifa_rank": 4, "away_fifa_rank": 19},
        ]
    )


def test_first_match_uses_default_priors():
    """Before any history exists, both teams should get neutral default stats."""
    matches = _toy_matches()
    builder = FeatureBuilder()
    features = builder.build(matches)

    first_row = features.iloc[0]
    assert first_row["home_form"] == 0.5
    assert first_row["away_form"] == 0.5
    assert first_row["home_win_pct"] == 0.4
    assert first_row["away_win_pct"] == 0.4


def test_second_match_reflects_only_prior_result():
    """The second match's features must reflect ONLY the first match's outcome,
    never the second match's own result (that would be leakage)."""
    matches = _toy_matches()
    builder = FeatureBuilder()
    features = builder.build(matches)

    second_row = features.iloc[1]
    # Beta is now "home" and lost the first match 0-2 as away side.
    assert second_row["home_team"] == "Beta"
    assert second_row["home_win_pct"] == 0.0  # Beta has 0 wins from 1 match so far
    # Alpha (now away) won its only prior match.
    assert second_row["away_win_pct"] == 1.0


def test_feature_row_count_matches_input():
    matches = _toy_matches()
    builder = FeatureBuilder()
    features = builder.build(matches)
    assert len(features) == len(matches)


def test_head_to_head_neutral_until_minimum_matches():
    """H2H strength should stay neutral (0.5) until H2H_MIN_MATCHES prior
    meetings exist, then reflect the actual head-to-head record."""
    matches = _toy_matches()
    builder = FeatureBuilder()
    features = builder.build(matches)

    # Match 1: no prior meetings -> neutral.
    assert features.iloc[0]["h2h_home_strength"] == 0.5
    # Match 2: only 1 prior meeting, below H2H_MIN_MATCHES=2 -> still neutral.
    assert features.iloc[1]["h2h_home_strength"] == 0.5
    # Match 3: 2 prior meetings now exist (Alpha won 1, drew 1) -> reflects it.
    # From Alpha's (home) perspective: 1 win + 1 draw out of 2 = 0.75.
    assert features.iloc[2]["h2h_home_strength"] == pytest.approx(0.75)


def test_current_team_snapshot_reflects_full_history():
    matches = _toy_matches()
    builder = FeatureBuilder()
    builder.build(matches)
    snapshot = builder.current_team_snapshot(["Alpha", "Beta"])
    snapshot = snapshot.set_index("team")

    # Alpha: W, D, L across 3 matches -> 1 win, 1 draw, 1 loss
    assert snapshot.loc["Alpha", "win_pct"] == pytest.approx(1 / 3)
    # Beta: L, D, W -> 1 win, 1 draw, 1 loss (same distribution, mirrored)
    assert snapshot.loc["Beta", "win_pct"] == pytest.approx(1 / 3)
