"""Unit tests for ingestion/generate_sample_data.py and features/preprocessing.py."""

from __future__ import annotations

import pandas as pd

from config import N_GROUPS, TEAMS_PER_GROUP
from features.preprocessing import _label_outcome
from ingestion.generate_sample_data import _build_teams_dataframe, _elo_win_probability


def test_teams_dataframe_has_expected_team_count():
    df = _build_teams_dataframe()
    assert len(df) == N_GROUPS * TEAMS_PER_GROUP
    assert df["team"].is_unique


def test_teams_dataframe_required_columns_present():
    df = _build_teams_dataframe()
    required = {"team", "confederation", "fifa_rank", "elo_rating", "coach",
                "avg_squad_age", "squad_value_million_eur", "world_cup_titles",
                "world_cup_appearances"}
    assert required.issubset(set(df.columns))


def test_elo_win_probability_symmetry():
    p_a = _elo_win_probability(1800, 1600)
    p_b = _elo_win_probability(1600, 1800)
    assert abs((p_a + p_b) - 1.0) < 1e-9
    assert p_a > 0.5 > p_b


def test_elo_win_probability_equal_ratings_is_half():
    assert abs(_elo_win_probability(1700, 1700) - 0.5) < 1e-9


def test_label_outcome_home_win():
    row = pd.Series({"home_goals": 2, "away_goals": 1})
    assert _label_outcome(row) == 0


def test_label_outcome_draw():
    row = pd.Series({"home_goals": 1, "away_goals": 1})
    assert _label_outcome(row) == 1


def test_label_outcome_away_win():
    row = pd.Series({"home_goals": 0, "away_goals": 2})
    assert _label_outcome(row) == 2
