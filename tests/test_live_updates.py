"""Unit tests for ingestion/live_updates.py -- especially the idempotency
guarantee (re-applying the same results must never double-count Elo)."""

from __future__ import annotations

import shutil

import pandas as pd
import pytest

from config import HISTORICAL_MATCHES_CSV, TEAMS_CSV
from ingestion import live_updates as lu


@pytest.fixture
def backup_and_restore_csvs(tmp_path):
    """Snapshot teams.csv / historical_matches.csv, restore after the test so
    tests never leave the real sample data mutated."""
    teams_backup = tmp_path / "teams_backup.csv"
    matches_backup = tmp_path / "matches_backup.csv"
    shutil.copy(TEAMS_CSV, teams_backup)
    shutil.copy(HISTORICAL_MATCHES_CSV, matches_backup)
    try:
        yield
    finally:
        shutil.copy(teams_backup, TEAMS_CSV)
        shutil.copy(matches_backup, HISTORICAL_MATCHES_CSV)


def _sample_new_result() -> pd.DataFrame:
    teams = pd.read_csv(TEAMS_CSV)["team"].tolist()
    home, away = teams[0], teams[1]
    return pd.DataFrame([{
        "date": "2099-01-01",  # far-future date guaranteed not to already exist
        "home_team": home, "away_team": away,
        "home_goals": 3, "away_goals": 0, "tournament": "World Cup",
    }])


def test_refresh_from_dataframe_updates_elo_and_history(backup_and_restore_csvs):
    new_result = _sample_new_result()
    home, away = new_result.iloc[0]["home_team"], new_result.iloc[0]["away_team"]

    elo_before = pd.read_csv(TEAMS_CSV).set_index("team")["elo_rating"]
    n_matches_before = len(pd.read_csv(HISTORICAL_MATCHES_CSV))

    summary = lu.refresh_from_dataframe(new_result)

    elo_after = pd.read_csv(TEAMS_CSV).set_index("team")["elo_rating"]
    n_matches_after = len(pd.read_csv(HISTORICAL_MATCHES_CSV))

    assert summary.n_applied == 1
    assert n_matches_after == n_matches_before + 1
    assert elo_after[home] > elo_before[home]  # winner's Elo increases
    assert elo_after[away] < elo_before[away]  # loser's Elo decreases


def test_refresh_is_idempotent_on_rerun(backup_and_restore_csvs):
    new_result = _sample_new_result()

    first = lu.refresh_from_dataframe(new_result)
    elo_after_first = pd.read_csv(TEAMS_CSV).set_index("team")["elo_rating"].copy()
    n_matches_after_first = len(pd.read_csv(HISTORICAL_MATCHES_CSV))

    second = lu.refresh_from_dataframe(new_result)
    elo_after_second = pd.read_csv(TEAMS_CSV).set_index("team")["elo_rating"]
    n_matches_after_second = len(pd.read_csv(HISTORICAL_MATCHES_CSV))

    assert first.n_applied == 1
    assert second.n_applied == 0  # already-seen match must be a no-op
    assert n_matches_after_second == n_matches_after_first
    pd.testing.assert_series_equal(elo_after_first, elo_after_second)


def test_unknown_team_names_are_skipped_not_crashed(backup_and_restore_csvs):
    bogus = pd.DataFrame([{
        "date": "2099-01-02", "home_team": "Atlantis", "away_team": "Narnia",
        "home_goals": 1, "away_goals": 1, "tournament": "World Cup",
    }])
    summary = lu.refresh_from_dataframe(bogus)
    assert summary.n_applied == 0
    assert summary.n_skipped_unknown_team == 1


def test_missing_required_columns_raises_clear_error():
    bad_df = pd.DataFrame([{"home_team": "A", "away_team": "B"}])
    with pytest.raises(ValueError):
        lu.refresh_from_dataframe(bad_df)
