"""
features/preprocessing.py
==========================
Cleans and normalizes raw CSV inputs before feature engineering.
"""

from __future__ import annotations

import pandas as pd

from config import HISTORICAL_MATCHES_CSV, OUTCOME_AWAY_WIN, OUTCOME_DRAW, OUTCOME_HOME_WIN, TEAMS_CSV
from utils.logger import get_logger

logger = get_logger(__name__, log_file="preprocessing.log")


def load_teams() -> pd.DataFrame:
    """Load and lightly validate the qualified-teams reference table."""
    df = pd.read_csv(TEAMS_CSV)
    required = {"team", "confederation", "fifa_rank", "elo_rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"teams.csv is missing required columns: {missing}")
    df["team"] = df["team"].str.strip()
    df = df.drop_duplicates(subset="team").reset_index(drop=True)
    return df


def load_historical_matches() -> pd.DataFrame:
    """Load, clean, and label historical match results."""
    df = pd.read_csv(HISTORICAL_MATCHES_CSV, parse_dates=["date"])
    required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"historical_matches.csv is missing required columns: {missing}")

    df = df.dropna(subset=list(required)).copy()
    df["home_team"] = df["home_team"].str.strip()
    df["away_team"] = df["away_team"].str.strip()
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)

    df["outcome"] = df.apply(_label_outcome, axis=1)
    logger.info("Loaded %d cleaned historical matches spanning %s to %s",
                len(df), df["date"].min().date(), df["date"].max().date())
    return df


def _label_outcome(row: pd.Series) -> int:
    if row["home_goals"] > row["away_goals"]:
        return OUTCOME_HOME_WIN
    if row["home_goals"] < row["away_goals"]:
        return OUTCOME_AWAY_WIN
    return OUTCOME_DRAW


def merge_team_metadata(matches: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Attach static team metadata (Elo, FIFA rank, confederation) to each match."""
    teams_lookup = teams.set_index("team")
    for side in ("home", "away"):
        cols = teams_lookup.add_prefix(f"{side}_")
        matches = matches.join(cols, on=f"{side}_team")
    return matches
