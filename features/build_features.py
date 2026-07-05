"""
features/build_features.py
===========================
Orchestrates: load raw data -> clean -> engineer features -> persist.

Run directly:  ``python -m features.build_features``
"""

from __future__ import annotations

import pandas as pd

from config import FEATURES_CSV, TEAM_STRENGTH_CSV
from features.feature_engineering import FeatureBuilder
from features.preprocessing import load_historical_matches, load_teams, merge_team_metadata
from utils.logger import get_logger

logger = get_logger(__name__, log_file="feature_engineering.log")


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    teams = load_teams()
    matches = load_historical_matches()
    matches = merge_team_metadata(matches, teams)

    builder = FeatureBuilder()
    feature_df = builder.build(matches)
    feature_df.to_csv(FEATURES_CSV, index=False)
    logger.info("Saved match features to %s", FEATURES_CSV)

    snapshot_df = builder.current_team_snapshot(teams["team"].tolist())
    combined = teams.merge(snapshot_df, on="team", how="left").fillna(
        {"form": 0.5, "avg_goals_for": 1.2, "avg_goals_against": 1.2,
         "win_pct": 0.4, "clean_sheet_pct": 0.25, "wc_experience": 0.1}
    )
    combined.to_csv(TEAM_STRENGTH_CSV, index=False)
    logger.info("Saved current team strength snapshot to %s", TEAM_STRENGTH_CSV)
    return feature_df, combined


if __name__ == "__main__":
    run()
