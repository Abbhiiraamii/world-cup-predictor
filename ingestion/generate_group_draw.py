"""
ingestion/generate_group_draw.py
=================================
Builds a plausible 12-group (4 teams each) draw for the 48-team World Cup
format, seeded by Elo rating so the groups look like a real FIFA draw
(one strong seed per group, avoiding two Tier-1 sides in the same group).

This is a simplification of the real confederation-constrained draw
procedure -- good enough for a simulation/demo product, not an official
draw tool.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import FIXTURES_CSV, GROUP_NAMES, RANDOM_SEED, TEAMS_CSV, TEAMS_PER_GROUP
from utils.logger import get_logger

logger = get_logger(__name__, log_file="ingestion.log")
rng = np.random.default_rng(RANDOM_SEED)


def build_group_draw(teams_df: pd.DataFrame) -> pd.DataFrame:
    """Assign teams to groups using seeded pots (pot 1 = strongest 12, etc.)."""
    df = teams_df.sort_values("elo_rating", ascending=False).reset_index(drop=True)
    n_groups = len(GROUP_NAMES)

    pots = [df.iloc[i * n_groups:(i + 1) * n_groups] for i in range(TEAMS_PER_GROUP)]

    assignment = {g: [] for g in GROUP_NAMES}
    for pot in pots:
        shuffled_groups = list(GROUP_NAMES)
        rng.shuffle(shuffled_groups)
        pot_teams = pot["team"].tolist()
        rng.shuffle(pot_teams)
        for group, team in zip(shuffled_groups, pot_teams):
            assignment[group].append(team)

    records = []
    for group, teams in assignment.items():
        for team in teams:
            records.append({"group": group, "team": team})
    return pd.DataFrame(records)


def generate_and_save() -> pd.DataFrame:
    if FIXTURES_CSV.exists():
        logger.info("Group draw already exists at %s", FIXTURES_CSV)
        return pd.read_csv(FIXTURES_CSV)

    teams_df = pd.read_csv(TEAMS_CSV)
    draw_df = build_group_draw(teams_df)
    draw_df.to_csv(FIXTURES_CSV, index=False)
    logger.info("Wrote group draw (%d teams, %d groups) to %s",
                len(draw_df), draw_df["group"].nunique(), FIXTURES_CSV)
    return draw_df


if __name__ == "__main__":
    generate_and_save()
