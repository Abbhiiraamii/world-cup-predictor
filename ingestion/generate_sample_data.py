"""
ingestion/generate_sample_data.py
==================================
Generates realistic **sample** data so the whole pipeline (features -> model
-> simulation -> dashboard) runs end-to-end without any paid API or manual
download.

The data is synthetic but not arbitrary:

* Each of the 48 teams is placed in a strength tier (rough real-world
  standing) and given an Elo rating + FIFA ranking consistent with that tier.
* Historical matches are simulated with an Elo-based win-probability model
  (the same logistic formula used later for prediction), so the resulting
  CSV has realistic structure: stronger teams win more often, draws happen
  at a realistic rate, score-lines follow a Poisson-ish pattern.

Replace these files with real data at any time -- see ``DATA_SOURCES.md``.
Run directly:  ``python -m ingestion.generate_sample_data``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    HISTORICAL_MATCHES_CSV,
    RANDOM_SEED,
    TEAMS_CSV,
)
from utils.logger import get_logger

logger = get_logger(__name__, log_file="ingestion.log")

rng = np.random.default_rng(RANDOM_SEED)

# --------------------------------------------------------------------------
# 48 qualified/likely-qualified teams for the 2026 FIFA World Cup, grouped
# into rough strength tiers. Elo/FIFA numbers are illustrative placeholders
# spread within each tier, NOT official live figures.
# --------------------------------------------------------------------------
TIERED_TEAMS: dict[str, list[str]] = {
    # Tier 1: elite contenders
    "tier1": [
        "Argentina", "France", "Brazil", "England", "Spain",
        "Portugal", "Netherlands", "Belgium",
    ],
    # Tier 2: strong, capable of deep runs
    "tier2": [
        "Germany", "Italy", "Croatia", "Uruguay", "Colombia",
        "Morocco", "Switzerland", "Denmark", "United States", "Mexico",
    ],
    # Tier 3: solid international sides
    "tier3": [
        "Japan", "South Korea", "Senegal", "Ecuador", "Austria",
        "Ukraine", "Poland", "Serbia", "Canada", "Iran",
        "Australia", "Wales",
    ],
    # Tier 4: competitive, occasional upsets
    "tier4": [
        "Tunisia", "Nigeria", "Egypt", "Algeria", "Saudi Arabia",
        "Qatar", "Panama", "Costa Rica", "Peru", "Chile",
        "Scotland", "Turkey",
    ],
    # Tier 5: developing / debutant-tier sides
    "tier5": [
        "Jordan", "Uzbekistan", "New Zealand", "Jamaica",
        "Curacao", "Cape Verde",
    ],
}

CONFEDERATION_MAP = {
    "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL", "Peru": "CONMEBOL", "Chile": "CONMEBOL",
    "France": "UEFA", "England": "UEFA", "Spain": "UEFA", "Portugal": "UEFA",
    "Netherlands": "UEFA", "Belgium": "UEFA", "Germany": "UEFA", "Italy": "UEFA",
    "Croatia": "UEFA", "Switzerland": "UEFA", "Denmark": "UEFA", "Ukraine": "UEFA",
    "Poland": "UEFA", "Serbia": "UEFA", "Wales": "UEFA", "Scotland": "UEFA", "Turkey": "UEFA",
    "Austria": "UEFA",
    "Morocco": "CAF", "Senegal": "CAF", "Tunisia": "CAF", "Nigeria": "CAF",
    "Egypt": "CAF", "Algeria": "CAF", "South Africa": "CAF", "Cape Verde": "CAF",
    "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Panama": "CONCACAF", "Costa Rica": "CONCACAF", "Jamaica": "CONCACAF",
    "Curacao": "CONCACAF", "Haiti": "CONCACAF",
    "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC", "Saudi Arabia": "AFC",
    "Qatar": "AFC", "Jordan": "AFC", "Uzbekistan": "AFC",
    "Australia": "AFC",
    "New Zealand": "OFC",
}

COACH_POOL = [
    "L. Scaloni", "D. Deschamps", "D. Ancelotti", "T. Tuchel", "L. de la Fuente",
    "R. Martinez", "R. Koeman", "D. Tedesco", "J. Nagelsmann", "L. Spalletti",
    "Z. Dalic", "M. Bielsa", "N. Lorenzo", "W. Regragui", "M. Yakin",
    "B. Hjulmand", "M. Pochettino", "J. Berhalter", "A. Fonseca",
]

TIER_BASE_ELO = {"tier1": 1980, "tier2": 1830, "tier3": 1720, "tier4": 1620, "tier5": 1500}
TIER_BASE_RANK = {"tier1": 8, "tier2": 22, "tier3": 40, "tier4": 60, "tier5": 90}
TIER_WC_TITLES = {
    "Brazil": 5, "Germany": 4, "Italy": 4, "Argentina": 3, "France": 2,
    "Uruguay": 2, "England": 1, "Spain": 1,
}


def _build_teams_dataframe() -> pd.DataFrame:
    """Construct the qualified-teams reference table with plausible ratings."""
    records = []
    for tier, teams in TIERED_TEAMS.items():
        base_elo = TIER_BASE_ELO[tier]
        base_rank = TIER_BASE_RANK[tier]
        for i, team in enumerate(teams):
            elo = base_elo - i * rng.integers(2, 10) + rng.normal(0, 8)
            fifa_rank = base_rank + i * 2 + int(rng.integers(0, 3))
            records.append(
                {
                    "team": team,
                    "confederation": CONFEDERATION_MAP.get(team, "UNK"),
                    "fifa_rank": max(1, int(fifa_rank)),
                    "elo_rating": round(float(elo), 1),
                    "coach": COACH_POOL[int(rng.integers(0, len(COACH_POOL)))],
                    "avg_squad_age": round(float(rng.normal(26.5, 1.4)), 1),
                    "squad_value_million_eur": round(
                        float(max(15, rng.normal(
                            {"tier1": 900, "tier2": 450, "tier3": 220,
                             "tier4": 110, "tier5": 40}[tier], 80))), 1
                    ),
                    "world_cup_titles": TIER_WC_TITLES.get(team, 0),
                    "world_cup_appearances": int(rng.integers(1, 22)),
                    "tier": tier,
                }
            )
    df = pd.DataFrame(records)
    df = df.sort_values("elo_rating", ascending=False).reset_index(drop=True)
    df["fifa_rank"] = df["fifa_rank"].rank(method="first", ascending=False).astype(int)
    df["fifa_rank"] = df["fifa_rank"].max() - df["fifa_rank"] + 1
    return df


def _elo_win_probability(elo_a: float, elo_b: float) -> float:
    """Standard Elo expected-score formula."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def _simulate_score(goal_rate: float) -> int:
    return int(rng.poisson(max(0.15, goal_rate)))


def _build_historical_matches(teams_df: pd.DataFrame, n_matches: int = 3000) -> pd.DataFrame:
    """Simulate past international matches using an Elo-based outcome model.

    This produces a dataset with the realistic *statistical shape* of
    international football (favourites win more often, draws are common,
    goal totals follow a Poisson-like pattern) so downstream feature
    engineering and model training behave sensibly.
    """
    teams = teams_df["team"].tolist()
    elos = dict(zip(teams_df["team"], teams_df["elo_rating"]))

    dates = pd.date_range("2019-01-01", "2026-06-01", periods=n_matches)
    records = []
    for i in range(n_matches):
        home, away = rng.choice(teams, size=2, replace=False)
        elo_h, elo_a = elos[home] + 50, elos[away]  # small home advantage
        p_home = _elo_win_probability(elo_h, elo_a)

        # Draw probability shrinks as the Elo gap widens.
        gap = abs(elo_h - elo_a)
        p_draw = max(0.15, 0.30 - gap / 1500)
        p_home_win = p_home * (1 - p_draw)
        p_away_win = 1 - p_draw - p_home_win

        outcome = rng.choice(["H", "D", "A"], p=[p_home_win, p_draw, p_away_win])

        base_rate = 1.3 + (elo_h - elo_a) / 800
        home_goals = _simulate_score(base_rate)
        away_rate = 1.3 - (elo_h - elo_a) / 800
        away_goals = _simulate_score(away_rate)

        # Nudge scoreline to respect the sampled categorical outcome.
        if outcome == "H" and home_goals <= away_goals:
            home_goals = away_goals + 1
        elif outcome == "A" and away_goals <= home_goals:
            away_goals = home_goals + 1
        elif outcome == "D":
            away_goals = home_goals

        records.append(
            {
                "date": dates[i].strftime("%Y-%m-%d"),
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "tournament": rng.choice(
                    ["Friendly", "WC Qualifier", "Continental Cup", "Nations League"],
                    p=[0.35, 0.30, 0.20, 0.15],
                ),
            }
        )

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return df


def generate_and_save() -> None:
    """Entry point: builds teams.csv and historical_matches.csv if absent."""
    if TEAMS_CSV.exists() and HISTORICAL_MATCHES_CSV.exists():
        logger.info("Sample data already present, skipping generation.")
        return

    logger.info("Generating synthetic sample teams table...")
    teams_df = _build_teams_dataframe()
    teams_df.to_csv(TEAMS_CSV, index=False)
    logger.info("Wrote %d teams to %s", len(teams_df), TEAMS_CSV)

    logger.info("Simulating historical match dataset...")
    matches_df = _build_historical_matches(teams_df)
    matches_df.to_csv(HISTORICAL_MATCHES_CSV, index=False)
    logger.info("Wrote %d historical matches to %s", len(matches_df), HISTORICAL_MATCHES_CSV)


if __name__ == "__main__":
    generate_and_save()
