"""
features/feature_engineering.py
================================
Turns cleaned historical matches into a leakage-free feature table for
training the match-outcome model, and produces a "current form" snapshot per
team (as of the most recent match) used later for tournament simulation.

Leakage safety: for every match, the features describing each team are
computed using ONLY matches that happened strictly before that match's date.
The running stats are updated with the match's result only *after* the
feature row has been emitted.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import FORM_WINDOW, GOALS_WINDOW, H2H_MIN_MATCHES
from utils.logger import get_logger

logger = get_logger(__name__, log_file="feature_engineering.log")


@dataclass
class _TeamState:
    """Rolling per-team statistics updated match-by-match in time order."""

    results: deque = field(default_factory=lambda: deque(maxlen=FORM_WINDOW))
    goals_for: deque = field(default_factory=lambda: deque(maxlen=GOALS_WINDOW))
    goals_against: deque = field(default_factory=lambda: deque(maxlen=GOALS_WINDOW))
    clean_sheets: deque = field(default_factory=lambda: deque(maxlen=GOALS_WINDOW))
    wins: int = 0
    draws: int = 0
    losses: int = 0
    matches_played: int = 0
    world_cup_matches: int = 0
    world_cup_wins: int = 0

    # ---- read-only summaries -------------------------------------------------
    def form_points(self) -> float:
        """Points-per-game over the last N matches (3/1/0), normalized 0-1."""
        if not self.results:
            return 0.5
        points = sum(self.results)
        return points / (3 * len(self.results))

    def avg_goals_for(self) -> float:
        return float(np.mean(self.goals_for)) if self.goals_for else 1.2

    def avg_goals_against(self) -> float:
        return float(np.mean(self.goals_against)) if self.goals_against else 1.2

    def win_pct(self) -> float:
        return self.wins / self.matches_played if self.matches_played else 0.4

    def clean_sheet_pct(self) -> float:
        return float(np.mean(self.clean_sheets)) if self.clean_sheets else 0.25

    def world_cup_experience(self) -> float:
        """Blend of appearances and win ratio in past World Cups, 0-1 scaled."""
        if self.world_cup_matches == 0:
            return 0.1
        win_ratio = self.world_cup_wins / self.world_cup_matches
        exposure = min(self.world_cup_matches / 20.0, 1.0)
        return 0.5 * win_ratio + 0.5 * exposure

    # ---- update ---------------------------------------------------------------
    def update(self, goals_for: int, goals_against: int, is_world_cup: bool) -> None:
        if goals_for > goals_against:
            self.results.append(3)
            self.wins += 1
        elif goals_for == goals_against:
            self.results.append(1)
            self.draws += 1
        else:
            self.results.append(0)
            self.losses += 1

        self.goals_for.append(goals_for)
        self.goals_against.append(goals_against)
        self.clean_sheets.append(1 if goals_against == 0 else 0)
        self.matches_played += 1

        if is_world_cup:
            self.world_cup_matches += 1
            if goals_for > goals_against:
                self.world_cup_wins += 1


class FeatureBuilder:
    """Stateful, chronological feature builder guaranteeing no data leakage."""

    def __init__(self) -> None:
        self._team_states: dict[str, _TeamState] = defaultdict(_TeamState)
        self._h2h: dict[tuple[str, str], list[int]] = defaultdict(list)  # 1=A win,0=draw,-1=B win

    def _h2h_key(self, a: str, b: str) -> tuple[str, str]:
        return tuple(sorted((a, b)))  # type: ignore[return-value]

    def _h2h_strength(self, team: str, opponent: str) -> float:
        """Return team's historical head-to-head win rate vs opponent (0-1, 0.5=neutral)."""
        key = self._h2h_key(team, opponent)
        history = self._h2h[key]
        if len(history) < H2H_MIN_MATCHES:
            return 0.5
        # history stored relative to sorted(key)[0]; flip sign if `team` is the other side
        sign = 1 if team == key[0] else -1
        wins = sum(1 for r in history if r * sign == 1)
        draws = sum(1 for r in history if r == 0)
        return (wins + 0.5 * draws) / len(history)

    def _record_h2h(self, home: str, away: str, home_goals: int, away_goals: int) -> None:
        key = self._h2h_key(home, away)
        if home_goals > away_goals:
            result = 1 if home == key[0] else -1
        elif home_goals < away_goals:
            result = -1 if home == key[0] else 1
        else:
            result = 0
        self._h2h[key].append(result)

    def build(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Emit one feature row per historical match, updating state as it goes."""
        rows = []
        for row in matches.itertuples(index=False):
            home, away = row.home_team, row.away_team
            home_state = self._team_states[home]
            away_state = self._team_states[away]
            is_wc = str(getattr(row, "tournament", "")).lower().startswith("world cup")

            elo_diff = float(getattr(row, "home_elo_rating", np.nan)) - float(
                getattr(row, "away_elo_rating", np.nan)
            )
            rank_diff = float(getattr(row, "away_fifa_rank", np.nan)) - float(
                getattr(row, "home_fifa_rank", np.nan)
            )  # positive => home team ranked better (lower rank number)

            feature_row = {
                "date": row.date,
                "home_team": home,
                "away_team": away,
                "home_form": home_state.form_points(),
                "away_form": away_state.form_points(),
                "home_avg_goals_for": home_state.avg_goals_for(),
                "away_avg_goals_for": away_state.avg_goals_for(),
                "home_avg_goals_against": home_state.avg_goals_against(),
                "away_avg_goals_against": away_state.avg_goals_against(),
                "home_win_pct": home_state.win_pct(),
                "away_win_pct": away_state.win_pct(),
                "home_clean_sheet_pct": home_state.clean_sheet_pct(),
                "away_clean_sheet_pct": away_state.clean_sheet_pct(),
                "home_wc_experience": home_state.world_cup_experience(),
                "away_wc_experience": away_state.world_cup_experience(),
                "h2h_home_strength": self._h2h_strength(home, away),
                "elo_diff": elo_diff,
                "fifa_rank_diff": rank_diff,
                "outcome": row.outcome,
            }
            rows.append(feature_row)

            home_state.update(row.home_goals, row.away_goals, is_wc)
            away_state.update(row.away_goals, row.home_goals, is_wc)
            self._record_h2h(home, away, row.home_goals, row.away_goals)

        feature_df = pd.DataFrame(rows).dropna().reset_index(drop=True)
        logger.info("Built %d feature rows from %d raw matches", len(feature_df), len(matches))
        return feature_df

    def current_team_snapshot(self, teams: list[str]) -> pd.DataFrame:
        """Return each team's latest rolling stats -- used to seed simulation."""
        records = []
        for team in teams:
            state = self._team_states[team]
            records.append(
                {
                    "team": team,
                    "form": state.form_points(),
                    "avg_goals_for": state.avg_goals_for(),
                    "avg_goals_against": state.avg_goals_against(),
                    "win_pct": state.win_pct(),
                    "clean_sheet_pct": state.clean_sheet_pct(),
                    "wc_experience": state.world_cup_experience(),
                }
            )
        return pd.DataFrame(records)

    def h2h_strength(self, team: str, opponent: str) -> float:
        return self._h2h_strength(team, opponent)
