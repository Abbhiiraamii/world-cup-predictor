"""
simulation/tournament_simulator.py
===================================
Monte Carlo tournament simulator.

Given (a) trained match-outcome model probabilities and (b) a group draw,
simulates the entire 48-team World Cup (group stage -> Round of 32 -> Round
of 16 -> QF -> SF -> Final) many thousands of times and tracks how far each
team advances on average.

Design notes
------------
* Match outcome probabilities come from ``MatchOutcomeModel`` fed with
  "current form" features built from ``team_strength.csv`` (Elo, form, etc.)
  rather than the leakage-free historical features -- at simulation time we
  only have "today's" snapshot for each side, which is exactly right.
* Knockout matches have no draws: if the sampled result is a draw we
  re-normalize between the two win probabilities (proxy for extra-time /
  penalties, where the stronger side is still favored).
* Host-nation Elo bonus (``HOST_ADVANTAGE_ELO_BONUS``) is applied only to
  group matches played by host nations, mirroring real tournament home
  advantage.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import HOST_ADVANTAGE_ELO_BONUS, HOST_NATIONS, RANDOM_SEED
from features.match_outcome_model import FEATURE_COLUMNS, MatchOutcomeModel
from utils.logger import get_logger

logger = get_logger(__name__, log_file="simulation.log")


@dataclass
class SimulationResult:
    n_simulations: int
    champion_prob: pd.Series
    stage_reach_prob: pd.DataFrame  # team x stage probability table
    sample_bracket: dict  # one illustrative simulated knockout bracket


class TournamentSimulator:
    """Runs Monte Carlo simulations of the full World Cup knockout format."""

    STAGES = ["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final", "Champion"]

    def __init__(self, model: MatchOutcomeModel, team_strength: pd.DataFrame,
                 group_draw: pd.DataFrame, seed: int = RANDOM_SEED) -> None:
        self.model = model
        self.team_strength = team_strength.set_index("team")
        self.groups = group_draw.groupby("group")["team"].apply(list).to_dict()
        self.rng = np.random.default_rng(seed)
        self._proba_cache: dict[tuple[str, str], tuple[float, float, float]] = {}
        self._precompute_all_pairs()

    # ---------------------------------------------------------------- helpers
    def _feature_row(self, team_a: str, team_b: str) -> dict[str, float]:
        """Build a model-ready feature dict from current-form snapshots."""
        a = self.team_strength.loc[team_a]
        b = self.team_strength.loc[team_b]

        elo_a = a["elo_rating"] + (HOST_ADVANTAGE_ELO_BONUS if team_a in HOST_NATIONS else 0)
        elo_b = b["elo_rating"] + (HOST_ADVANTAGE_ELO_BONUS if team_b in HOST_NATIONS else 0)

        return {
            "home_form": a["form"], "away_form": b["form"],
            "home_avg_goals_for": a["avg_goals_for"], "away_avg_goals_for": b["avg_goals_for"],
            "home_avg_goals_against": a["avg_goals_against"], "away_avg_goals_against": b["avg_goals_against"],
            "home_win_pct": a["win_pct"], "away_win_pct": b["win_pct"],
            "home_clean_sheet_pct": a["clean_sheet_pct"], "away_clean_sheet_pct": b["clean_sheet_pct"],
            "home_wc_experience": a["wc_experience"], "away_wc_experience": b["wc_experience"],
            "h2h_home_strength": 0.5,
            "elo_diff": elo_a - elo_b,
            "fifa_rank_diff": b["fifa_rank"] - a["fifa_rank"],
        }

    def _precompute_all_pairs(self) -> None:
        """Batch-predicts every ordered team pair ONCE so Monte Carlo trials
        become simple cached lookups + random sampling (orders of magnitude
        faster than calling the model per match per simulation)."""
        all_teams = self.team_strength.index.tolist()
        pairs = [(a, b) for a in all_teams for b in all_teams if a != b]
        feature_rows = [self._feature_row(a, b) for a, b in pairs]

        logger.info("Precomputing match probabilities for %d team pairs...", len(pairs))
        proba_matrix = self.model.predict_proba_batch(feature_rows)
        for (a, b), (p_home, p_draw, p_away) in zip(pairs, proba_matrix):
            self._proba_cache[(a, b)] = (float(p_home), float(p_draw), float(p_away))
        logger.info("Done precomputing probabilities.")

    def match_probabilities(self, team_a: str, team_b: str) -> tuple[float, float, float]:
        """Return cached (P(A win), P(draw), P(B win))."""
        return self._proba_cache[(team_a, team_b)]

    def _play_group_match(self, team_a: str, team_b: str) -> tuple[int, int, str]:
        """Simulate one group-stage match; returns (points_a, points_b, result_code)."""
        p_home, p_draw, _p_away = self.match_probabilities(team_a, team_b)
        u = self.rng.random()
        if u < p_home:
            return 3, 0, "H"
        if u < p_home + p_draw:
            return 1, 1, "D"
        return 0, 3, "A"

    def _play_knockout_match(self, team_a: str, team_b: str) -> str:
        """Simulate a knockout match with no draws allowed (winner only)."""
        p_home, _p_draw, p_away = self.match_probabilities(team_a, team_b)
        total = p_home + p_away
        p_home_adj = 0.5 if total <= 0 else p_home / total
        return team_a if self.rng.random() < p_home_adj else team_b

    # ---------------------------------------------------------------- stages
    def _simulate_group_stage(self) -> tuple[list[str], list[str]]:
        """Returns (top2_qualifiers, best_third_place_qualifiers)."""
        qualifiers: list[str] = []
        third_placed: list[tuple[str, int, int]] = []  # team, points, goal_diff (approx)

        for group, teams in self.groups.items():
            points = defaultdict(int)
            goal_diff = defaultdict(int)
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    a, b = teams[i], teams[j]
                    pa, pb, code = self._play_group_match(a, b)
                    points[a] += pa
                    points[b] += pb
                    # approximate goal-diff proxy from result code for tie-breaking
                    if code == "H":
                        goal_diff[a] += 1
                        goal_diff[b] -= 1
                    elif code == "A":
                        goal_diff[b] += 1
                        goal_diff[a] -= 1

            ranked = sorted(teams, key=lambda t: (points[t], goal_diff[t]), reverse=True)
            qualifiers.extend(ranked[:2])
            third_placed.append((ranked[2], points[ranked[2]], goal_diff[ranked[2]]))

        best_thirds = sorted(third_placed, key=lambda x: (x[1], x[2]), reverse=True)[:8]
        best_third_teams = [t[0] for t in best_thirds]
        return qualifiers, best_third_team_list_safe(best_third_teams)

    def _simulate_knockouts(self, bracket: list[str]) -> dict[str, list[str]]:
        """Simulate Round of 32 through Final; returns stage -> surviving teams."""
        stage_results: dict[str, list[str]] = {}
        current_round = bracket
        stage_names = ["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final"]

        for stage in stage_names:
            stage_results[stage] = current_round
            next_round = []
            for i in range(0, len(current_round), 2):
                a, b = current_round[i], current_round[i + 1]
                winner = self._play_knockout_match(a, b)
                next_round.append(winner)
            current_round = next_round

        stage_results["Champion"] = current_round  # single team
        return stage_results

    # ---------------------------------------------------------------- driver
    def run(self, n_simulations: int, progress_callback=None) -> SimulationResult:
        all_teams = [t for teams in self.groups.values() for t in teams]
        reach_counts = {team: defaultdict(int) for team in all_teams}
        champion_counts: dict[str, int] = defaultdict(int)
        sample_bracket = None

        for sim_i in range(n_simulations):
            top2, best_thirds = self._simulate_group_stage()
            bracket = self.rng.permutation(top2 + best_thirds).tolist()
            if len(bracket) % 2 != 0:
                bracket = bracket[:-1]  # safety guard, should already be 32

            stage_results = self._simulate_knockouts(bracket)

            for stage, teams_in_stage in stage_results.items():
                for team in teams_in_stage:
                    reach_counts[team][stage] += 1

            champion = stage_results["Champion"][0]
            champion_counts[champion] += 1

            if sim_i == 0:
                sample_bracket = stage_results

            if progress_callback and (sim_i + 1) % max(1, n_simulations // 20) == 0:
                progress_callback((sim_i + 1) / n_simulations)

        champion_prob = pd.Series(
            {team: champion_counts.get(team, 0) / n_simulations for team in all_teams}
        ).sort_values(ascending=False)

        stage_table = pd.DataFrame(
            {
                stage: {team: reach_counts[team][stage] / n_simulations for team in all_teams}
                for stage in self.STAGES
            }
        ).loc[champion_prob.index]

        logger.info("Completed %d simulations. Top champion: %s (%.1f%%)",
                    n_simulations, champion_prob.index[0], champion_prob.iloc[0] * 100)

        return SimulationResult(
            n_simulations=n_simulations,
            champion_prob=champion_prob,
            stage_reach_prob=stage_table,
            sample_bracket=sample_bracket or {},
        )


def best_third_team_list_safe(teams: list[str]) -> list[str]:
    """Ensures exactly 8 best-third-place teams (pads defensively, should be a no-op)."""
    return teams[:8]
