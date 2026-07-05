"""Unit tests for simulation/tournament_simulator.py using a tiny fake model
so tests run fast and deterministically without needing a trained model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simulation.tournament_simulator import TournamentSimulator


class _FakeModel:
    """Deterministic stand-in for MatchOutcomeModel: stronger Elo always wins
    group matches with high probability, no draws favoured either way."""

    def predict_proba_batch(self, feature_rows: list[dict]) -> np.ndarray:
        out = []
        for row in feature_rows:
            elo_diff = row["elo_diff"]
            # Simple logistic-ish mapping from elo_diff to win probability.
            p_home = 1 / (1 + 10 ** (-elo_diff / 400))
            p_draw = 0.2
            p_home_adj = p_home * (1 - p_draw)
            p_away_adj = 1 - p_draw - p_home_adj
            out.append([p_home_adj, p_draw, p_away_adj])
        return np.array(out)


def _make_team_strength(n_teams: int = 48) -> pd.DataFrame:
    teams = [f"Team_{i:02d}" for i in range(n_teams)]
    rows = []
    for i, team in enumerate(teams):
        rows.append({
            "team": team,
            "elo_rating": 2000 - i * 10,  # strictly decreasing strength
            "fifa_rank": i + 1,
            "form": 0.5, "avg_goals_for": 1.3, "avg_goals_against": 1.1,
            "win_pct": 0.4, "clean_sheet_pct": 0.3, "wc_experience": 0.2,
        })
    return pd.DataFrame(rows)


def _make_group_draw(team_strength: pd.DataFrame) -> pd.DataFrame:
    teams = team_strength["team"].tolist()
    groups = []
    group_names = [chr(ord("A") + i) for i in range(12)]
    for i, team in enumerate(teams):
        groups.append({"group": group_names[i % 12], "team": team})
    return pd.DataFrame(groups)


@pytest.fixture
def simulator() -> TournamentSimulator:
    team_strength = _make_team_strength()
    group_draw = _make_group_draw(team_strength)
    return TournamentSimulator(_FakeModel(), team_strength, group_draw, seed=123)


def test_champion_probabilities_sum_to_one(simulator: TournamentSimulator):
    result = simulator.run(n_simulations=200)
    total = result.champion_prob.sum()
    assert total == pytest.approx(1.0, abs=1e-9)


def test_strongest_team_is_favourite(simulator: TournamentSimulator):
    """Team_00 has the highest Elo in every match-up, so across many trials it
    should have the highest champion probability."""
    result = simulator.run(n_simulations=500)
    assert result.champion_prob.index[0] == "Team_00"


def test_stage_reach_probabilities_are_monotonically_non_increasing(simulator: TournamentSimulator):
    """A team can't reach the Final more often than it reaches the Semifinal, etc."""
    result = simulator.run(n_simulations=300)
    stages = ["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final", "Champion"]
    for team in result.stage_reach_prob.index[:5]:
        values = result.stage_reach_prob.loc[team, stages].values
        assert all(values[i] >= values[i + 1] - 1e-9 for i in range(len(values) - 1))


def test_reproducible_with_same_seed():
    team_strength = _make_team_strength()
    group_draw = _make_group_draw(team_strength)
    sim1 = TournamentSimulator(_FakeModel(), team_strength, group_draw, seed=99)
    sim2 = TournamentSimulator(_FakeModel(), team_strength, group_draw, seed=99)

    result1 = sim1.run(n_simulations=100)
    result2 = sim2.run(n_simulations=100)

    pd.testing.assert_series_equal(result1.champion_prob, result2.champion_prob)
