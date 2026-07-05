"""
prediction/predictor_service.py
================================
Thin orchestration layer used by the Streamlit app (and reusable by any other
frontend / notebook / test). Loads all artifacts once and exposes a simple
API: run_simulation(), explain_team(), single_match_prediction().
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from config import (
    DEFAULT_N_SIMULATIONS,
    FIXTURES_CSV,
    MODEL_METADATA_PATH,
    TEAM_STRENGTH_CSV,
)
from explainability.explainer import Explainer, TeamExplanation
from features.match_outcome_model import MatchOutcomeModel
from simulation.tournament_simulator import SimulationResult, TournamentSimulator
from utils.logger import get_logger

logger = get_logger(__name__, log_file="prediction.log")


@dataclass
class PredictorAssets:
    model: MatchOutcomeModel
    team_strength: pd.DataFrame
    group_draw: pd.DataFrame
    simulator: TournamentSimulator
    explainer: Explainer


def artifacts_available() -> bool:
    return (
        MODEL_METADATA_PATH.exists()
        and TEAM_STRENGTH_CSV.exists()
        and FIXTURES_CSV.exists()
    )


@lru_cache(maxsize=1)
def load_assets() -> PredictorAssets:
    """Load model + data once and cache for the lifetime of the process."""
    logger.info("Loading model and data artifacts...")
    model = MatchOutcomeModel()
    model.load()

    team_strength = pd.read_csv(TEAM_STRENGTH_CSV)
    group_draw = pd.read_csv(FIXTURES_CSV)

    simulator = TournamentSimulator(model, team_strength, group_draw)
    explainer = Explainer(team_strength)

    return PredictorAssets(
        model=model,
        team_strength=team_strength,
        group_draw=group_draw,
        simulator=simulator,
        explainer=explainer,
    )


def run_simulation(n_simulations: int = DEFAULT_N_SIMULATIONS, progress_callback=None) -> SimulationResult:
    assets = load_assets()
    return assets.simulator.run(n_simulations, progress_callback=progress_callback)


def explain_team(team: str, champion_probability: float) -> TeamExplanation:
    assets = load_assets()
    return assets.explainer.explain(team, champion_probability)


def single_match_prediction(team_a: str, team_b: str) -> tuple[float, float, float]:
    """Return (P(team_a win), P(draw), P(team_b win)) using current form."""
    assets = load_assets()
    return assets.simulator.match_probabilities(team_a, team_b)


def live_match_prediction(
    team_a: str, team_b: str, home_goals: int, away_goals: int, minute: int
) -> tuple[float, float, float]:
    """Pre-match probability adjusted for the current live scoreline/minute."""
    from prediction.live_match_predictor import live_adjusted_probabilities

    pre_p_home, pre_p_draw, pre_p_away = single_match_prediction(team_a, team_b)
    return live_adjusted_probabilities(pre_p_home, pre_p_draw, pre_p_away, home_goals, away_goals, minute)


def get_team_table() -> pd.DataFrame:
    return load_assets().team_strength.copy()


def get_group_draw() -> pd.DataFrame:
    return load_assets().group_draw.copy()


def get_model_metadata() -> dict:
    import json

    return json.loads(MODEL_METADATA_PATH.read_text())
