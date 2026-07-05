"""
config.py
=========
Central configuration for the FIFA World Cup Winner Predictor.

All tunable constants, file paths, and simulation parameters live here so the
rest of the codebase never hard-codes a path or a magic number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Raw data files (see ingestion/README.md for free-source download links)
TEAMS_CSV = RAW_DATA_DIR / "teams.csv"
HISTORICAL_MATCHES_CSV = RAW_DATA_DIR / "historical_matches.csv"
WORLD_CUP_RESULTS_CSV = RAW_DATA_DIR / "world_cup_results.csv"
FIXTURES_CSV = RAW_DATA_DIR / "tournament_fixtures.csv"

# Live-update inputs (see ingestion/live_updates.py)
MANUAL_LIVE_RESULTS_CSV = RAW_DATA_DIR / "live_results_manual.csv"
ACTUAL_RESULTS_CSV = RAW_DATA_DIR / "actual_tournament_results.csv"

# Processed artifacts
FEATURES_CSV = PROCESSED_DATA_DIR / "match_features.csv"
TEAM_STRENGTH_CSV = PROCESSED_DATA_DIR / "team_strength.csv"

# Model artifacts
XGB_MODEL_PATH = MODELS_DIR / "xgb_match_model.json"
BASELINE_MODEL_PATH = MODELS_DIR / "logreg_baseline.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Live score updates (free tier only, no paid plans)
# --------------------------------------------------------------------------
import os

# football-data.org free tier: 12 competitions (incl. World Cup, code "WC"),
# 10 requests/minute, free forever for these competitions. Get a free API
# key at https://www.football-data.org/client/register and set it as an
# environment variable -- no key is bundled with this project.
FOOTBALL_DATA_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
WC_COMPETITION_CODE = "WC"

# Standard Elo update K-factor applied when a real match result comes in.
# 20 is a common choice for international football (higher = more reactive).
ELO_K_FACTOR = 20.0

# --------------------------------------------------------------------------
# Simulation parameters
# --------------------------------------------------------------------------
DEFAULT_N_SIMULATIONS = 10_000
MAX_N_SIMULATIONS = 100_000

# Home-advantage nudge applied only to the host nation(s) in group play.
HOST_ADVANTAGE_ELO_BONUS = 60.0

# Draw probability model: draws are estimated from the absolute Elo gap
# between two sides using a simple logistic curve fitted to historical
# international-football draw rates (see features/match_outcome_model.py).
DRAW_BASE_RATE = 0.26

# --------------------------------------------------------------------------
# Feature engineering windows
# --------------------------------------------------------------------------
FORM_WINDOW = 5          # last-N-matches form
GOALS_WINDOW = 10        # matches used for scoring/conceding averages
H2H_MIN_MATCHES = 2      # minimum head-to-head matches to trust the signal

# --------------------------------------------------------------------------
# Model training
# --------------------------------------------------------------------------
TEST_SIZE = 0.2
XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "random_state": RANDOM_SEED,
}

# Fallback used automatically if the xgboost package is not installed.
FALLBACK_MODEL_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "max_iter": 400,
    "random_state": RANDOM_SEED,
}

# Whether to weight training samples inversely to class frequency (Home
# Win / Draw / Away Win are naturally imbalanced, with draws rarest).
# Measured trade-off on this project's sample data: draw recall roughly
# 10% -> 25% and macro-F1 improves, but raw accuracy and log-loss get
# SLIGHTLY worse (~2-3 points). Since the simulator consumes full
# probability triples (not just the predicted class), well-calibrated
# probabilities usually matter more than raw accuracy -- but if you'd
# rather optimize for accuracy/log-loss and don't mind the model rarely
# predicting draws, set this to False and retrain.
BALANCE_CLASSES = True

# Search a small hyperparameter grid (3-fold CV, optimizing log-loss) before
# the final fit, instead of using the fixed XGB_PARAMS/FALLBACK_MODEL_PARAMS
# above. Off by default because it multiplies training time; the measured
# gain on this project's data is modest (a couple points of log-loss) but
# real and free to enable.
TUNE_HYPERPARAMS = True
HYPERPARAM_CV_FOLDS = 3

# Blend the primary model's probabilities with the Logistic Regression
# baseline's, using a weight tuned on a held-out validation split (never the
# reported test set, to keep the final metrics honest). Averaging two
# differently-biased models is a simple, well-established way to reduce
# variance -- see the "Getting more accurate predictions" section of the
# README for what this measurably changes on this project's data.
ENABLE_ENSEMBLE_BLEND = True
VAL_SIZE = 0.1  # carved out of the training split, on top of TEST_SIZE

# --------------------------------------------------------------------------
# Outcome label encoding (shared everywhere so it never drifts)
# --------------------------------------------------------------------------
OUTCOME_HOME_WIN = 0
OUTCOME_DRAW = 1
OUTCOME_AWAY_WIN = 2
OUTCOME_LABELS = {
    OUTCOME_HOME_WIN: "Home Win",
    OUTCOME_DRAW: "Draw",
    OUTCOME_AWAY_WIN: "Away Win",
}

# --------------------------------------------------------------------------
# Tournament structure (2026 FIFA World Cup: 48 teams, 12 groups of 4)
# --------------------------------------------------------------------------
N_GROUPS = 12
TEAMS_PER_GROUP = 4
GROUP_NAMES = [chr(ord("A") + i) for i in range(N_GROUPS)]  # A..L

# Knockout bracket size after group stage: top 2 per group (24) + 8 best
# third-placed teams = 32, matching the official 2026 format.
ROUND_OF_32_SIZE = 32

HOST_NATIONS = ["United States", "Canada", "Mexico"]

# --------------------------------------------------------------------------
# App metadata
# --------------------------------------------------------------------------
APP_TITLE = "FIFA World Cup Winner Predictor"
APP_ICON = "\U0001F3C6"  # trophy emoji
