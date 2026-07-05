"""Unit tests for features/match_outcome_model.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.match_outcome_model import FEATURE_COLUMNS, MatchOutcomeModel


def _synthetic_feature_df(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Small synthetic dataset where elo_diff strongly predicts the outcome,
    so a trained model should beat random-guess accuracy comfortably."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        elo_diff = rng.normal(0, 300)
        p_home = 1 / (1 + 10 ** (-elo_diff / 400))
        u = rng.random()
        outcome = 0 if u < p_home * 0.8 else (1 if u < p_home * 0.8 + 0.2 else 2)
        rows.append({
            "home_form": rng.uniform(0, 1), "away_form": rng.uniform(0, 1),
            "home_avg_goals_for": rng.uniform(0.5, 2.5), "away_avg_goals_for": rng.uniform(0.5, 2.5),
            "home_avg_goals_against": rng.uniform(0.5, 2.5), "away_avg_goals_against": rng.uniform(0.5, 2.5),
            "home_win_pct": rng.uniform(0.2, 0.7), "away_win_pct": rng.uniform(0.2, 0.7),
            "home_clean_sheet_pct": rng.uniform(0, 0.5), "away_clean_sheet_pct": rng.uniform(0, 0.5),
            "home_wc_experience": rng.uniform(0, 1), "away_wc_experience": rng.uniform(0, 1),
            "h2h_home_strength": 0.5,
            "elo_diff": elo_diff,
            "fifa_rank_diff": rng.normal(0, 20),
            "outcome": outcome,
        })
    return pd.DataFrame(rows)


def test_train_returns_valid_result():
    df = _synthetic_feature_df()
    model = MatchOutcomeModel()
    result = model.train(df, tune=False)

    assert 0.0 <= result.test_accuracy <= 1.0
    assert 0.0 <= result.baseline_accuracy <= 1.0
    assert result.test_log_loss > 0
    assert set(result.feature_importance.keys()) == set(FEATURE_COLUMNS)


def test_model_beats_random_guessing():
    df = _synthetic_feature_df(n=800)
    model = MatchOutcomeModel()
    result = model.train(df, tune=False)
    # 3-class random guessing baseline is ~0.33; elo_diff is a strong signal here.
    assert result.test_accuracy > 0.4


def test_predict_proba_sums_to_one():
    df = _synthetic_feature_df()
    model = MatchOutcomeModel()
    model.train(df, tune=False)

    sample_row = {col: df.iloc[0][col] for col in FEATURE_COLUMNS}
    proba = model.predict_proba(sample_row)
    assert len(proba) == 3
    assert proba.sum() == pytest.approx(1.0, abs=1e-6)


def test_predict_proba_batch_matches_single_predictions():
    df = _synthetic_feature_df()
    model = MatchOutcomeModel()
    model.train(df, tune=False)

    rows = [{col: df.iloc[i][col] for col in FEATURE_COLUMNS} for i in range(5)]
    batch_result = model.predict_proba_batch(rows)
    for i, row in enumerate(rows):
        single_result = model.predict_proba(row)
        np.testing.assert_allclose(batch_result[i], single_result, atol=1e-6)


def test_hyperparameter_tuning_produces_valid_params():
    """Slower path (grid search) -- kept to a small dataset so it still runs
    quickly, just verifying it doesn't crash and returns usable params."""
    df = _synthetic_feature_df(n=300)
    model = MatchOutcomeModel()
    result = model.train(df, tune=True)
    assert result.best_hyperparams is not None
    assert len(result.best_hyperparams) > 0


def test_ensemble_blend_weight_is_between_zero_and_one():
    df = _synthetic_feature_df()
    model = MatchOutcomeModel()
    result = model.train(df, tune=False)
    assert 0.0 <= result.blend_weight <= 1.0
    assert 0.0 <= model.blend_weight <= 1.0


def test_blended_test_metrics_are_reported_alongside_primary_only():
    df = _synthetic_feature_df()
    model = MatchOutcomeModel()
    result = model.train(df, tune=False)
    assert 0.0 <= result.primary_only_accuracy <= 1.0
    assert result.primary_only_log_loss > 0
