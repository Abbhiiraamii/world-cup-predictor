"""
features/match_outcome_model.py
================================
Match-outcome classifier: Home Win / Draw / Away Win.

Primary model: XGBoost (multi:softprob).
Baseline model: Logistic Regression, trained for comparison/reporting.
Fallback: if the ``xgboost`` package is not installed, we transparently fall
back to scikit-learn's ``HistGradientBoostingClassifier`` (very similar
gradient-boosted-tree algorithm) so the project still runs anywhere with only
``requirements.txt`` partially satisfied. The active backend is always
recorded in ``model_metadata.json`` so results stay explainable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from config import (
    BALANCE_CLASSES,
    BASELINE_MODEL_PATH,
    ENABLE_ENSEMBLE_BLEND,
    FALLBACK_MODEL_PARAMS,
    HYPERPARAM_CV_FOLDS,
    MODEL_METADATA_PATH,
    OUTCOME_LABELS,
    RANDOM_SEED,
    TEST_SIZE,
    TUNE_HYPERPARAMS,
    VAL_SIZE,
    XGB_MODEL_PATH,
    XGB_PARAMS,
)
from utils.logger import get_logger

logger = get_logger(__name__, log_file="model_training.log")

FEATURE_COLUMNS = [
    "home_form", "away_form",
    "home_avg_goals_for", "away_avg_goals_for",
    "home_avg_goals_against", "away_avg_goals_against",
    "home_win_pct", "away_win_pct",
    "home_clean_sheet_pct", "away_clean_sheet_pct",
    "home_wc_experience", "away_wc_experience",
    "h2h_home_strength", "elo_diff", "fifa_rank_diff",
]

try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover - exercised only when xgboost is absent
    _HAS_XGBOOST = False


@dataclass
class TrainingResult:
    backend: str
    test_accuracy: float
    test_log_loss: float
    baseline_accuracy: float
    report: str
    feature_importance: dict[str, float]
    draw_recall: float
    macro_f1: float
    blend_weight: float
    primary_only_accuracy: float
    primary_only_log_loss: float
    best_hyperparams: dict | None = None


class MatchOutcomeModel:
    """Thin wrapper unifying XGBoost / sklearn-fallback prediction APIs."""

    def __init__(self) -> None:
        self.model: Any = None
        self.baseline: LogisticRegression | None = None
        self.scaler = StandardScaler()
        self.backend = "xgboost" if _HAS_XGBOOST else "sklearn_hist_gbm"
        self.blend_weight = 1.0  # 1.0 = primary model only, tuned during training if enabled

    # ------------------------------------------------------------------ train
    def train(self, feature_df: pd.DataFrame, tune: bool | None = None) -> TrainingResult:
        """Train the model.

        Args:
            feature_df: leakage-free feature table from ``features/build_features.py``.
            tune: whether to run the hyperparameter grid search (~30-60s on
                this dataset size). Defaults to ``config.TUNE_HYPERPARAMS``.
                Callers doing frequent incremental retrains (e.g. after every
                live-score update) should pass ``tune=False`` to keep those
                fast; run ``python train_model.py`` (which uses the config
                default) periodically to get the fully-tuned model.
        """
        tune = TUNE_HYPERPARAMS if tune is None else tune
        X = feature_df[FEATURE_COLUMNS].values
        y = feature_df["outcome"].values

        # Three-way split: train / val / test. `val` is used ONLY to tune the
        # ensemble blend weight below -- the reported test_accuracy/log_loss
        # never touch data used for any tuning decision, so they stay honest.
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y_train_full
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Draws are structurally the hardest outcome to call (see README for
        # why ~100% accuracy isn't a realistic target for football at all)
        # and are naturally under-represented, so the model tends to almost
        # never predict them. Class-balanced sample weights push back on
        # that specifically -- see config.BALANCE_CLASSES for the measured
        # trade-off (better draw recall & macro-F1, marginally worse raw
        # accuracy/log-loss).
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train) if BALANCE_CLASSES else None

        # ---- baseline: logistic regression -------------------------------
        self.baseline = LogisticRegression(max_iter=1000)
        self.baseline.fit(X_train_scaled, y_train, sample_weight=sample_weight)
        baseline_preds = self.baseline.predict(X_test_scaled)
        baseline_acc = accuracy_score(y_test, baseline_preds)
        logger.info("Baseline LogisticRegression test accuracy: %.4f", baseline_acc)

        # ---- primary: xgboost (or fallback), optionally hyperparameter-tuned
        best_params = None
        if tune:
            best_params = self._tune_hyperparameters(X_train, y_train, sample_weight)
            logger.info("Hyperparameter search selected: %s", best_params)

        if _HAS_XGBOOST:
            params = {**XGB_PARAMS, **(best_params or {})}
            self.model = xgb.XGBClassifier(**params)
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
            val_proba = self.model.predict_proba(X_val)
            test_proba_primary = self.model.predict_proba(X_test)
            importances = dict(zip(FEATURE_COLUMNS, self.model.feature_importances_.tolist()))
        else:
            logger.warning(
                "xgboost is not installed; falling back to "
                "sklearn.HistGradientBoostingClassifier. Install xgboost for "
                "the primary model described in the PRD."
            )
            params = {**FALLBACK_MODEL_PARAMS, **(best_params or {})}
            self.model = HistGradientBoostingClassifier(**params)
            self.model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
            val_proba = self.model.predict_proba(X_val_scaled)
            test_proba_primary = self.model.predict_proba(X_test_scaled)
            importances = self._permutation_importance(X_test_scaled, y_test)

        # ---- ensemble blend weight, tuned on the VALIDATION split only -----
        if ENABLE_ENSEMBLE_BLEND:
            val_proba_baseline = self.baseline.predict_proba(X_val_scaled)
            self.blend_weight = self._tune_blend_weight(val_proba, val_proba_baseline, y_val)
        else:
            self.blend_weight = 1.0
        logger.info("Ensemble blend weight (1.0 = primary model only): %.2f", self.blend_weight)

        test_proba_baseline = self.baseline.predict_proba(X_test_scaled)
        proba = self.blend_weight * test_proba_primary + (1 - self.blend_weight) * test_proba_baseline
        preds = np.argmax(proba, axis=1)
        primary_only_preds = np.argmax(test_proba_primary, axis=1)

        test_acc = accuracy_score(y_test, preds)
        test_ll = log_loss(y_test, proba, labels=[0, 1, 2])
        primary_only_acc = accuracy_score(y_test, primary_only_preds)
        primary_only_ll = log_loss(y_test, test_proba_primary, labels=[0, 1, 2])

        report = classification_report(
            y_test, preds, target_names=[OUTCOME_LABELS[i] for i in range(3)]
        )
        # Draws are the outcome the model most often ignores entirely (predicting
        # "draw" is the riskiest guess), so we track its recall specifically --
        # and macro-F1 (unweighted across all 3 classes) as a single number that
        # reflects being *right about draws* rather than just "right most often".
        draw_recall = recall_score(y_test, preds, labels=[0, 1, 2], average=None)[1]
        macro_f1 = f1_score(y_test, preds, average="macro")
        logger.info(
            "Final (blended) test accuracy: %.4f | log-loss: %.4f | draw recall: %.4f | macro-F1: %.4f "
            "(primary-only was accuracy: %.4f | log-loss: %.4f)",
            test_acc, test_ll, draw_recall, macro_f1, primary_only_acc, primary_only_ll,
        )

        self._save(importances, test_acc, test_ll, baseline_acc, draw_recall, macro_f1, best_params, tune)

        return TrainingResult(
            backend=self.backend,
            test_accuracy=test_acc,
            test_log_loss=test_ll,
            baseline_accuracy=baseline_acc,
            report=report,
            feature_importance=importances,
            draw_recall=draw_recall,
            macro_f1=macro_f1,
            blend_weight=self.blend_weight,
            primary_only_accuracy=primary_only_acc,
            primary_only_log_loss=primary_only_ll,
            best_hyperparams=best_params,
        )

    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, sample_weight) -> dict:
        """Small grid search (log-loss scored, cross-validated) over the most
        impactful gradient-boosting hyperparameters. Works for either backend
        since XGBClassifier and HistGradientBoostingClassifier are both
        scikit-learn-API compatible."""
        from sklearn.model_selection import GridSearchCV

        if _HAS_XGBOOST:
            estimator = xgb.XGBClassifier(
                objective="multi:softprob", num_class=3, eval_metric="mlogloss", random_state=RANDOM_SEED
            )
            param_grid = {
                "max_depth": [3, 4, 6],
                "learning_rate": [0.03, 0.05, 0.1],
                "n_estimators": [200, 400],
            }
        else:
            estimator = HistGradientBoostingClassifier(random_state=RANDOM_SEED)
            param_grid = {
                "max_depth": [3, 4, 6],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_iter": [200, 400],
            }

        search = GridSearchCV(
            estimator, param_grid, scoring="neg_log_loss", cv=HYPERPARAM_CV_FOLDS, n_jobs=-1
        )
        search.fit(X_train, y_train, sample_weight=sample_weight)
        return search.best_params_

    def _tune_blend_weight(
        self, val_proba_primary: np.ndarray, val_proba_baseline: np.ndarray, y_val: np.ndarray
    ) -> float:
        """Sweep the primary-vs-baseline blend weight on the validation split
        and return whichever minimizes log-loss (a proper scoring rule for
        probabilistic predictions, which is what the simulator consumes)."""
        best_weight, best_ll = 1.0, float("inf")
        for w in np.arange(0.0, 1.01, 0.05):
            blended = w * val_proba_primary + (1 - w) * val_proba_baseline
            ll = log_loss(y_val, blended, labels=[0, 1, 2])
            if ll < best_ll:
                best_ll, best_weight = ll, float(w)
        return best_weight

    def _permutation_importance(self, X_test_scaled: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            self.model, X_test_scaled, y_test, n_repeats=5, random_state=RANDOM_SEED
        )
        return dict(zip(FEATURE_COLUMNS, result.importances_mean.tolist()))

    # -------------------------------------------------------------- predict
    def predict_proba(self, feature_row: dict[str, float]) -> np.ndarray:
        """Return [P(home win), P(draw), P(away win)] for one match."""
        return self.predict_proba_batch([feature_row])[0]

    def predict_proba_batch(self, feature_rows: list[dict[str, float]]) -> np.ndarray:
        """Vectorized prediction for many matches at once (fast path).

        Simulating tens of thousands of tournaments one match-prediction at a
        time is far too slow (Python/sklearn call overhead dominates). Since
        "current form" features are static for a given pair of teams during a
        simulation run, callers should precompute every needed pair ONCE with
        this batch method and cache the results before running Monte Carlo
        trials -- see ``simulation/tournament_simulator.py``.

        Returns the ensemble blend (primary model + Logistic Regression
        baseline, weighted by ``self.blend_weight`` as tuned during training)
        rather than the primary model alone, unless blending is disabled
        (``config.ENABLE_ENSEMBLE_BLEND = False``), in which case
        ``blend_weight`` is fixed at 1.0 and this reduces to the primary
        model's raw output.
        """
        X = np.array([[row[c] for c in FEATURE_COLUMNS] for row in feature_rows])
        X_scaled = self.scaler.transform(X)

        if _HAS_XGBOOST and self.backend == "xgboost":
            proba_primary = self.model.predict_proba(X)
        else:
            proba_primary = self.model.predict_proba(X_scaled)

        if self.blend_weight >= 1.0:
            return proba_primary

        proba_baseline = self.baseline.predict_proba(X_scaled)
        return self.blend_weight * proba_primary + (1 - self.blend_weight) * proba_baseline

    # ------------------------------------------------------------ persistence
    def _save(self, importances: dict[str, float], test_acc: float,
              test_ll: float, baseline_acc: float, draw_recall: float, macro_f1: float,
              best_hyperparams: dict | None, hyperparams_tuned: bool) -> None:
        if _HAS_XGBOOST and self.backend == "xgboost":
            self.model.save_model(str(XGB_MODEL_PATH))
        else:
            joblib.dump(self.model, str(XGB_MODEL_PATH) + ".sklearn.joblib")
        joblib.dump({"model": self.baseline, "scaler": self.scaler}, BASELINE_MODEL_PATH)

        metadata = {
            "backend": self.backend,
            "feature_columns": FEATURE_COLUMNS,
            "test_accuracy": test_acc,
            "test_log_loss": test_ll,
            "baseline_accuracy": baseline_acc,
            "draw_recall": draw_recall,
            "macro_f1": macro_f1,
            "class_balanced_training": BALANCE_CLASSES,
            "blend_weight": self.blend_weight,
            "hyperparams_tuned": hyperparams_tuned,
            "best_hyperparams": best_hyperparams,
            "feature_importance": importances,
        }
        MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
        logger.info("Saved model artifacts and metadata to %s", MODEL_METADATA_PATH.parent)

    def load(self) -> None:
        metadata = json.loads(MODEL_METADATA_PATH.read_text())
        self.backend = metadata["backend"]
        self.blend_weight = metadata.get("blend_weight", 1.0)
        baseline_bundle = joblib.load(BASELINE_MODEL_PATH)
        self.baseline = baseline_bundle["model"]
        self.scaler = baseline_bundle["scaler"]

        if self.backend == "xgboost":
            params = {**XGB_PARAMS, **(metadata.get("best_hyperparams") or {})}
            self.model = xgb.XGBClassifier(**params)
            self.model.load_model(str(XGB_MODEL_PATH))
        else:
            self.model = joblib.load(str(XGB_MODEL_PATH) + ".sklearn.joblib")
