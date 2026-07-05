"""
train_model.py
===============
End-to-end training entrypoint:

    1. Generate sample data if raw data is missing (safe no-op otherwise).
    2. Build the leakage-free feature table.
    3. Train the primary model (XGBoost, or sklearn fallback) + LogReg baseline.
    4. Print/save an evaluation report.

Usage:
    python train_model.py
"""

from __future__ import annotations

from ingestion.generate_group_draw import generate_and_save as generate_group_draw
from ingestion.generate_sample_data import generate_and_save as generate_sample_data
from features.build_features import run as build_features
from features.match_outcome_model import MatchOutcomeModel
from utils.logger import get_logger

logger = get_logger(__name__, log_file="model_training.log")


def main() -> None:
    logger.info("Step 1/3: Ensuring raw data is available...")
    generate_sample_data()
    generate_group_draw()

    logger.info("Step 2/3: Building features...")
    feature_df, team_strength_df = build_features()

    logger.info("Step 3/3: Training model on %d matches...", len(feature_df))
    model = MatchOutcomeModel()
    result = model.train(feature_df)

    print("\n" + "=" * 70)
    print(f"Backend used            : {result.backend}")
    print(f"Baseline (LogReg) acc.  : {result.baseline_accuracy:.4f}")
    print(f"Primary model accuracy  : {result.test_accuracy:.4f}  (ensemble blend, weight={result.blend_weight:.2f})")
    print(f"  ...primary model alone: {result.primary_only_accuracy:.4f} (log-loss {result.primary_only_log_loss:.4f})")
    print(f"Primary model log-loss  : {result.test_log_loss:.4f}")
    print(f"Draw recall             : {result.draw_recall:.4f}")
    print(f"Macro-F1 (all 3 classes): {result.macro_f1:.4f}")
    if result.best_hyperparams:
        print(f"Tuned hyperparameters   : {result.best_hyperparams}")
    print("-" * 70)
    print(result.report)
    print("Top 5 most important features:")
    top5 = sorted(result.feature_importance.items(), key=lambda kv: kv[1], reverse=True)[:5]
    for name, score in top5:
        print(f"  {name:<28s} {score:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
