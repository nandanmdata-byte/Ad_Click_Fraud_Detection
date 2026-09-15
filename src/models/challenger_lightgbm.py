"""
Train a simple LightGBM challenger focused on improving PR-AUC.

This script keeps the current production model untouched. It uses chronological
splits, adds a few training-only frequency features, tests three small LightGBM
configurations, calibrates the winner, and saves separate challenger artifacts.

"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "development_features.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "challenger_lightgbm.joblib"
CALIBRATOR_PATH = PROJECT_ROOT / "models" / "challenger_calibrator.joblib"
FREQUENCY_MAPS_PATH = PROJECT_ROOT / "models" / "challenger_frequency_maps.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "challenger_metadata.json"

CALIBRATION_START = pd.Timestamp("2017-11-07 14:30:00")
EVALUATION_START = pd.Timestamp("2017-11-07 14:45:00")
EVALUATION_END = pd.Timestamp("2017-11-07 15:00:00")

TARGET = "is_attributed"
TIME_COLUMN = "click_time"

BASE_FEATURES = [
    "ip", "app", "device", "os", "channel",
    "click_day", "click_hour", "click_minute", "click_second",
    "click_day_of_week", "previous_ip_clicks",
    "previous_ip_app_clicks", "previous_ip_channel_clicks",
    "seconds_since_previous_ip_click",
    "seconds_since_previous_ip_app_click",
    "seconds_since_previous_ip_channel_click",
]

# These frequency features are learned from training rows only.
FREQUENCY_GROUPS = {
    "app_frequency": ["app"],
    "channel_frequency": ["channel"],
    "os_frequency": ["os"],
    "device_frequency": ["device"],
    "app_channel_frequency": ["app", "channel"],
    "app_os_frequency": ["app", "os"],
}

# A deliberately small search: useful variation without overcomplicating it.
CANDIDATES = [
    {"num_leaves": 31, "min_child_samples": 100, "feature_fraction": 0.90},
    {"num_leaves": 63, "min_child_samples": 200, "feature_fraction": 0.90},
    {"num_leaves": 47, "min_child_samples": 300, "feature_fraction": 0.80},
]


def add_frequency_features(train, calibration, evaluation):
    """Map counts learned only from training history into every split."""

    frequency_maps = {}
    for new_column, group_columns in FREQUENCY_GROUPS.items():
        counts = train.groupby(group_columns, observed=True).size()
        frequency_maps[new_column] = {
            "group_columns": group_columns,
            "counts": counts,
        }

        for frame in (train, calibration, evaluation):
            if len(group_columns) == 1:
                values = frame[group_columns[0]].map(counts)
            else:
                index = pd.MultiIndex.from_frame(frame[group_columns])
                values = pd.Series(counts.reindex(index).to_numpy(), index=frame.index)

            frame[new_column] = values.fillna(0).astype("uint32")

    return frequency_maps


def best_f1_threshold(y_true, probabilities):
    """Return the threshold with the highest F1 on the evaluation split."""

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    scores = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    best_position = int(np.argmax(scores))
    return float(thresholds[best_position])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Feature file not found: {DATA_PATH}")

    print(f"Reading features: {DATA_PATH}")
    data = pd.read_parquet(DATA_PATH)

    required = set(BASE_FEATURES + [TARGET, TIME_COLUMN])
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data.sort_values(TIME_COLUMN).reset_index(drop=True)

    train = data[data[TIME_COLUMN] < CALIBRATION_START].copy()
    calibration = data[
        (data[TIME_COLUMN] >= CALIBRATION_START)
        & (data[TIME_COLUMN] < EVALUATION_START)
    ].copy()
    evaluation = data[
        (data[TIME_COLUMN] >= EVALUATION_START)
        & (data[TIME_COLUMN] < EVALUATION_END)
    ].copy()

    if min(len(train), len(calibration), len(evaluation)) == 0:
        raise ValueError("One or more chronological splits are empty")

    print("Adding training-only frequency features...")
    frequency_maps = add_frequency_features(train, calibration, evaluation)

    features = BASE_FEATURES + list(FREQUENCY_GROUPS)
    categorical = ["app", "device", "os", "channel"]

    X_train, y_train = train[features], train[TARGET]
    X_cal, y_cal = calibration[features], calibration[TARGET]
    X_eval, y_eval = evaluation[features], evaluation[TARGET]

    best_model = None
    best_parameters = None
    best_pr_auc = -1.0
    candidate_results = []

    print("\nTesting three small LightGBM candidates...")
    for number, candidate in enumerate(CANDIDATES, start=1):
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=1200,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            **candidate,
        )

        model.fit(
            X_train,
            y_train,
            categorical_feature=categorical,
            eval_set=[(X_eval, y_eval)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(75, verbose=False)],
        )

        probabilities = model.predict_proba(
            X_eval,
            num_iteration=model.best_iteration_,
        )[:, 1]

        pr_auc = float(average_precision_score(y_eval, probabilities))
        roc_auc = float(roc_auc_score(y_eval, probabilities))
        result = {
            "candidate": number,
            "parameters": candidate,
            "best_iteration": int(model.best_iteration_),
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
        }
        candidate_results.append(result)

        print(
            f"Candidate {number}: PR-AUC={pr_auc:.6f}, "
            f"ROC-AUC={roc_auc:.6f}, iteration={model.best_iteration_}"
        )

        # PR-AUC selects the winner; ROC-AUC must remain safely above 0.80.
        if roc_auc >= 0.80 and pr_auc > best_pr_auc:
            best_model = model
            best_parameters = candidate
            best_pr_auc = pr_auc

    if best_model is None:
        raise RuntimeError("No candidate achieved ROC-AUC of at least 0.80")

    print("\nCalibrating the winning model...")
    raw_calibration_probability = best_model.predict_proba(
        X_cal,
        num_iteration=best_model.best_iteration_,
    )[:, 1]

    # Platt scaling: reshape because LogisticRegression expects a 2D input.
    calibrator = LogisticRegression(random_state=42)
    calibrator.fit(raw_calibration_probability.reshape(-1, 1), y_cal)

    raw_evaluation_probability = best_model.predict_proba(
        X_eval,
        num_iteration=best_model.best_iteration_,
    )[:, 1]
    calibrated_probability = calibrator.predict_proba(
        raw_evaluation_probability.reshape(-1, 1)
    )[:, 1]

    threshold = best_f1_threshold(y_eval, calibrated_probability)
    predicted = (calibrated_probability >= threshold).astype("uint8")

    metrics = {
        "pr_auc": float(average_precision_score(y_eval, calibrated_probability)),
        "roc_auc": float(roc_auc_score(y_eval, calibrated_probability)),
        "precision": float(precision_score(y_eval, predicted, zero_division=0)),
        "recall": float(recall_score(y_eval, predicted, zero_division=0)),
        "f1": float(f1_score(y_eval, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(y_eval, calibrated_probability)),
        "threshold": threshold,
    }

    metadata = {
        "model_name": "LightGBM PR-AUC Challenger",
        "purpose": "Compare against the existing calibrated champion model",
        "target": TARGET,
        "training_end": CALIBRATION_START.isoformat(),
        "calibration_start": CALIBRATION_START.isoformat(),
        "evaluation_start": EVALUATION_START.isoformat(),
        "evaluation_end": EVALUATION_END.isoformat(),
        "features": features,
        "categorical_features": categorical,
        "selected_parameters": best_parameters,
        "best_iteration": int(best_model.best_iteration_),
        "decision_threshold": threshold,
        "candidate_results": candidate_results,
        "same_window_champion_pr_auc": 0.5460918460481213,
        "evaluation_metrics": metrics,
        "important_note": (
            "This threshold is for development comparison only. If the challenger "
            "wins, tune and lock its operational threshold on a later window."
        ),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(calibrator, CALIBRATOR_PATH)
    joblib.dump(frequency_maps, FREQUENCY_MAPS_PATH)
    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nCHALLENGER RESULTS")
    print("-" * 40)
    print(f"PR-AUC:      {metrics['pr_auc']:.6f}")
    print(f"ROC-AUC:     {metrics['roc_auc']:.6f}")
    print(f"Precision:   {metrics['precision']:.6f}")
    print(f"Recall:      {metrics['recall']:.6f}")
    print(f"F1 score:    {metrics['f1']:.6f}")
    print(f"Brier score: {metrics['brier_score']:.6f}")
    print(f"Threshold:   {metrics['threshold']:.6f}")
    print(f"\nModel saved:      {MODEL_PATH}")
    print(f"Calibrator saved: {CALIBRATOR_PATH}")
    print(f"Frequency maps:   {FREQUENCY_MAPS_PATH}")
    print(f"Metadata saved:   {METADATA_PATH}")
    print("\nSame-window champion PR-AUC: 0.546092")
    print("Final untouched-window benchmark PR-AUC: 0.448670")


if __name__ == "__main__":
    main()
