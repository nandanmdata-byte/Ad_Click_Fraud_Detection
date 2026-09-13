"""Make calibrated predictions with the final LightGBM model."""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd


# -----------------------------------------------------------------------------
# Project paths
# Note: Change a filename here only if the saved file uses a different name.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "inference" / "new_clicks.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "predictions.parquet"

MODEL_PATH = PROJECT_ROOT / "models" / "final_lightgbm_model.joblib"
CALIBRATOR_PATH = PROJECT_ROOT / "models" / "score_calibrator.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "final_calibrated_metadata.json"


# This makes the existing feature-building script available when this file is
# run with: python src/models/predict.py
sys.path.insert(0, str(PROJECT_ROOT))
from src.features.build_features import build_features


REQUIRED_COLUMNS = ["ip", "app", "device", "os", "channel", "click_time"]


def main():
    # Check that every required project file exists before starting.
    required_files = [
        INPUT_PATH,
        MODEL_PATH,
        CALIBRATOR_PATH,
        METADATA_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    print(f"Reading clicks: {INPUT_PATH}")
    clicks = pd.read_parquet(INPUT_PATH)

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in clicks.columns
    ]
    if missing_columns:
        raise ValueError(f"Input is missing columns: {missing_columns}")

    if clicks.empty:
        raise ValueError("The input file contains no clicks")

    if clicks[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("The required input columns contain missing values")

    # The model must never use attributed_time or is_attributed as inputs.
    clicks = clicks[REQUIRED_COLUMNS].copy()
    clicks["click_time"] = pd.to_datetime(clicks["click_time"])

    # Save the original row order because build_features sorts clicks by time.
    clicks["original_row_number"] = range(len(clicks))

    print("Building the same features used during model training...")
    featured_clicks = build_features(clicks)

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not metadata["calibration_accepted"]:
        raise ValueError("The metadata does not contain an accepted calibration")

    feature_names = metadata["features"]
    threshold = metadata["decision_threshold"]

    model = joblib.load(MODEL_PATH)
    calibrator = joblib.load(CALIBRATOR_PATH)

    # == Stage 1: LightGBM produces its original model probability. ==
    raw_probability = model.predict_proba(featured_clicks[feature_names])[:, 1]

    ''' ==
    Stage 2: Platt scaling converts the raw probability into a better
    calibrated probability. The reshape makes it a one-column table, which
    is the format expected by the logistic-regression calibrator.
        ==
    '''
    calibrated_probability = calibrator.predict_proba(
        raw_probability.reshape(-1, 1)
    )[:, 1]

    featured_clicks["raw_model_probability"] = raw_probability
    featured_clicks["calibrated_probability"] = calibrated_probability

    '''
    A click is classified as attributed when its calibrated probability is
    equal to or greater than the threshold selected during calibration.
    '''
    featured_clicks["predicted_is_attributed"] = (
        featured_clicks["calibrated_probability"] >= threshold
    ).astype("uint8")

    # == Return predictions in the same order as the original input file. ==
    featured_clicks = featured_clicks.sort_values("original_row_number")

    prediction_columns = REQUIRED_COLUMNS + [
        "raw_model_probability",
        "calibrated_probability",
        "predicted_is_attributed",
    ]
    predictions = featured_clicks[prediction_columns]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(OUTPUT_PATH, index=False)

    predicted_installs = int(predictions["predicted_is_attributed"].sum())

    print(f"Predictions saved: {OUTPUT_PATH}")
    print(f"Clicks scored: {len(predictions):,}")
    print(f"Predicted attributed clicks: {predicted_installs:,}")
    print(f"Decision threshold: {threshold:.6f}")


if __name__ == "__main__":
    main()
