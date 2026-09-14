"""Validate predictions on the unseen 15-minute inference period."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "predictions" / "predictions.parquet"
)

METADATA_PATH = PROJECT_ROOT / "models" / "final_calibrated_metadata.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "inference_validation_metrics.json"
VALIDATED_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "predictions"
    / "validated_predictions.parquet"
)

START_TIME = pd.Timestamp("2017-11-07 15:15:00")
END_TIME = pd.Timestamp("2017-11-07 15:30:00")
CHUNK_SIZE = 1_000_000

CLICK_COLUMNS = ["ip", "app", "device", "os", "channel", "click_time"]
RAW_COLUMNS = CLICK_COLUMNS + ["is_attributed"]

DTYPES = {
    "ip": "uint32",
    "app": "uint16",
    "device": "uint16",
    "os": "uint16",
    "channel": "uint16",
    "is_attributed": "uint8",
}


def load_true_outcomes():
    """Read the true labels for the same period used during inference."""

    selected_chunks = []

    reader = pd.read_csv(
        RAW_DATA_PATH,
        usecols=RAW_COLUMNS,
        dtype=DTYPES,
        parse_dates=["click_time"],
        chunksize=CHUNK_SIZE,
    )

    print("Retrieving the hidden outcomes from the raw dataset...")

    for chunk_number, chunk in enumerate(reader, start=1):
        window_mask = (
            (chunk["click_time"] >= START_TIME)
            & (chunk["click_time"] < END_TIME)
        )

        if window_mask.any():
            selected_chunks.append(chunk.loc[window_mask].copy())

        # The raw file is chronological, so we can stop after the window.
        if chunk["click_time"].max() >= END_TIME:
            break

        if chunk_number % 10 == 0:
            print(f"Processed {chunk_number * CHUNK_SIZE:,} raw rows...")

    if not selected_chunks:
        raise ValueError("No true outcomes were found for the inference period")

    return pd.concat(selected_chunks, ignore_index=True)


def main():
    required_files = [RAW_DATA_PATH, PREDICTIONS_PATH, METADATA_PATH]
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    predictions = pd.read_parquet(PREDICTIONS_PATH)
    true_outcomes = load_true_outcomes()

    if len(predictions) != len(true_outcomes):
        raise ValueError(
            "Prediction and outcome row counts do not match: "
            f"{len(predictions):,} versus {len(true_outcomes):,}"
        )

    # Confirm that every prediction still matches the correct original click.
    for column in CLICK_COLUMNS:
        if not predictions[column].equals(true_outcomes[column]):
            raise ValueError(f"Prediction rows do not align on column: {column}")

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    threshold = float(metadata["decision_threshold"])
    y_true = true_outcomes["is_attributed"]
    y_probability = predictions["calibrated_probability"]
    y_predicted = predictions["predicted_is_attributed"]

    # Save the actual labels beside the calibrated probabilities so that a
    # separate script can compare alternative decision thresholds.
    validated_predictions = pd.DataFrame({
        "is_attributed": y_true,
        "calibrated_probability": y_probability,
    })
    validated_predictions.to_parquet(
        VALIDATED_PREDICTIONS_PATH,
        index=False,
    )

    if not y_probability.between(0, 1).all():
        raise ValueError("Calibrated probabilities must be between 0 and 1")

    # Recreate the threshold decision and verify the saved predictions.
    expected_predictions = (y_probability >= threshold).astype("uint8")
    if not y_predicted.equals(expected_predictions):
        raise ValueError("Saved classes do not match the calibrated threshold")

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_predicted,
        labels=[0, 1],
    ).ravel()

    actual_positive_rate = float(y_true.mean())
    predicted_positive_rate = float(y_predicted.mean())
    precision = float(precision_score(y_true, y_predicted, zero_division=0))

    metrics = {
        "evaluation_window_start": START_TIME.isoformat(),
        "evaluation_window_end": END_TIME.isoformat(),
        "rows": int(len(predictions)),
        "actual_attributed_clicks": int(y_true.sum()),
        "predicted_attributed_clicks": int(y_predicted.sum()),
        "actual_positive_rate": actual_positive_rate,
        "predicted_positive_rate": predicted_positive_rate,
        "decision_threshold": threshold,
        "pr_auc": float(average_precision_score(y_true, y_probability)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "accuracy": float(accuracy_score(y_true, y_predicted)),
        "precision": precision,
        "recall": float(recall_score(y_true, y_predicted, zero_division=0)),
        "f1": float(f1_score(y_true, y_predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_probability)),
        "log_loss": float(log_loss(y_true, y_probability, labels=[0, 1])),
        "precision_lift_over_baseline": (
            precision / actual_positive_rate if actual_positive_rate > 0 else None
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nInference validation results")
    print("-" * 31)
    print(f"Rows evaluated:            {metrics['rows']:,}")
    print(f"Actual attributed clicks:  {metrics['actual_attributed_clicks']:,}")
    print(f"Predicted attributed:      {metrics['predicted_attributed_clicks']:,}")
    print(f"PR-AUC:                    {metrics['pr_auc']:.6f}")
    print(f"ROC-AUC:                   {metrics['roc_auc']:.6f}")
    print(f"Accuracy:                  {metrics['accuracy']:.6f}")
    print(f"Precision:                 {metrics['precision']:.6f}")
    print(f"Recall:                    {metrics['recall']:.6f}")
    print(f"F1 score:                  {metrics['f1']:.6f}")
    print(f"Brier score:               {metrics['brier_score']:.6f}")
    print(f"Log loss:                  {metrics['log_loss']:.6f}")
    print(f"True negatives:            {metrics['true_negatives']:,}")
    print(f"False positives:           {metrics['false_positives']:,}")
    print(f"False negatives:           {metrics['false_negatives']:,}")
    print(f"True positives:            {metrics['true_positives']:,}")
    print(f"\nReport saved: {REPORT_PATH}")
    print(f"Validated predictions saved: {VALIDATED_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
