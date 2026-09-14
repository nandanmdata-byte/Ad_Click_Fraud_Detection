"""Choose a better threshold using unseen validation predictions.

Input file columns: is_attributed and predicted_probability.
Keep the natural class distribution; do not balance this validation data.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


# SETTINGS — edit these values when needed
PREDICTIONS_PATH = Path("data/predictions/validated_predictions.parquet")
OUTPUT_DIR = Path("reports/threshold_optimization")

TARGET_COLUMN = "is_attributed"
PROBABILITY_COLUMN = "calibrated_probability"

MINIMUM_PRECISION = 0.25
MINIMUM_RECALL = 0.70

# Example amounts; replace these with realistic business values later.
VALUE_PER_TRUE_POSITIVE = 10.0
COST_PER_FALSE_POSITIVE = 1.0
COST_PER_FALSE_NEGATIVE = 20.0


def load_predictions():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_PATH}\n"
            "First save the actual labels and probabilities from inference."
        )

    if PREDICTIONS_PATH.suffix == ".parquet":
        df = pd.read_parquet(PREDICTIONS_PATH)
    else:
        df = pd.read_csv(PREDICTIONS_PATH)

    required = {TARGET_COLUMN, PROBABILITY_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    return df[[TARGET_COLUMN, PROBABILITY_COLUMN]].dropna()


def evaluate_thresholds(df):
    y_true = df[TARGET_COLUMN].to_numpy()
    probabilities = df[PROBABILITY_COLUMN].to_numpy()
    results = []

    for threshold in np.arange(0.01, 1.00, 0.01):
        y_pred = (probabilities >= threshold).astype(int)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        business_value = (
            tp * VALUE_PER_TRUE_POSITIVE
            - fp * COST_PER_FALSE_POSITIVE
            - fn * COST_PER_FALSE_NEGATIVE
        )

        results.append({
            "threshold": round(float(threshold), 2),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "business_value": business_value,
        })

    return pd.DataFrame(results)


def main():
    predictions = load_predictions()
    results = evaluate_thresholds(predictions)

    # Prefer thresholds that meet both requirements, then maximize F1.
    qualified = results[
        (results["precision"] >= MINIMUM_PRECISION)
        & (results["recall"] >= MINIMUM_RECALL)
    ]

    if qualified.empty:
        best_balanced = results.loc[results["f1_score"].idxmax()]
        message = "No threshold met both minimum requirements; using maximum F1."
    else:
        best_balanced = qualified.loc[qualified["f1_score"].idxmax()]
        message = "The selected threshold met both minimum requirements."

    best_business = results.loc[results["business_value"].idxmax()]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "threshold_results.csv", index=False)

    recommendations = {
        "balanced_threshold": round(float(best_balanced["threshold"]), 2),
        "balanced_precision": float(best_balanced["precision"]),
        "balanced_recall": float(best_balanced["recall"]),
        "balanced_f1": float(best_balanced["f1_score"]),
        "business_threshold": round(float(best_business["threshold"]), 2),
        "maximum_business_value": float(best_business["business_value"]),
        "message": message,
    }

    with (OUTPUT_DIR / "recommended_thresholds.json").open("w", encoding="utf-8") as file:
        json.dump(recommendations, file, indent=2)

    print("\nTHRESHOLD OPTIMIZATION RESULTS")
    print("-" * 40)
    print(f"Balanced threshold: {best_balanced['threshold']:.2f}")
    print(f"Precision:          {best_balanced['precision']:.4f}")
    print(f"Recall:             {best_balanced['recall']:.4f}")
    print(f"F1 score:           {best_balanced['f1_score']:.4f}")
    print(f"False positives:    {int(best_balanced['false_positives']):,}")
    print(f"False negatives:    {int(best_balanced['false_negatives']):,}")
    print(f"\nBusiness threshold: {best_business['threshold']:.2f}")
    print(f"Business value:     {best_business['business_value']:,.2f}")
    print(f"\n{message}")


if __name__ == "__main__":
    main()
