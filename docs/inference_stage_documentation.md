# Inference Dataset Creation and Model Inference

## 1. Stage Overview

This stage tests whether the completed model pipeline can accept new click data and produce usable predictions.

It contains two parts:

1. Create a new, chronologically later inference dataset.
2. Generate calibrated predictions with the final LightGBM model.

This stage does not yet measure model performance on the new period. Performance will be checked in the next validation stage by comparing predictions with the true labels.

---

## 2. Inference Dataset Creation

### Purpose

The inference dataset represents future clicks that were not used during model training, model selection, or calibration.

Using later data is important because a real fraud-detection system must make predictions on new clicks rather than on data it has already learned from.

### Script

```text
src/data/create_inference_dataset.py
```

### Source and Time Window

The script reads the original TalkingData file:

```text
data/raw/train.csv
```

It selects this chronological window:

```text
2017-11-07 15:00:00 <= click_time < 2017-11-07 15:15:00
```

The earlier development dataset ended at `15:00:00`. Therefore, this 15-minute period occurs strictly after the data used for development and calibration.

### Why Chunked Reading Is Used

The original CSV contains approximately 184.9 million rows and is too large to load into memory at once on a 16 GB system.

The script reads one million rows at a time. It keeps only rows inside the required time window and stops once the end of the window is reached. This reduces memory usage while preserving the chronological order of the data.

### Columns Included

| Column | Meaning |
| --- | --- |
| `ip` | Encoded IP address associated with the click |
| `app` | Encoded application ID |
| `device` | Encoded device type ID |
| `os` | Encoded operating-system ID |
| `channel` | Encoded advertising channel ID |
| `click_time` | Time at which the click occurred |

The following columns are intentionally excluded:

- `is_attributed`: the target or true outcome.
- `attributed_time`: information recorded after an installation and therefore unavailable when the click first occurs.

Excluding them prevents target leakage and makes the dataset resemble real prediction input.

### Dataset Checks

Before saving, the script confirms that:

- All six required columns are present.
- Required values are not missing.
- Clicks remain chronologically ordered.
- Every row falls inside the selected time window.
- The selected period is not empty.

### Output

```text
data/inference/new_clicks.parquet
```

The Parquet format is used because it is compact, fast to read, and preserves data types better than CSV.

---

## 3. Model Inference

### Purpose

Inference is the process of using a trained model to generate predictions for new observations.

Here, each observation is an ad click. The system estimates the probability that the click will lead to an attributed app installation.

### Script

```text
src/models/predict.py
```

### Files Used

| File | Purpose |
| --- | --- |
| `new_clicks.parquet` | New clicks to score |
| `final_lightgbm_model.joblib` | Selected 16-feature LightGBM model |
| `score_calibrator.joblib` | Logistic-regression Platt calibrator |
| `final_calibrated_metadata.json` | Feature order, accepted threshold, and model information |

### Inference Flow

#### Step 1: Build the Same Features

The prediction script calls the existing `build_features()` function. This produces the same 16 inputs used during model training:

- Five original categorical identifiers
- Five time-based features
- Three previous-click count features
- Three previous-click time-gap features

Reusing the same function reduces the risk of training–inference mismatch.

All count and time-gap features are backward-looking. They use only clicks observed earlier in the inference batch and do not use future outcomes.

#### Step 2: Generate the Raw Model Probability

```python
raw_probability = model.predict_proba(featured_clicks[feature_names])[:, 1]
```

`predict_proba()` returns the estimated probabilities of class `0` and class `1` for every click. `[:, 1]` selects the probability of class `1`, meaning that the click led to an attributed installation.

#### Step 3: Apply Platt Calibration

The raw LightGBM probability is passed to the saved logistic-regression calibrator.

Platt scaling adjusts the model score so that it behaves more like a realistic probability. For example, clicks receiving a calibrated probability near 0.20 should, across a sufficiently representative sample, have an outcome rate closer to 20% than before calibration.

Calibration changes probability quality but does not change the underlying ranking ability of the LightGBM model.

#### Step 4: Apply the Decision Threshold

The accepted calibrated threshold is:

```text
0.18567326747288235
```

The final rule is:

```text
Calibrated probability >= 0.185673 -> predicted class 1
Calibrated probability < 0.185673  -> predicted class 0
```

This threshold was previously selected to maintain approximately 80% recall on the calibration period. It is deliberately lower than `0.50` because attributed clicks are extremely rare and missing a genuine attributed click has been treated as important.

### Prediction Output

The prediction file contains the original click identifiers and time, followed by:

| Output column | Meaning |
| --- | --- |
| `raw_model_probability` | Original probability produced by LightGBM |
| `calibrated_probability` | Probability after Platt scaling |
| `predicted_is_attributed` | Final class produced using the calibrated threshold |

The predictions are returned in the same row order as the original input file.

---

## 4. Inference Results

| Result | Value |
| --- | ---: |
| Clicks scored | 775,437 |
| Predicted attributed clicks | 4,737 |
| Predicted non-attributed clicks | 770,700 |
| Predicted attribution rate | 0.611% |
| Decision threshold | 0.185673 |

The predicted attribution rate is calculated as:

```text
4,737 / 775,437 x 100 = approximately 0.611%
```

### How to Interpret These Numbers

`4,737` is the number of clicks classified as likely attributed by the model. It is not yet the number of clicks that actually produced installations.

Similarly, `0.611%` is the predicted positive rate, not the true attribution rate and not an accuracy score.

The predicted rate can be higher than the dataset's usual true positive rate because the chosen threshold prioritizes recall. This allows the model to capture more possible attributed clicks, with the accepted trade-off that some additional non-attributed clicks may also be flagged.

The next validation stage must compare these predictions with the withheld `is_attributed` values before conclusions are made about accuracy, precision, recall, F1 score, or PR-AUC on this future period.

---

## 5. Important Notes

### Behavioral Feature Starting Point

Previous-click counts and time gaps begin at zero or `-1` at the start of this inference batch because earlier history is not supplied to the simplified script.

This is acceptable for the current portfolio demonstration because the input is a continuous, chronologically ordered batch. In a production system, feature state should normally be carried forward from earlier clicks so the first rows also have complete history.

### Environment Consistency

The model and calibrator were originally saved with scikit-learn `1.7.2`. After aligning the environment, inference ran without version warnings. Matching saved-model library versions reduces the risk of inconsistent behavior during loading and prediction.

---

## 6. Overall Insights and Stage Decision

The completed run confirms that the end-to-end inference pipeline works:

- A genuinely later click window was extracted from the full dataset.
- Target and post-outcome information were excluded from model input.
- The same leakage-safe feature logic used during training was reused.
- All 775,437 rows were scored successfully.
- Raw model probabilities were passed through the accepted Platt calibrator.
- The saved calibrated threshold was applied successfully.
- Predictions were written to the expected output file without errors or version warnings.

The prediction count alone does not prove model quality, but it does prove that the saved model pipeline can process unseen data and produce complete, calibrated outputs.

**Decision: The inference dataset creation and inference execution stages are complete. It is safe to proceed to inference validation**, where predictions will be checked structurally and then compared with the hidden true outcomes from the same future window.
