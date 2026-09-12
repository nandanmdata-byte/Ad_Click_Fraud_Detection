# TalkingData Ad Click Attribution — Model Development

## 1. Stage overview

This document records the model-development stage of the TalkingData Ad Click Fraud Detection portfolio project. The objective of this stage was to build a simple, reproducible classification model that can distinguish clicks that lead to an attributed app installation from clicks that do not.

The completed progression was:

1. Validate the processed development dataset.
2. Apply a chronological training and validation split.
3. Train a raw-feature LightGBM baseline.
4. Train a LightGBM model with leakage-safe engineered features.
5. Compare feature-set performance.
6. Refine the selected LightGBM configuration.
7. Evaluate alternative classification thresholds.
8. Test XGBoost as a challenger.
9. Select and save the final model and metadata.

> **Important target interpretation:** `is_attributed = 1` means that an app installation was attributed to the click. It is not a verified non-fraud label. Similarly, `is_attributed = 0` means that no installation was attributed; it does not prove that the click was fraudulent.

---

## 2. Development data

The modelling dataset was loaded from:

```text
data/processed/development_features.parquet
```

It was created from a two-hour chronological window of the full TalkingData training data.

| Property | Value |
|---|---:|
| Development period | 2017-11-07 13:00:00 to 15:00:00 |
| Total rows | 6,397,036 |
| Attributed clicks | 15,775 |
| Attribution rate | 0.246599% |
| Target | `is_attributed` |

### Pre-training validation

Before model training, the notebook verifies that:

- the processed Parquet file exists and can be loaded;
- all required raw, time and target columns exist;
- `click_time` is stored as a datetime value;
- `is_attributed` contains only `0` and `1` with no missing values;
- rows remain sorted chronologically;
- the validation boundary falls inside the dataset time range;
- model features are numeric; and
- the feature matrix contains no missing or infinite values.

These checks prevent silent schema or data-quality problems from entering model training.

---

## 3. Chronological validation strategy

The data was divided using time rather than random sampling:

```text
Training:   click_time < 2017-11-07 14:30:00
Validation: click_time >= 2017-11-07 14:30:00
```

A chronological split is necessary because the model is intended to make predictions about future clicks. A random split could allow later traffic patterns to influence model development and produce an unrealistically optimistic result.

The validation period contained:

| Validation statistic | Value |
|---|---:|
| Rows | 1,530,918 |
| Attributed clicks | 3,086 |
| Non-attributed clicks | 1,527,832 |
| Attribution rate | approximately 0.20% |

The severe class imbalance makes accuracy insufficient as the main evaluation metric. A model predicting every click as non-attributed would appear highly accurate while detecting no attributed installations.

---

## 4. Feature sets

### 4.1 Raw baseline features

The baseline used five original click identifiers:

```text
ip
app
device
os
channel
```

Its purpose was to establish how well LightGBM could perform without behavioural or temporal feature engineering.

### 4.2 Engineered feature set

The engineered model used 16 features:

| Feature group | Features |
|---|---|
| Raw identifiers | `ip`, `app`, `device`, `os`, `channel` |
| Click-time components | `click_day`, `click_hour`, `click_minute`, `click_second`, `click_day_of_week` |
| Previous-click counts | `previous_ip_clicks`, `previous_ip_app_clicks`, `previous_ip_channel_clicks` |
| Previous-click intervals | `seconds_since_previous_ip_click`, `seconds_since_previous_ip_app_click`, `seconds_since_previous_ip_channel_click` |

The historical features were designed to use only the current click and clicks that occurred earlier. Future clicks and `attributed_time` were excluded to prevent target leakage.

---

## 5. Evaluation metrics

The following metrics were retained because each provides a useful and understandable view of performance:

| Metric | Purpose |
|---|---|
| PR-AUC | Primary model-selection metric; measures ranking quality for the rare positive class across thresholds |
| Accuracy | Percentage of all validation rows classified correctly; reported only as a secondary metric |
| Precision | Percentage of predicted attributed clicks that were actually attributed |
| Recall | Percentage of actual attributed clicks successfully identified |
| F1-score | Harmonic balance between precision and recall |
| Confusion matrix | Exact counts of true positives, true negatives, false positives and false negatives |

PR-AUC was selected as the primary metric because it focuses on the precision–recall relationship under extreme class imbalance. Threshold-dependent metrics were then used to choose an operating policy.

---

## 6. Raw versus engineered LightGBM

Both models used the same algorithm and chronological validation period. This isolated the effect of feature engineering.

### Final comparison at the default 0.50 threshold

| Metric | Raw LightGBM | Engineered LightGBM |
|---|---:|---:|
| Features | 5 | 16 |
| PR-AUC | 0.464747 | **0.549241** |
| Accuracy | 97.8983% | **98.7315%** |
| Precision | 8.0908% | **12.6333%** |
| Recall | **90.9916%** | 89.4686% |
| F1-score | 0.148603 | **0.221403** |
| True negatives | 1,495,934 | **1,508,738** |
| False positives | 31,898 | **19,094** |
| False negatives | **278** | 325 |
| True positives | **2,808** | 2,761 |

The raw model detected 47 more attributed clicks but generated 12,804 additional false positives. The engineered model achieved substantially higher PR-AUC, precision, accuracy and F1 while greatly reducing false positives.

This result demonstrates that the temporal and previous-click features provide meaningful predictive information beyond the original identifiers.

---

## 7. Final LightGBM configuration

The final model used a small, controlled refinement of the original configuration:

```python
lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=50,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)
```

Early stopping was applied using chronological validation PR-AUC. Although the model was allowed a maximum of 1,200 boosting iterations, validation performance selected iteration **494** as the best point.

### Why these changes were retained

- Increasing `num_leaves` from 31 to 63 allowed the model to represent more detailed feature interactions.
- `min_child_samples=50` limited excessively small and specific leaves.
- Reducing `learning_rate` to 0.03 allowed the boosting process to learn more gradually.
- A larger `n_estimators` value provided enough room for the lower learning rate, while early stopping prevented unnecessary training.
- `class_weight="balanced"` increased the influence of the rare attributed class during training.
- `random_state=42` was kept fixed for reproducibility. Random seeds were not searched for a favourable validation result.

### Improvement over the original engineered configuration

| Metric | Original engineered model | Refined engineered model |
|---|---:|---:|
| PR-AUC | 0.538654 | **0.549241** |
| Precision at threshold 0.50 | 11.0263% | **12.6333%** |
| Recall at threshold 0.50 | **89.8898%** | 89.4686% |
| F1 at threshold 0.50 | 0.196431 | **0.221403** |
| False positives at threshold 0.50 | 22,384 | **19,094** |

The refined model improved overall probability ranking, default-threshold precision and F1, while reducing false positives by 3,290. Its recall decreased only slightly.

At the exact 80% recall operating point, the earlier configuration achieved marginally higher precision—27.30% versus 27.01%. The refined configuration was nevertheless retained because PR-AUC was the preselected primary model-selection metric and its increase from 0.538654 to 0.549241 represents stronger ranking performance across thresholds. This trade-off is recorded rather than hidden.

---

## 8. Threshold evaluation

LightGBM produces a continuous model score. A classification threshold converts that score into a predicted label:

```python
prediction = (model_score >= threshold).astype(int)
```

Threshold selection does not retrain or alter the fitted trees. It changes the operating balance between precision and recall.

### Evaluated threshold policies

| Policy | Threshold | Precision | Recall | F1 | False positives | False negatives | True positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default | 0.500000 | 12.6333% | 89.4686% | 0.221403 | 19,094 | 325 | 2,761 |
| Maximum F1 | 0.990447 | **53.5980%** | 55.9948% | **0.547702** | **1,496** | 1,358 | 1,728 |
| **Minimum 80% recall** | **0.954368** | **27.0072%** | **80.0065%** | **0.403827** | **6,673** | **617** | **2,469** |

The default threshold found more attributed clicks but generated 19,094 false-positive predictions. The maximum-F1 threshold produced much higher precision but detected only about 56% of the actual attributed clicks.

The final operational threshold was therefore selected as the highest-precision validation threshold that maintained at least 80% recall:

```text
Final decision threshold: 0.9543675520814835
```

This policy identified 2,469 of the 3,086 attributed validation clicks while reducing false positives from 19,094 at the default threshold to 6,673.

Because balanced class weighting was used, the raw LightGBM outputs should currently be treated as ranking scores rather than literal real-world installation probabilities. Calibration is a later project stage.

---

## 9. XGBoost challenger

An XGBoost classifier was trained separately using the same engineered features, chronological validation data and 80% recall requirement. This provided an algorithm-level challenge without complicating the main modelling workflow.

### Comparison at approximately 80% recall

| Metric | LightGBM | XGBoost |
|---|---:|---:|
| PR-AUC | **0.538654** | 0.528682 |
| Accuracy | **99.5303%** | 99.4524% |
| Precision | **27.3029%** | 24.1231% |
| Recall | 80.0065% | 80.0065% |
| F1-score | **0.407123** | 0.370693 |
| False positives | **6,574** | 7,766 |
| False negatives | 617 | 617 |
| True positives | 2,469 | 2,469 |

This challenger experiment used the earlier LightGBM configuration available at the time. At equal recall, LightGBM produced 1,192 fewer false positives and achieved higher PR-AUC, precision, accuracy and F1. XGBoost therefore did not justify replacing LightGBM.

The raw threshold values of different algorithms were not compared directly because LightGBM and XGBoost can generate scores on different scales.

---

## 10. Why LightGBM was selected

The final engineered LightGBM was selected because it:

- achieved the highest chronological-validation PR-AUC of the tested final configurations;
- substantially outperformed the raw-feature baseline;
- outperformed the XGBoost challenger under an equal-recall comparison;
- handled more than six million development rows efficiently;
- supported early stopping and class-imbalance handling;
- provided an operational threshold that maintained approximately 80% recall;
- reduced false positives substantially compared with the default threshold; and
- remained simple enough to explain, reproduce and deploy in a portfolio application.

The model was selected based on a combination of ranking quality, operational performance, computational practicality and interpretability of the development process—not accuracy alone.

---

## 11. Final saved artifacts

```text
models/final_lightgbm_model.joblib
models/final_lightgbm_metadata.json
```

The Joblib file stores the fitted 16-feature engineered LightGBM model. It does not embed a fixed classification threshold.

The JSON metadata stores:

- model and target names;
- the 16 features in their required order;
- categorical feature names;
- chronological validation boundary;
- best boosting iteration;
- default-threshold metrics;
- maximum-F1 metrics;
- final operational-threshold metrics;
- the final decision threshold; and
- relevant library versions.

The final prediction system must use both artifacts:

```python
scores = model.predict_proba(input_features)[:, 1]
predictions = (scores >= metadata["decision_threshold"]).astype(int)
```

---

## 12. Limitations

### 12.1 The target is an attribution proxy, not verified fraud

The largest conceptual limitation is that TalkingData's `is_attributed` field records whether a click led to an attributed installation. It does not confirm whether an individual click was fraudulent.

Accordingly, the model should be described as predicting **attribution likelihood** or identifying **low-attribution-risk signals**, not proving fraud. False negatives are attributed clicks predicted as non-attributed; they are not fraudulent clicks that escaped detection.

### 12.2 Restricted development window

The model was developed on a two-hour subset rather than the entire 184.9-million-row training dataset. This made development feasible on a machine with 16 GB RAM, but the resulting traffic patterns may not represent every day or hour in the full data.

### 12.3 One chronological validation period

The model was evaluated on a later 30-minute window, which is preferable to a random split. However, stability has not yet been confirmed across multiple days or substantially different traffic periods.

### 12.4 Model and threshold decisions share the validation set

The validation period was used for model comparison, limited parameter refinement and threshold selection. The operational metrics are therefore development estimates rather than performance from a completely untouched final test set.

### 12.5 Uncalibrated scores

Balanced class weighting changes how the model learns from the rare positive class. Consequently, a score such as `0.95` should not yet be interpreted as a literal 95% probability of installation. Calibration must be performed before presenting scores as probabilities.

### 12.6 Historical features require state at inference time

Previous-click count and elapsed-time features depend on earlier events. A production-style application must either maintain click history or clearly explain how demonstration inputs obtain these values. They cannot be accurately reconstructed from a single isolated click without context.

### 12.7 Dataset age and anonymisation

The dataset contains anonymised identifiers and historical competition traffic. The model demonstrates methodology and engineering ability, but it should not be presented as a ready-to-deploy modern advertising fraud solution without newer labelled data and business validation.

---

## 13. Scope decision and next stage

Further parameter searching was intentionally stopped. Repeatedly testing configurations against the same validation period would increase the risk of overfitting model-development decisions to that window while adding limited portfolio value.

The model-development stage is considered complete because it produced:

- a validated and chronologically evaluated baseline;
- a stronger leakage-safe engineered model;
- a documented parameter refinement;
- an algorithm challenger;
- an explicit operating-threshold policy;
- reproducible model and metadata artifacts; and
- an honest account of performance and limitations.

The next project stages are:

1. Calibrate model scores using an appropriate untouched or separately defined calibration period.
2. Re-select the operational threshold after calibration because calibration changes the score scale.
3. Build a reusable inference pipeline that reproduces all 16 features safely.
4. Develop the Streamlit application for individual and batch scoring.
5. If resources permit, evaluate the frozen pipeline on a later untouched time window.

---

## 14. Final conclusion

The engineered LightGBM provides a strong and professionally defensible result for this portfolio project. It achieved a chronological-validation PR-AUC of **0.549241** on a target with an approximately **0.20% validation positive rate**. Under the selected operating policy, it maintained **80.01% recall** with **27.01% precision**, identifying 2,469 attributed clicks while substantially reducing false positives compared with the default threshold.

More importantly, the development process demonstrates the correct handling of chronological data, leakage-safe feature engineering, extreme class imbalance, threshold trade-offs, reproducible artifact saving and honest target interpretation. These methodological strengths make the model suitable for continuing into calibration, inference and application development.
