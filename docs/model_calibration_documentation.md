# TalkingData Ad Click Attribution — Model Calibration

## 1. Stage overview

The model-calibration stage converts the final LightGBM model's raw output scores into values that more closely represent real attribution probabilities.

The trained LightGBM model already ranks clicks effectively, but it was trained with balanced class weights because attributed clicks are extremely rare. This weighting improves the model's ability to learn the minority class, but it also means that its raw scores should not automatically be interpreted as real probabilities.

For example, a raw LightGBM score of `0.95` does not necessarily mean that the click has a 95% chance of producing an attributed installation. Before calibration, the score mainly tells us that the click appears much more attribution-like than clicks receiving lower scores.

Calibration adds a small probability-conversion model after LightGBM:

```text
16 click features
       ↓
Frozen engineered LightGBM
       ↓
Raw model score
       ↓
Sigmoid calibrator
       ↓
Calibrated attribution probability
       ↓
Operational threshold
       ↓
Final attributed / non-attributed prediction
```

> **Target interpretation:** `is_attributed = 1` means that an app installation was attributed to the click. It is not a confirmed legitimate-click label. Similarly, `is_attributed = 0` does not prove fraud.

---

## 2. Purpose of the calibration notebook

The calibration notebook performs the following responsibilities:

1. Loads and validates the frozen LightGBM model.
2. Loads the model metadata and confirms the expected 16 features.
3. Loads the processed development features.
4. Divides the original validation period into two chronological sections.
5. Generates raw LightGBM scores without retraining the model.
6. Fits a simple sigmoid probability calibrator.
7. Compares raw and calibrated probability quality.
8. Selects a new threshold targeting at least 80% recall.
9. Applies that threshold unchanged to the later evaluation period.
10. Saves the calibrator and updated calibration metadata only when calibration improves the Brier score.

The calibration notebook does **not** change the fitted LightGBM trees, the 16 input features or the selected LightGBM parameters. It adds a separate conversion layer on top of the frozen model.

---

## 3. Chronological calibration design

The original validation period was split into two consecutive 15-minute sections:

```text
Calibration period:            2017-11-07 14:30:00 to 14:45:00
Calibration evaluation period: 2017-11-07 14:45:00 to 15:00:00
```

The earlier period was used to:

- fit the sigmoid calibrator; and
- select a calibrated threshold that achieved at least 80% recall.

The later period was used to:

- evaluate probability quality;
- inspect the calibration curve; and
- test whether the earlier threshold continued to perform reasonably on later clicks.

This is stronger than fitting and evaluating the calibrator on exactly the same rows. It also preserves the time order expected in a real prediction setting.

However, the later 15-minute period was part of the broader validation window previously used for LightGBM early stopping and model development. It is therefore a useful calibration evaluation period, but not a completely untouched final test set. This limitation is acknowledged later in this document.

---

## 4. Calibration theory

### 4.1 Ranking and calibration are different

A classification model has two related but different abilities.

**Ranking quality** asks:

> Does the model generally assign higher scores to attributed clicks than to non-attributed clicks?

**Probability calibration** asks:

> When the model predicts a certain probability, does approximately that proportion of cases actually become attributed?

A model can rank cases well while producing probability values that are too high or too low. This was the situation with the class-weighted LightGBM: its rankings were useful, but the average raw score was much higher than the real attribution rate.

### 4.2 Sigmoid or Platt calibration

The project uses a simple sigmoid calibrator, implemented with logistic regression. The calibrator receives one input—the raw LightGBM score—and learns a smooth mapping to the observed target values.

Conceptually, the calibrator learns a function similar to:

\[
P(y=1 \mid s) = \frac{1}{1 + e^{-(a s + b)}}
\]

where:

- \(s\) is the raw LightGBM score;
- \(a\) and \(b\) are learned from the calibration period; and
- the output is the calibrated attribution probability.

The sigmoid mapping is monotonic: higher LightGBM scores remain higher after calibration. This normally preserves ranking while changing the numerical scale of the scores.

### 4.3 Why sigmoid calibration was selected

Sigmoid calibration was chosen because it is:

- simple to explain and reproduce;
- computationally efficient;
- appropriate for a large, imbalanced dataset;
- less flexible and therefore less likely to overfit than complex calibration methods; and
- easy to save and apply in the later inference pipeline.

The project did not require several calibration algorithms or a complicated tuning search. The objective was to determine whether one reliable calibration layer could improve probability interpretation.

---

## 5. Metrics used in this stage

### 5.1 Brier score

The Brier score measures the average squared difference between the predicted probability and the actual outcome:

\[
\text{Brier Score} = \frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2
\]

where:

- \(p_i\) is the predicted probability;
- \(y_i\) is the actual target, either 0 or 1; and
- \(N\) is the number of observations.

A lower Brier score is better. It rewards probability estimates that remain close to the observed outcomes.

The notebook uses Brier score as its main calibration acceptance check. If calibration does not improve the evaluation Brier score, the saving cell stops rather than adding a worse calibration layer to the project.

### 5.2 Log loss

Log loss evaluates probability accuracy and gives a larger penalty to confident predictions that are wrong:

\[
\text{Log Loss} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right]
\]

A lower value is better. Log loss complements the Brier score by checking whether the model makes unrealistic, overconfident probability predictions.

### 5.3 Mean predicted score

The mean predicted score is compared with the actual attribution rate.

For a reasonably calibrated model over a sufficiently representative group:

```text
Mean predicted probability ≈ Actual positive rate
```

This is not a complete calibration test by itself, but a large difference would indicate that the score scale is unrealistic.

### 5.4 PR-AUC

PR-AUC measures ranking performance across precision and recall levels. It remains important because the positive class represents only a very small part of the data.

Calibration is not expected to improve PR-AUC. A monotonic sigmoid mapping changes score values but normally keeps clicks in the same order. Therefore, stable PR-AUC is a positive result: probability interpretation improved without damaging ranking.

### 5.5 Precision, recall and F1-score

These metrics evaluate decisions after applying a threshold:

- **Precision:** Of the clicks predicted as attributed, how many were actually attributed?
- **Recall:** Of all actual attributed clicks, how many did the system identify?
- **F1-score:** A combined measure balancing precision and recall.

The operational policy prioritises recall by selecting the highest-precision threshold that achieves at least 80% recall on the calibration period.

### 5.6 Confusion matrix

The confusion matrix reports exact decision counts:

- **True positive:** an attributed click predicted as attributed;
- **True negative:** a non-attributed click predicted as non-attributed;
- **False positive:** a non-attributed click predicted as attributed; and
- **False negative:** an attributed click predicted as non-attributed.

These labels describe attribution prediction errors. They must not be reinterpreted as confirmed fraud outcomes.

---

## 6. Probability-calibration results

### Raw versus calibrated scores on the later evaluation period

| Metric | Raw LightGBM score | Calibrated score | Interpretation |
|---|---:|---:|---|
| PR-AUC | 0.54609185 | 0.54609185 | Ranking was preserved |
| Brier score | 0.01198701 | **0.00147514** | Probability error improved substantially |
| Log loss | 0.05426860 | **0.00581081** | Overconfident probability errors were greatly reduced |
| Mean score | 0.03432517 | **0.00196369** | Calibrated average became realistic |
| Actual positive rate | 0.00190213 | 0.00190213 | Evaluation reference rate |

The notebook returned:

```text
Calibration improved evaluation Brier score: True
```

### 6.1 Brier-score improvement

The Brier score decreased from `0.01198701` to `0.00147514`, an improvement of approximately **87.7%**.

This indicates that the calibrated probabilities are much closer to the observed outcomes than the original class-weighted scores.

### 6.2 Log-loss improvement

Log loss decreased from `0.05426860` to `0.00581081`, an improvement of approximately **89.3%**.

This shows that calibration substantially reduced unrealistic probability confidence.

### 6.3 Realistic average probability

Before calibration, the average raw LightGBM score was approximately **3.43%**, while the actual attribution rate was only **0.19%**. The raw scores therefore overstated the overall probability level.

After calibration:

```text
Mean calibrated probability: 0.1964%
Actual attribution rate:     0.1902%
```

The difference is very small. This is strong evidence that the calibrated score scale is much more realistic for this evaluation period.

### 6.4 Ranking was preserved

PR-AUC remained exactly the same at the reported precision:

```text
Raw PR-AUC:        0.54609185
Calibrated PR-AUC: 0.54609185
```

The calibration layer improved probability interpretation without damaging LightGBM's ability to rank clicks.

---

## 7. Understanding the calibration graph

The calibration curve compares:

```text
Horizontal axis: Mean predicted score
Vertical axis:   Observed attribution rate
```

The dashed diagonal line represents perfect calibration. A point on this line means that the predicted probability and observed event rate agree.

The raw LightGBM curve appeared far from the perfect-calibration line because balanced class weights caused the raw scores to be much larger than the actual attribution rates.

The calibrated curve moved substantially closer to the appropriate probability region. Most points appear near the lower-left corner because attributed clicks are extremely rare. This compressed appearance is expected and does not mean the graph failed.

The graph should be interpreted together with the numerical evidence. In this project, the major reductions in Brier score and log loss provide the clearest confirmation that calibration improved probability quality.

---

## 8. Calibrated threshold selection

The operational threshold was selected on the earlier calibration period using the following rule:

> Among all thresholds that achieve at least 80% recall, select the threshold with the highest precision.

The result was:

```text
Calibrated threshold:       0.18567327
Calibration-period recall:  80.0363%
Calibration-period precision: 26.0947%
```

The old raw-score threshold was approximately `0.954368`. After calibration, the new threshold is approximately `0.185673`.

This large numerical change is expected because the calibrator changed the score scale. The two thresholds should not be compared as if they represented the same kind of value:

- `0.954368` was a cutoff applied to a class-weighted ranking score;
- `0.185673` is a cutoff applied to a calibrated attribution probability.

The new threshold therefore has a clearer interpretation: clicks with a calibrated attribution probability of approximately **18.57% or higher** are classified as attributed under the selected operating policy.

---

## 9. Threshold transfer to the later period

The threshold selected on the earlier period was applied unchanged to the later evaluation period.

| Metric | Raw score with old threshold | Calibrated score with selected threshold |
|---|---:|---:|
| Threshold | 0.95436755 | 0.18567327 |
| Accuracy | **99.5521%** | 99.5356% |
| Precision | **27.2130%** | 26.5866% |
| Recall | 80.8793% | **81.8562%** |
| F1-score | **0.407238** | 0.401369 |
| True negatives | **748,834** | 748,695 |
| False positives | **3,100** | 3,239 |
| False negatives | 274 | **260** |
| True positives | 1,159 | **1,173** |

The calibrated threshold successfully transferred to later data:

- recall remained above the 80% target;
- precision remained close to the calibration-period result;
- 14 additional attributed clicks were identified;
- false negatives decreased by 14; and
- the model retained a similar overall precision–recall balance.

This provides evidence that the threshold was not useful only on the rows where it was selected.

---

## 10. Small operational trade-offs

Calibration was introduced to improve probability reliability, not to increase every classification metric simultaneously.

Compared with the old raw-score policy, the calibrated policy produced:

- a small precision decrease from 27.21% to 26.59%;
- a small F1 decrease from 0.4072 to 0.4014;
- 139 additional false positives; but
- a recall increase from 80.88% to 81.86%;
- 14 additional true positives; and
- 14 fewer false negatives.

These are small and understandable operating-point differences. They do not represent a calibration failure.

The calibrated system provides two valuable improvements in return:

1. Its scores correspond much more closely to observed attribution rates.
2. Its threshold selected on one period maintained the recall objective on the next period.

For an application that explicitly prioritises recall near 80%, the slightly higher recall and lower number of missed attributed clicks are reasonable trade-offs. More importantly, the probability displayed to users is now far more defensible than the original weighted LightGBM score.

---

## 11. Why threshold 0.50 produced no positives

When the calibrated scores were evaluated at a threshold of `0.50`, the model produced no positive predictions.

This is not an error. The actual attribution rate is approximately 0.19%, so a properly calibrated probability above 50% represents an exceptionally rare and confident prediction. None of the clicks in the later 15-minute period reached that level.

For highly imbalanced problems, `0.50` is not automatically an appropriate operational cutoff. This is why the project selects a threshold based on the required precision–recall policy rather than accepting the default value.

---

## 12. Saved artifacts and their roles

The calibrated prediction system uses three artifacts:

```text
models/final_lightgbm_model.joblib
models/score_calibrator.joblib
models/final_calibrated_metadata.json
```

### `final_lightgbm_model.joblib`

Stores the frozen engineered LightGBM model. It receives the 16 model features and returns a raw model score.

### `score_calibrator.joblib`

Stores the fitted sigmoid calibrator. It converts the one-dimensional LightGBM score into a calibrated attribution probability.

### `final_calibrated_metadata.json`

Stores the information needed to reproduce the calibrated prediction process, including:

- ordered feature names;
- categorical feature names;
- calibration method;
- calibration and evaluation time boundaries;
- calibrated decision threshold;
- raw and calibrated probability metrics;
- calibrated classification metrics; and
- software-library versions.

The later inference pipeline must apply these artifacts in the correct order:

```python
raw_score = model.predict_proba(features)[:, 1]

calibrated_probability = calibrator.predict_proba(
    raw_score.reshape(-1, 1)
)[:, 1]

prediction = (
    calibrated_probability >= metadata["decision_threshold"]
).astype(int)
```

---

## 13. Limitations

### 13.1 Calibration data came from the development window

The calibrator was fitted and evaluated using two chronological parts of the original 30-minute validation period. Although the later part was not used to fit the calibrator or select its threshold, the full window had previously influenced LightGBM early stopping and model-development decisions.

The evaluation is therefore a strong development-stage calibration check, not a fully independent final test.

### 13.2 The calibration period is short

Both calibration and evaluation used 15-minute traffic windows. The large number of clicks provides substantial data, but the period may not capture different days, hours or changes in traffic behaviour.

### 13.3 Calibration depends on the future data distribution

Probability calibration can become less accurate if the underlying attribution rate or traffic patterns change. In a real production system, calibration quality would need periodic monitoring and possible refitting.

### 13.4 The target remains attribution, not verified fraud

Calibration makes attribution probabilities more meaningful. It does not transform the target into confirmed fraud labels. The application must continue to describe the result as attribution likelihood or a proxy signal related to traffic quality.

### 13.5 Historical features still require prior-click context

Calibration does not solve feature-generation requirements. The final application still needs a reliable method for obtaining previous-click counts and elapsed-time features without using future information.

---

## 14. Overall insights

The main findings from the calibration stage are:

1. **The original LightGBM scores were strongly overestimated as probabilities.** Their mean was 3.43% against an actual rate of 0.19%.
2. **Sigmoid calibration corrected the score scale.** The calibrated mean of 0.1964% closely matched the actual rate of 0.1902%.
3. **Probability error improved substantially.** Brier score fell by approximately 87.7%, while log loss fell by approximately 89.3%.
4. **Predictive ranking was preserved.** PR-AUC remained at 0.54609185.
5. **The recall policy transferred to later data.** A threshold selected for 80% recall produced 81.86% recall in the following period.
6. **The small classification trade-offs were controlled.** Precision and F1 declined slightly, while recall improved and fewer attributed clicks were missed.
7. **The calibrated probability is more appropriate for user-facing output.** The Streamlit application can present an estimated attribution probability instead of displaying an inflated class-weighted score.

---

## 15. Why it is safe to move to the next stage

The calibration stage meets the necessary development acceptance conditions:

| Acceptance check | Result |
|---|---|
| Brier score improved | Passed |
| Log loss improved | Passed |
| Mean probability close to actual rate | Passed |
| PR-AUC preserved | Passed |
| Later-period recall remained near or above 80% | Passed |
| Calibrated threshold transferred to later data | Passed |
| Calibrator and metadata saved separately | Passed, provided the saving cell completed successfully |

The calibrated system is therefore suitable for the next project stage. No further model or calibration tuning is required at this point.

The next step is to build a reusable inference pipeline that:

1. validates incoming click data;
2. creates the required 16 features without future leakage;
3. loads the frozen LightGBM model;
4. generates the raw LightGBM score;
5. converts it into a calibrated probability;
6. applies the saved threshold of approximately `0.185673`; and
7. returns a clearly worded attribution prediction.

---

## 16. Final conclusion

The sigmoid calibration stage was successful. It substantially improved the reliability of LightGBM's score scale while preserving the model's ranking performance. The calibrated average probability closely matched the observed attribution rate, both Brier score and log loss improved sharply, and the selected recall-focused threshold remained stable on later chronological data.

The small decrease in precision and F1 is an acceptable operating trade-off rather than a failure: the calibrated system achieved higher recall, identified additional attributed clicks and produced probabilities that are much more suitable for interpretation and application use.

With the LightGBM model frozen, calibration accepted and the new threshold saved, the project is ready to proceed to reusable inference-pipeline development.
