# Bias checks & mitigation recommendations (SHAP/LIME assignment)

This report describes the bias checks and mitigations implemented in `notebook_fairness_shap_lime.ipynb` using a **synthetic tabular classification** dataset.

## 1) Sensitive groups and fairness checks

**Sensitive attributes used:**
- `gender` ∈ {M, F}
- `age_group` ∈ {young, mid, old}
- `race` ∈ {A, B, C}

The model predicts a binary outcome (`label`) based on both sensitive attributes and non-sensitive numeric features.

### Metrics computed per sensitive group
For each sensitive attribute and each group value, the notebook reports:
1. **Selection rate** = fraction predicted as positive (`P(ŷ=1)`) within that group.
2. **Equal Opportunity (TPR)** = `P(ŷ=1 | y=1)` within that group.
3. **Precision / Recall / F1** for additional context.

### What bias would look like
- Large **selection rate gaps** across groups indicate differences in decision frequency.
- Large **TPR gaps (equal opportunity gaps)** indicate that some groups are more likely to receive correct positive predictions.

The synthetic data is intentionally constructed with **group-dependent offsets**, so the baseline model is expected to show measurable group differences.

## 2) Explainability (feature importance + SHAP)

The notebook computes:
- **Permutation feature importance** (global): measures how performance changes when a feature is permuted.
- **Coefficient magnitudes** (global): from the logistic regression weights after one-hot encoding.
- **SHAP explanations** (global + local): SHAP values are computed on the transformed one-hot feature space to show which input features most influence a prediction.

SHAP is used to support fairness analysis by answering questions like:
- Which encoded features (e.g., `gender_F`, `race_B`, or numeric variables) contribute most to predicted risk?
- For an individual instance, which features pushed the prediction toward/away from the positive class?

This helps distinguish whether group-level effects come from sensitive attributes themselves or from proxy features.

## 3) Mitigation strategies implemented

### Mitigation A (train-time): remove sensitive features
**Approach:** Train a new logistic regression model using only numeric features (`x1`, `x2`, `x3`) and **exclude** `gender`, `age_group`, and `race`.

**Why it helps:**
- Removes direct information about protected attributes.
- Reduces the chance that the model learns group-specific decision rules directly from sensitive fields.

**Trade-off:**
- Accuracy/ROC-AUC and other metrics can change.
- If proxies remain strongly correlated with sensitive attributes, gaps may persist.

The notebook compares baseline vs. variant group metrics (especially TPR gap).

---

### Mitigation B (post-processing): group-wise threshold adjustment
**Approach:** After training a baseline model, apply **per-group thresholds** for a chosen sensitive attribute (implemented for `gender` in the notebook).

**Goal:** Reduce **equal opportunity (TPR) gaps** by selecting thresholds for each group so their TPR moves closer to a target group’s TPR.

**Why it helps:**
- Requires no retraining.
- Directly targets the fairness metric (TPR gap) after observing where the model under- or over-predicts positives.

**Trade-off:**
- Post-processing can change selection rates.
- Thresholding typically changes the distribution of predicted outcomes and should be validated carefully.

The notebook prints bias reports and TPR-gap comparisons before vs. after thresholding.

## 4) Practical recommendations

1. **Use explainability to guide mitigation**
   - If SHAP shows heavy reliance on sensitive one-hot features, prefer **Mitigation A** (feature removal / debiasing).
   - If sensitive attributes are low importance but proxies dominate, consider additional preprocessing (e.g., proxy detection, feature selection, or reweighing).

2. **Adopt an iterative fairness loop**
   - Baseline → compute bias metrics → apply mitigation → recompute fairness metrics.
   - Track both **fairness and performance** so you can justify trade-offs.

3. **Prefer training-time mitigation when feasible**
   - Removing sensitive attributes is straightforward and often effective.

4. **Use thresholding for metric-specific fairness goals**
   - If equal opportunity is the main concern and gaps remain, apply group-wise thresholds.

5. **Validate under realistic distributions**
   - Synthetic data is useful for demonstration, but real deployments should re-run these checks on the actual dataset with sufficiently large group counts.

## 5) Deliverables mapping

- `notebook_fairness_shap_lime.ipynb`:
  - dataset creation
  - feature importances
  - SHAP plots (summary + dependence + local explanation)
  - bias checks across sensitive groups
  - both mitigation steps and comparison

- This file: `writeup_bias_mitigation.md`:
  - short write-up of bias checks and mitigations

