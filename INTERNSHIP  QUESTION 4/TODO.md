# TODO - Fairness/Bias/Explainability (SHAP/LIME)

- [x] Create synthetic dataset with sensitive attributes (gender, age_group, race) + target label
- [x] Train a baseline ML model
- [x] Compute feature importances (permutation + model coefficients)
- [x] Create SHAP explanations (global + local) and interpretation plots
- [x] Bias checks across sensitive groups (selection rate + TPR + group metrics)
- [x] Mitigation steps:
  - [x] Train a variant excluding sensitive features
  - [x] Apply group-wise thresholding to reduce equal opportunity gaps
  - [x] Summarize mitigation impact

- [x] Produce `notebook_fairness_shap_lime.ipynb`
- [x] Produce `writeup_bias_mitigation.md`
- [x] Sanity-run notebook cells (best-effort) via a quick import/lint check



