# Supervised Classification Model Comparison Report

This report summarizes the construction, evaluation, and selection of a supervised machine learning model for predicting breast cancer diagnosis (malignant vs. benign) using the Breast Cancer Wisconsin (Diagnostic) dataset.

---

## 1. Dataset & Preprocessing

- **Dataset**: `sklearn.datasets.load_breast_cancer` (569 samples, 30 numeric features, binary classification target).
  - Malignant (Class 0): 212 samples (37.3%)
  - Benign (Class 1): 357 samples (62.7%)
- **Train/Test Split**: 80% training (455 samples) and 20% testing (114 samples) using a stratified split to maintain target class distribution.
- **Preprocessing Pipeline**: 
  - Standardized numeric features using `StandardScaler` to ensure zero mean and unit variance.
  - Implemented inside a scikit-learn `Pipeline` to prevent data leakage during training, cross-validation, and testing.

---

## 2. Models Compared

1. **Logistic Regression**: Linear classifier with L2 regularization (`solver='lbfgs'`, `max_iter=5000`). Features are scaled.
2. **Random Forest Classifier**: Tree-based ensemble model (`n_estimators=400`, `random_state=42`, `n_jobs=-1`). Features are scaled consistently to match the pipeline structure.

---

## 3. Evaluation Metrics

### 5-Fold Stratified Cross-Validation Results
Computed using `cross_val_predict` over the entire dataset (569 samples) to evaluate training stability:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.9736** | **0.9672** | **0.9916** | **0.9793** | **0.9947** |
| **Random Forest** | 0.9525 | 0.9583 | 0.9664 | 0.9623 | 0.9884 |

### Held-Out Test Set Results
Evaluated on the 20% test set (114 samples):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.9825** | **0.9861** | **0.9861** | **0.9861** | **0.9954** |
| **Random Forest** | 0.9561 | 0.9589 | **0.9722** | 0.9655 | 0.9932 |

---

## 4. Key Findings & Model Selection

### Performance & Recommendation
- **Best Model**: **Logistic Regression** is the recommended model.
- **Justification**: 
  - It consistently outperformed Random Forest across all validation and test metrics, achieving a test **Accuracy of 98.25%** and a test **ROC-AUC of 0.9954** (compared to Random Forest's 95.61% accuracy and 0.9932 ROC-AUC).
  - In a medical context, missing a malignant tumor (False Negative) is highly critical. Logistic Regression achieved a very high test recall of **98.61%**, making it highly reliable.
  - The Wisconsin Breast Cancer dataset is highly linearly separable, meaning linear decision boundaries are highly effective. Simple, regularized models like Logistic Regression generalize exceptionally well here, whereas tree-based models like Random Forests can over-partition the small feature space and suffer from slightly higher variance.

### Feature Importances (Random Forest)
The top five features driving the Random Forest predictions are:
1. **worst area** (13.56% importance)
2. **worst perimeter** (12.42% importance)
3. **worst concave points** (11.92% importance)
4. **worst radius** (8.98% importance)
5. **mean concave points** (8.79% importance)

These importances highlight that cell nucleus size parameters (`area`, `perimeter`, `radius`) and shapes (`concave points`) are the most critical diagnostic factors.

---

## 5. Artifacts and Visualization Links

The evaluation figures are saved in the project workspace and can be viewed below:
- **Notebook**: [classification_notebook.ipynb](file:///c:/Users/K%20Pranav%20Bhargav/OneDrive/Desktop/AIML%20internship%20Alfido%20tech/classification_notebook.ipynb)
- **Logistic Regression Evaluation (Confusion Matrix & ROC)**: [logistic_regression_evaluation.png](file:///c:/Users/K%20Pranav%20Bhargav/OneDrive/Desktop/AIML%20internship%20Alfido%20tech/logistic_regression_evaluation.png)
- **Random Forest Evaluation (Confusion Matrix & ROC)**: [random_forest_evaluation.png](file:///c:/Users/K%20Pranav%20Bhargav/OneDrive/Desktop/AIML%20internship%20Alfido%20tech/random_forest_evaluation.png)
- **Random Forest Feature Importance Chart**: [rf_feature_importance.png](file:///c:/Users/K%20Pranav%20Bhargav/OneDrive/Desktop/AIML%20internship%20Alfido%20tech/rf_feature_importance.png)
