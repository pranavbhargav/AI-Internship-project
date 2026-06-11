import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)

def main():
    print("=" * 80)
    print("EXECUTE CLASSIFICATION MODEL COMPARISON")
    print("=" * 80)

    # 1) Load data
    print("[1/5] Loading Breast Cancer Wisconsin Diagnostic dataset...")
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    print(f"  X shape: {X.shape}")
    print(f"  y distribution: Malignant (0)={np.bincount(y)[0]}, Benign (1)={np.bincount(y)[1]}")

    # 2) Preprocessing + Train/Test split
    print("\n[2/5] Splitting data into train/test sets (80/20 stratified split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train size: {X_train.shape[0]} samples")
    print(f"  Test size:  {X_test.shape[0]} samples")

    # 3) Define algorithms
    print("\n[3/5] Instantiating models (Logistic Regression & Random Forest)...")
    log_reg = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=5000, solver='lbfgs'))
    ])

    rf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1))
    ])

    models = {
        'Logistic Regression': log_reg,
        'Random Forest': rf,
    }

    # 4) Cross-validation
    print("\n[4/5] Performing 5-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    for name, model in models.items():
        y_prob_cv = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
        y_pred_cv = (y_prob_cv >= 0.5).astype(int)
        cv_results[name] = {
            'accuracy': accuracy_score(y, y_pred_cv),
            'precision': precision_score(y, y_pred_cv, zero_division=0),
            'recall': recall_score(y, y_pred_cv, zero_division=0),
            'f1': f1_score(y, y_pred_cv, zero_division=0),
            'roc_auc': roc_auc_score(y, y_prob_cv),
        }
    
    cv_df = pd.DataFrame(cv_results).T
    print("\n--- 5-Fold CV Metrics ---")
    print(cv_df.to_string())

    # 5) Train & Evaluate on Test Set
    print("\n[5/5] Training on X_train and evaluating on held-out X_test...")
    test_metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        test_metrics[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob),
        }
        
        # Save plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=axes[0], values_format='d', cmap='Blues')
        axes[0].set_title(f'Confusion Matrix - {name}')
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[1], name=name)
        axes[1].set_title(f'ROC Curve - {name}')
        plt.tight_layout()
        img_path = f"{name.replace(' ', '_').lower()}_evaluation.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved evaluation plot for {name} to {img_path}")

    # Random Forest Feature Importance
    rf_fitted = models['Random Forest']
    rf_clf = rf_fitted.named_steps['clf']
    importances = rf_clf.feature_importances_
    imp_df = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df.head(15), x='importance', y='feature', orient='h')
    plt.title('Top 15 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved feature importance plot to rf_feature_importance.png")

    test_df = pd.DataFrame(test_metrics).T
    print("\n--- Held-Out Test Set Metrics ---")
    print(test_df.to_string())

    best_model = test_df['roc_auc'].idxmax()
    print("\n" + "=" * 80)
    print(f"RECOMMENDED MODEL: {best_model}")
    print(f"Test ROC-AUC: {test_df.loc[best_model, 'roc_auc']:.4f} | Test Accuracy: {test_df.loc[best_model, 'accuracy']:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
