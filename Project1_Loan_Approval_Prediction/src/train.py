"""
Algorithm training and comparative benchmarking for Loan Approval Prediction.
Evaluates Logistic Regression, Random Forest, Gradient Boosting, Decision Tree, SVM, and KNN.
"""

import os
import time
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

from data_loader import load_and_preprocess_loan_data


def train_and_compare_loan_models():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    X_train, X_test, y_train, y_test, preprocessor, feature_names, metadata = load_and_preprocess_loan_data()

    print("=" * 65)
    print(">>> Training & Comparing Machine Learning Models for Loan Approval")
    print(f"Train samples: {len(y_train)} | Test samples: {len(y_test)} | Features: {len(feature_names)}")
    print("=" * 65)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Support Vector Machine": SVC(probability=True, kernel="rbf", C=1.0, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7)
    }

    results = {}
    best_model_name = None
    best_f1 = -1.0
    trained_model_objects = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        start_time = time.time()
        # 5-fold cross validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        
        # Fit on training data
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict on unseen test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        print(f"\n[{name}]")
        print(f"  CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
        print(f"  Test Accuracy: {acc*100:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        results[name] = {
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
            "test_accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "training_time_sec": round(float(train_time), 3)
        }
        trained_model_objects[name] = model

        # Track best model by F1-score & ROC-AUC
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    print(f"\n{'*' * 65}")
    print(f"Best Performing Algorithm: {best_model_name} (F1-Score: {best_f1:.4f})")
    print(f"{'*' * 65}")

    # Save best model
    best_model = trained_model_objects[best_model_name]
    best_model_path = os.path.join(models_dir, "best_loan_model.joblib")
    joblib.dump(best_model, best_model_path)
    print(f"[Saved] Best Model: {best_model_path}")

    # Save preprocessor pipeline
    preprocessor_path = os.path.join(models_dir, "loan_preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"[Saved] Preprocessor: {preprocessor_path}")

    # Save feature names
    features_path = os.path.join(models_dir, "loan_feature_names.json")
    with open(features_path, "w") as f:
        json.dump(feature_names, f, indent=4)

    # Save comparison summary
    comparison_summary = {
        "best_model": best_model_name,
        "models_comparison": results,
        "metadata": metadata
    }
    summary_path = os.path.join(models_dir, "loan_model_comparison.json")
    with open(summary_path, "w") as f:
        json.dump(comparison_summary, f, indent=4)
    print(f"[Saved] Comparison JSON: {summary_path}")

    return trained_model_objects, results, best_model_name, (X_test, y_test, feature_names)


if __name__ == "__main__":
    train_and_compare_loan_models()
