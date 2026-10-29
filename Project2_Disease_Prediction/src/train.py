"""
Multi-class model training and algorithmic comparison for Disease Prediction System.
Evaluates Random Forest, Naive Bayes, Decision Tree, SVM, KNN, and MLP Neural Network.
"""

import os
import time
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

from data_loader import load_disease_data


def train_and_compare_disease_models():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    X_train, X_test, y_train, y_test, label_encoder, symptom_cols, metadata = load_disease_data()

    print("=" * 65)
    print(">>> Training & Comparing Multi-Class Classification Models for Disease Prediction")
    print(f"Symptoms: {len(symptom_cols)} | Diseases: {len(label_encoder.classes_)} | Train: {len(y_train)} | Test: {len(y_test)}")
    print("=" * 65)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Decision Tree": DecisionTreeClassifier(max_depth=14, random_state=42),
        "Support Vector Machine": SVC(kernel="linear", probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Multi-Layer Perceptron": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42)
    }

    results = {}
    best_model_name = None
    best_acc = -1.0
    trained_model_objects = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        start_time = time.time()
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"\n[{name}]")
        print(f"  CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
        print(f"  Test Accuracy: {acc*100:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f} | Time: {train_time:.2f}s")

        results[name] = {
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
            "test_accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "training_time_sec": round(float(train_time), 3)
        }
        trained_model_objects[name] = model

        if acc > best_acc:
            best_acc = acc
            best_model_name = name

    print(f"\n{'*' * 65}")
    print(f"Best Performing Algorithm: {best_model_name} (Accuracy: {best_acc*100:.2f}%)")
    print(f"{'*' * 65}")

    # Save best model
    best_model = trained_model_objects[best_model_name]
    model_save_path = os.path.join(models_dir, "best_disease_model.joblib")
    joblib.dump(best_model, model_save_path)
    print(f"[Saved] Best Model: {model_save_path}")

    # Save label encoder
    encoder_save_path = os.path.join(models_dir, "disease_label_encoder.joblib")
    joblib.dump(label_encoder, encoder_save_path)
    print(f"[Saved] Disease Label Encoder: {encoder_save_path}")

    # Save symptom features
    symptom_path = os.path.join(models_dir, "symptom_features.json")
    with open(symptom_path, "w") as f:
        json.dump(symptom_cols, f, indent=4)
    print(f"[Saved] Symptom Features List: {symptom_path}")

    # Save comparison summary
    comparison_summary = {
        "best_model": best_model_name,
        "models_comparison": results,
        "metadata": metadata
    }
    summary_path = os.path.join(models_dir, "disease_model_comparison.json")
    with open(summary_path, "w") as f:
        json.dump(comparison_summary, f, indent=4)
    print(f"[Saved] Comparison JSON: {summary_path}")

    return trained_model_objects, results, best_model_name, (X_test, y_test, label_encoder, symptom_cols)


if __name__ == "__main__":
    train_and_compare_disease_models()
