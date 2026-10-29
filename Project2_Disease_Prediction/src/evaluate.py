"""
Evaluation and visualization module for Disease Prediction System.
Produces algorithm benchmarks, multi-class confusion matrices, and clinical symptom rankings.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from train import train_and_compare_disease_models


def evaluate_and_plot_disease_models():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    models_dict, results, best_name, (X_test, y_test, label_encoder, symptom_cols) = train_and_compare_disease_models()

    best_model = models_dict[best_name]
    y_pred = best_model.predict(X_test)

    print("\n" + "=" * 65)
    print(f">>> Comprehensive Evaluation of Best Disease Model: {best_name}")
    print("=" * 65)

    disease_names = label_encoder.classes_
    report_str = classification_report(y_test, y_pred, target_names=disease_names, digits=4)
    print(report_str)

    # Save classification report
    report_file = os.path.join(outputs_dir, "disease_classification_report.txt")
    with open(report_file, "w") as f:
        f.write(f"Disease Prediction System - Evaluation Report ({best_name})\n")
        f.write("=" * 65 + "\n\n")
        f.write(report_str)

    # 1. Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(11, 5.5))
    names = list(results.keys())
    accs = [results[m]["test_accuracy"] * 100 for m in names]
    f1s = [results[m]["f1_score"] * 100 for m in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, accs, width, label="Test Accuracy (%)", color="#2980b9")
    bars2 = ax.bar(x + width/2, f1s, width, label="Weighted F1 (%)", color="#27ae60")

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Machine Learning Classification Algorithms - Disease Prediction", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(85, 103)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    comp_path = os.path.join(outputs_dir, "disease_model_comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"[Saved] Disease model comparison plot: {comp_path}")

    # 2. Multi-class Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=disease_names,
                yticklabels=disease_names,
                ax=ax, cbar=True)
    ax.set_title(f"Multi-Class Disease Confusion Matrix ({best_name})", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Disease", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual Condition", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)

    plt.tight_layout()
    cm_path = os.path.join(outputs_dir, "disease_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Saved] Multi-class confusion matrix: {cm_path}")

    # 3. Top Symptoms Importance (Random Forest)
    rf_model = models_dict.get("Random Forest", best_model)
    if hasattr(rf_model, "feature_importances_"):
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]

        top_symptoms = [symptom_cols[i].replace("_", " ").title() for i in indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 6.5))
        ax.barh(top_symptoms[::-1], top_importances[::-1], color="#e74c3c", height=0.6)
        ax.set_title("Top 15 Most Discriminative Diagnostic Symptoms", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Feature Importance Score (Gini)", fontsize=11, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        plt.tight_layout()
        sym_path = os.path.join(outputs_dir, "top_symptoms_importance.png")
        plt.savefig(sym_path, dpi=300)
        plt.close()
        print(f"[Saved] Top symptoms importance plot: {sym_path}")


if __name__ == "__main__":
    evaluate_and_plot_disease_models()
