"""
Evaluation and visualization module for Loan Approval Prediction.
Generates model comparison plots, confusion matrix, ROC curves, and feature importance charts.
"""

import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

from train import train_and_compare_loan_models


def evaluate_and_plot_loan_models():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # Train and retrieve models
    models_dict, results, best_name, (X_test, y_test, feature_names) = train_and_compare_loan_models()

    best_model = models_dict[best_name]
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else y_pred

    print("\n" + "=" * 65)
    print(f">>> Detailed Evaluation of Best Model: {best_name}")
    print("=" * 65)

    report_str = classification_report(y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"], digits=4)
    print(report_str)

    # Save classification report
    report_file = os.path.join(outputs_dir, "classification_report.txt")
    with open(report_file, "w") as f:
        f.write(f"Loan Approval Prediction - Evaluation Report ({best_name})\n")
        f.write("=" * 65 + "\n\n")
        f.write(report_str)

    # 1. Plot Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    names = list(results.keys())
    accs = [results[m]["test_accuracy"] * 100 for m in names]
    f1s = [results[m]["f1_score"] * 100 for m in names]
    aucs = [results[m]["roc_auc"] * 100 for m in names]

    x = np.arange(len(names))
    width = 0.25

    rects1 = ax.bar(x - width, accs, width, label="Accuracy (%)", color="#3498db")
    rects2 = ax.bar(x, f1s, width, label="F1-Score (%)", color="#2ecc71")
    rects3 = ax.bar(x + width, aucs, width, label="ROC-AUC (%)", color="#9b59b6")

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Machine Learning Algorithms Benchmark - Loan Approval Prediction", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(60, 100)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    comp_path = os.path.join(outputs_dir, "model_comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"[Saved] Model comparison plot: {comp_path}")

    # 2. Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Rejected", "Approved"],
                yticklabels=["Rejected", "Approved"],
                ax=ax, cbar=False)
    ax.set_title(f"Confusion Matrix: {best_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground Truth Label", fontsize=11, fontweight="bold")

    plt.tight_layout()
    cm_path = os.path.join(outputs_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Saved] Confusion matrix plot: {cm_path}")

    # 3. Plot ROC Curves
    fig, ax = plt.subplots(figsize=(8, 7))
    for m_name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{m_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    roc_path = os.path.join(outputs_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[Saved] ROC curve plot: {roc_path}")

    # 4. Feature Importance (from Random Forest / Gradient Boosting)
    rf_model = models_dict.get("Random Forest", best_model)
    if hasattr(rf_model, "feature_importances_"):
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]

        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_features[::-1], top_importances[::-1], color="#1abc9c", height=0.6)
        ax.set_title("Top 10 Feature Importances (Random Forest)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Gini Importance Score", fontsize=11, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        plt.tight_layout()
        fi_path = os.path.join(outputs_dir, "feature_importance.png")
        plt.savefig(fi_path, dpi=300)
        plt.close()
        print(f"[Saved] Feature importance plot: {fi_path}")


if __name__ == "__main__":
    evaluate_and_plot_loan_models()
