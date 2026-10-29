"""
Data loading and preprocessing module for Disease Prediction System.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_disease_data(
    dataset_path: str = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder, List[str], Dict[str, Any]]:
    """
    Loads symptom-disease dataset, extracts features and target classes,
    applies label encoding, and partitions into stratified train/test sets.
    """
    if dataset_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        excel_path = os.path.join(base_dir, "datasets", "disease_prediction_data.xlsx")
        csv_path = os.path.join(base_dir, "datasets", "disease_prediction_data.csv")
        dataset_path = excel_path if os.path.exists(excel_path) else csv_path

    if dataset_path.endswith(".xlsx"):
        df = pd.read_excel(dataset_path, sheet_name="Symptom_Disease_Matrix")
    else:
        df = pd.read_csv(dataset_path)

    # Features and Target
    feature_cols = [c for c in df.columns if c != "prognosis"]
    X = df[feature_cols].values
    y_raw = df["prognosis"].values

    # Encode disease labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    metadata = {
        "total_samples": len(df),
        "total_symptoms": len(feature_cols),
        "total_diseases": len(label_encoder.classes_),
        "disease_list": label_encoder.classes_.tolist(),
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    return X_train, X_test, y_train, y_test, label_encoder, feature_cols, metadata


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, le, s_cols, meta = load_disease_data()
    print("Disease Dataset Summary:")
    print(f"Total Samples: {meta['total_samples']} | Diseases: {meta['total_diseases']} | Symptoms: {meta['total_symptoms']}")
    print(f"Sample Diseases: {meta['disease_list'][:5]}")
