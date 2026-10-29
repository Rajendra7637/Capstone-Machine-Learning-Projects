"""
Data loading, preprocessing, imputation, and feature engineering for Loan Approval Prediction.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_and_preprocess_loan_data(
    dataset_path: str = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, list, dict]:
    """
    Loads Loan dataset, performs full missing value handling, feature engineering,
    builds scikit-learn preprocessing ColumnTransformer pipeline, and returns train/test splits.
    """
    if dataset_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        excel_path = os.path.join(base_dir, "datasets", "loan_approval_data.xlsx")
        csv_path = os.path.join(base_dir, "datasets", "loan_approval_data.csv")
        dataset_path = excel_path if os.path.exists(excel_path) else csv_path

    if dataset_path.endswith(".xlsx"):
        df = pd.read_excel(dataset_path, sheet_name="Raw_Applicant_Data")
    else:
        df = pd.read_csv(dataset_path)

    # Missing values audit
    null_counts_before = df.isnull().sum().to_dict()

    # Feature Engineering
    # Compute Total Income and ratios
    df["Total_Income"] = df["ApplicantIncome"].fillna(df["ApplicantIncome"].median()) + \
                         df["CoapplicantIncome"].fillna(0)
    df["Loan_to_Income_Ratio"] = (df["LoanAmount"] * 1000 / df["Total_Income"]).clip(upper=15)
    df["Income_Log"] = np.log1p(df["Total_Income"])
    df["LoanAmount_Log"] = np.log1p(df["LoanAmount"].fillna(df["LoanAmount"].median()))

    # Target variable
    y = (df["Loan_Status"] == "Y").astype(int).values

    # Feature subsets
    numeric_features = [
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
        "Loan_Amount_Term", "Total_Income", "Loan_to_Income_Ratio",
        "Income_Log", "LoanAmount_Log"
    ]
    categorical_features = [
        "Gender", "Married", "Dependents", "Education",
        "Self_Employed", "Credit_History", "Property_Area"
    ]

    X = df[numeric_features + categorical_features].copy()

    # Transformers
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_features),
        ("cat", cat_pipeline, categorical_features)
    ])

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit preprocessor on train, transform both
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_encoded_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_encoded_names

    metadata = {
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "approval_rate": round(float(y.mean()) * 100, 2),
        "null_counts_before": null_counts_before,
        "feature_count": len(all_feature_names)
    }

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor, all_feature_names, metadata


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, prep, f_names, meta = load_and_preprocess_loan_data()
    print("Loan Dataset Preprocessed Successfully:")
    print(f"Train Shape: {X_tr.shape}, Test Shape: {X_te.shape}")
    print(f"Features ({len(f_names)}): {f_names[:5]}...")
    print(f"Approval Rate: {meta['approval_rate']}%")
