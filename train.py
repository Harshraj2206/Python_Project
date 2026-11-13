"""Model training script for AgriSmart crop recommendation.

Trains a Random Forest classifier on soil and climate features,
evaluates performance, computes per-crop typical ranges, and saves
the model bundle.
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split

from explain import FEATURES

# Relative to this file, so the script works no matter which folder it is started from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "agrismart.joblib")


def main():
    # 1. Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(
            f"Error: Dataset not found at '{DATASET_PATH}'.\n"
            "Please download the 'Crop Recommendation Dataset' by Atharva Ingle "
            "from Kaggle and place 'Crop_recommendation.csv' in the 'data/' directory."
        )
        sys.exit(1)

    print(f"Loading dataset from '{DATASET_PATH}'...")
    df = pd.read_csv(DATASET_PATH)

    X = df[FEATURES]
    y = df["label"]

    # 2. Stratified 80/20 train/test split
    # stratify=y preserves the class distribution (100 samples per crop)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Dataset split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # 3. Train Random Forest Classifier on training set only
    print("Training RandomForestClassifier(n_estimators=200, random_state=42)...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate on test set
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n" + "=" * 50)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print("=" * 50)
    print("\nClassification Report (Test Set):")
    print(report)

    # 5-fold cross-validation on training set
    print("Running 5-fold cross-validation on the training set...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    print(
        f"5-Fold Cross-Validation Accuracy: {cv_mean * 100:.2f}% "
        f"(+/- {cv_std * 100:.2f}%)\n"
    )

    # Feature importances sorted highest first
    raw_importances = dict(zip(FEATURES, model.feature_importances_))
    sorted_importances = dict(
        sorted(raw_importances.items(), key=lambda item: item[1], reverse=True)
    )

    print("Feature Importances (highest first):")
    for feat, imp in sorted_importances.items():
        print(f"  {feat:<12}: {imp:.4f}")

    # 5. Compute per-crop typical ranges (25th and 75th percentiles) from training set only
    train_df = X_train.copy()
    train_df["label"] = y_train

    crop_ranges = {}
    for crop, group in train_df.groupby("label"):
        crop_ranges[crop] = {}
        for feat in FEATURES:
            p25 = float(group[feat].quantile(0.25))
            p75 = float(group[feat].quantile(0.75))
            crop_ranges[crop][feat] = (p25, p75)

    # 6. Save bundle to model/agrismart.joblib
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {
        "model": model,
        "features": FEATURES,
        "crop_ranges": crop_ranges,
        "importances": {feat: float(imp) for feat, imp in sorted_importances.items()},
    }

    joblib.dump(bundle, MODEL_PATH)
    print(f"\nModel bundle successfully saved to '{MODEL_PATH}'.")


if __name__ == "__main__":
    main()
