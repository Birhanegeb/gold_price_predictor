# =============================================================================
# tasks/train_model.py
# Task 4: Train a Random Forest classifier and save the model.
# Input:  data/processed/training_data.csv
# Output: models/gold_model.pkl
# =============================================================================

import logging
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Add project root to path so config.py can be found from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FEATURE_COLUMNS, MODEL_DIR, MODEL_PKL,
    N_ESTIMATORS, RANDOM_STATE, TEST_SIZE, TRAINING_CSV,
)


def chronological_train_test_split(df: pd.DataFrame):
    """Split rows by date, keeping the latest TEST_SIZE fraction for evaluation."""
    if "date" not in df.columns:
        raise ValueError("Training data must include a 'date' column for chronological splitting.")

    ordered = df.assign(date=pd.to_datetime(df["date"], errors="raise"))
    ordered = ordered.sort_values("date").reset_index(drop=True)
    return train_test_split(ordered, test_size=TEST_SIZE, shuffle=False)


def train_model():
    """
    1. Load data/processed/training_data.csv
    2. Split chronologically, holding out the latest 20% for testing
    3. Train a Random Forest classifier
    4. Log accuracy and classification report
    5. Save trained model to models/gold_model.pkl
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load training data ─────────────────────────────────────────────
    if not os.path.exists(TRAINING_CSV):
        raise FileNotFoundError(
            f"Training data not found: '{TRAINING_CSV}'\n"
            "Run task compute_sentiment_and_merge first."
        )

    df = pd.read_csv(TRAINING_CSV)
    logging.info(f"Loaded {len(df)} training rows")

    required_columns = [*FEATURE_COLUMNS, "target", "date"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in training data: {missing}\n"
            f"Expected at minimum: {required_columns}"
        )

    # ── Train / test split ─────────────────────────────────────────────
    train_df, test_df = chronological_train_test_split(df)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["target"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["target"]
    logging.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # ── Train ──────────────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,  # fixed seed → same result every run
    )
    model.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────
    preds    = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    logging.info(f"✅ Test accuracy: {accuracy:.4f}")
    logging.info("\n" + classification_report(
        y_test, preds,
        labels=[0, 1],
        target_names=["Down (0)", "Up (1)"],
        zero_division=0,
    ))

    if accuracy < 0.50:
        logging.warning(
            "⚠️  Accuracy is below 50% (worse than random guessing). "
            "Consider adding lagged prices or moving average features."
        )

    # ── Save model ─────────────────────────────────────────────────────
    joblib.dump(model, MODEL_PKL)
    logging.info(f"✅ Model saved → {MODEL_PKL}")
