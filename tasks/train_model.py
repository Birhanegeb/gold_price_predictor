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


def train_model():
    """
    1. Load data/processed/training_data.csv
    2. Split into train/test — shuffle=False preserves time order
       (shuffling would leak future prices into training = fake accuracy)
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

    # Confirm all required feature columns exist
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in training data: {missing}\n"
            f"Expected: {FEATURE_COLUMNS}"
        )

    X = df[FEATURE_COLUMNS]
    y = df["target"]

    # ── Train / test split ─────────────────────────────────────────────
    # shuffle=False is CRITICAL — this is time-series data.
    # Shuffling would allow the model to train on future dates,
    # which gives artificially high accuracy and is not honest evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        shuffle=False,
    )
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
        target_names=["Down (0)", "Up (1)"]
    ))

    if accuracy < 0.50:
        logging.warning(
            "⚠️  Accuracy is below 50% (worse than random guessing). "
            "Consider adding lagged prices or moving average features."
        )

    # ── Save model ─────────────────────────────────────────────────────
    joblib.dump(model, MODEL_PKL)
    logging.info(f"✅ Model saved → {MODEL_PKL}")
