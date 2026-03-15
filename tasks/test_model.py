# =============================================================================
# tests/test_model.py
# Load the trained gold prediction model and report its accuracy.
#
# Usage:
#   python tests/test_model.py
#   python tests/test_model.py --model models/gold_model.pkl --data data/processed
# =============================================================================

import argparse
import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Add project root to path so config.py can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURE_COLUMNS, MODEL_PKL, TRAINING_CSV


def load_model(model_path: str):
    """Load model from .pkl file. Exits with a clear message if file is missing."""
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: '{model_path}'")
        print("   Trigger the Airflow pipeline at least once first.")
        sys.exit(1)
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load training_data.csv. Exits with a clear message if file is missing."""
    if not os.path.exists(csv_path):
        print(f"❌ Data file not found: '{csv_path}'")
        print("   Run the full Airflow pipeline first.")
        sys.exit(1)
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Data loaded: {len(df)} rows from '{csv_path}'")
        return df
    except Exception as e:
        print(f"❌ Failed to read data: {e}")
        sys.exit(1)


def validate_columns(df: pd.DataFrame):
    """Check all required feature columns exist in the DataFrame."""
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        print(f"❌ Missing columns in CSV: {missing}")
        print(f"   Expected: {FEATURE_COLUMNS}")
        sys.exit(1)
    if "target" not in df.columns:
        print("❌ Column 'target' not found in training_data.csv")
        sys.exit(1)


def evaluate(model, df: pd.DataFrame):
    """Run predictions and print accuracy + full classification report."""
    X        = df[FEATURE_COLUMNS]
    y        = df["target"]
    preds    = model.predict(X)
    accuracy = accuracy_score(y, preds)

    print()
    print("=" * 50)
    print("        MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Rows evaluated : {len(df)}")
    print(f"  Features used  : {FEATURE_COLUMNS}")
    print(f"  Accuracy       : {accuracy:.4f}  ({accuracy * 100:.1f}%)")
    print("=" * 50)
    print()
    print(classification_report(y, preds, target_names=["Down (0)", "Up (1)"]))

    # Plain-language interpretation
    if accuracy >= 0.60:
        print("✅ Good accuracy for a financial prediction model.")
    elif accuracy >= 0.50:
        print("⚠️  Above 50% but room to improve.")
        print("   Tip: try adding lagged prices or moving average features.")
    else:
        print("❌ Below 50% — worse than random guessing.")
        print("   Tip: confirm shuffle=False was used in train/test split.")


def main():
    parser = argparse.ArgumentParser(description="Test the gold price prediction model")
    parser.add_argument(
        "--model",
        default=MODEL_PKL,   # uses path from config.py if not specified
        help=f"Path to .pkl model file (default: {MODEL_PKL})",
    )
    parser.add_argument(
        "--data",
        default=TRAINING_CSV,  # uses path from config.py if not specified
        help=f"Path to training_data.csv (default: {TRAINING_CSV})",
    )
    args = parser.parse_args()

    model = load_model(args.model)
    df    = load_data(args.data)
    validate_columns(df)
    evaluate(model, df)


if __name__ == "__main__":
    main()
