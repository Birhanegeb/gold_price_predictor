# =============================================================================
# tasks/fetch_gold.py
# Task 1: Download gold futures prices from Yahoo Finance.
# Output: data/raw/gold_prices.csv
# =============================================================================

import logging
import os
import sys
from datetime import datetime

import yfinance as yf

# Add project root to path so config.py can be found from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GOLD_CSV, GOLD_START_DATE, GOLD_SYMBOL, RAW_DIR, MODEL_DIR


def ensure_directories():
    """Create all required output directories if they do not exist yet."""
    os.makedirs(RAW_DIR,   exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def fetch_gold_prices():
    """
    Download daily gold futures prices (GC=F) from 2024-01-01 to today.
    Saves output to: data/raw/gold_prices.csv

    Columns: date, open, high, low, close
    """
    ensure_directories()
    today = datetime.today().strftime("%Y-%m-%d")

    logging.info(f"Downloading gold prices | {GOLD_START_DATE} → {today}")

    try:
        df = yf.download(GOLD_SYMBOL, start=GOLD_START_DATE, end=today, auto_adjust=True)

        if df.empty:
            raise ValueError(
                f"No data returned for symbol '{GOLD_SYMBOL}'. "
                "Check the symbol name and date range."
            )

        # Keep only price columns and rename to lowercase
        df = df[["Open", "High", "Low", "Close"]].rename(columns=str.lower)
        df.index.name = "date"
        df.index = df.index.strftime("%Y-%m-%d")

        df.to_csv(GOLD_CSV)
        logging.info(f"✅ {len(df)} rows saved → {GOLD_CSV}")

    except Exception as e:
        logging.error(f"❌ fetch_gold_prices failed: {e}")
        raise  # re-raise so Airflow marks the task as FAILED
