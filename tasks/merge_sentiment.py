# =============================================================================
# tasks/merge_sentiment.py
# Task 3: Compute news sentiment scores and merge with gold prices.
# Input:  data/raw/gold_prices.csv + data/raw/war_news.csv
# Output: data/processed/training_data.csv
# =============================================================================

import logging
import os
import sys

import pandas as pd
from textblob import TextBlob

# Add project root to path so config.py can be found from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GOLD_CSV, NEWS_CSV, TRAINING_CSV, PROCESSED_DIR


def safe_sentiment(text: str) -> float:
    """
    Return TextBlob polarity score for a string.
    Returns 0.0 if the text is empty or not a string.
    Polarity ranges from -1.0 (negative) to +1.0 (positive).
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return TextBlob(text).sentiment.polarity


def add_next_day_target(df: pd.DataFrame) -> pd.DataFrame:
    """Label each row by the next close and exclude rows without a next day."""
    next_close = df["close"].shift(-1)
    result = df.copy()
    result["target"] = next_close.gt(result["close"]).where(next_close.notna()).astype("Int64")
    return result.dropna(subset=["target"])


def compute_sentiment_and_merge():
    """
    1. Load gold_prices.csv  (data/raw/)
    2. Load war_news.csv     (data/raw/)
    3. Run TextBlob sentiment on each article title
    4. Aggregate to one row per date (mean polarity + article count)
    5. Left-join with gold prices so every trading day has a row
       (days with no news get sentiment_mean=0, news_count=0)
    6. Create binary target column:
           1 = next day close > today close  → price goes UP
           0 = next day close ≤ today close  → price goes DOWN
    7. Save result to data/processed/training_data.csv
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ── Load gold prices ───────────────────────────────────────────────
    if not os.path.exists(GOLD_CSV):
        raise FileNotFoundError(
            f"Gold prices file not found: '{GOLD_CSV}'\n"
            "Run task fetch_gold_prices first."
        )

    # ── Load war news ──────────────────────────────────────────────────
    if not os.path.exists(NEWS_CSV):
        raise FileNotFoundError(
            f"War news file not found: '{NEWS_CSV}'\n"
            "Run task fetch_war_news first."
        )

    gold = pd.read_csv(GOLD_CSV, parse_dates=["date"])
    news = pd.read_csv(NEWS_CSV, parse_dates=["date"])
    logging.info(f"Loaded {len(gold)} gold rows | {len(news)} news articles")

    # ── Sentiment per article ──────────────────────────────────────────
    news["sentiment"] = news["title"].apply(safe_sentiment)

    # ── Aggregate to one row per date ──────────────────────────────────
    daily = (
        news.groupby("date")
        .agg(
            sentiment_mean=("sentiment", "mean"),
            news_count=("sentiment", "count"),
        )
        .reset_index()
    )

    # ── Merge: left join keeps all trading days ────────────────────────
    merged = gold.merge(daily, on="date", how="left")
    merged["sentiment_mean"] = merged["sentiment_mean"].fillna(0.0)
    merged["news_count"]     = merged["news_count"].fillna(0).astype(int)

    # ── Create target column ───────────────────────────────────────────
    # IMPORTANT: sort by date first — never shuffle time-series data
    merged = merged.sort_values("date").reset_index(drop=True)
    merged = add_next_day_target(merged)

    if len(merged) < 10:
        raise ValueError(
            f"Only {len(merged)} rows after merging — too few to train a model. "
            "Check that gold_prices.csv and war_news.csv contain valid data."
        )

    merged.to_csv(TRAINING_CSV, index=False)
    logging.info(f"✅ {len(merged)} training rows saved → {TRAINING_CSV}")
