# =============================================================================
# config.py  (project root)
# Central configuration for the Gold & War News ML Pipeline.
# All file paths are relative to the project root — no hardcoded /data paths.
# =============================================================================

import os

# ── Project root ───────────────────────────────────────────────────────────────
# os.path.abspath(__file__) = full path to this config.py file
# os.path.dirname(...)      = folder that contains config.py = project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data directories ───────────────────────────────────────────────────────────
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")        # fetched raw data
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")  # merged + cleaned
SAMPLES_DIR   = os.path.join(BASE_DIR, "data", "samples")    # small sample CSVs
MODEL_DIR     = os.path.join(BASE_DIR, "models")             # saved model files

# ── Output file paths ──────────────────────────────────────────────────────────
GOLD_CSV     = os.path.join(RAW_DIR,       "gold_prices.csv")
NEWS_CSV     = os.path.join(RAW_DIR,       "war_news.csv")
TRAINING_CSV = os.path.join(PROCESSED_DIR, "training_data.csv")
MODEL_PKL    = os.path.join(MODEL_DIR,     "gold_model.pkl")

# ── Gold price settings ────────────────────────────────────────────────────────
GOLD_SYMBOL     = "GC=F"        # Yahoo Finance ticker for gold futures
GOLD_START_DATE = "2024-01-01"  # Fetch all prices from this date to today

# ── NYT RSS feeds ──────────────────────────────────────────────────────────────
NYT_RSS_FEEDS = [
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
]

# Keywords used to filter war/conflict-related articles
WAR_KEYWORDS = {
    "war", "conflict", "attack", "military", "invasion",
    "troops", "missile", "combat", "battle", "offensive",
    "ceasefire", "airstrike", "bombing", "casualties", "soldier",
}

# ── ML settings ────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = ["close", "sentiment_mean", "news_count"]
TEST_SIZE       = 0.2   # 20% of data held out for testing
RANDOM_STATE    = 42    # Fixed seed so results are reproducible
N_ESTIMATORS    = 100   # Number of trees in the Random Forest
