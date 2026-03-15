# =============================================================================
# tasks/fetch_news.py
# Task 2: Fetch war-related articles from NYT RSS feeds.
# Output: data/raw/war_news.csv
# =============================================================================

import logging
import os
import sys
from datetime import datetime

import feedparser
import pandas as pd

# Add project root to path so config.py can be found from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NEWS_CSV, NYT_RSS_FEEDS, WAR_KEYWORDS, RAW_DIR


def fetch_war_news():
    """
    Pull articles from NYT RSS feeds and keep only those that contain
    at least one war/conflict keyword in the title or summary.
    Saves output to: data/raw/war_news.csv

    Columns: date, title, summary

    Note: RSS feeds return only the latest ~30-50 articles.
    Most historical training rows will have sentiment = 0 as a result.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    records = []

    for feed_url in NYT_RSS_FEEDS:
        logging.info(f"Fetching RSS: {feed_url}")

        try:
            feed = feedparser.parse(feed_url)

            # feedparser sets bozo=True when the feed is malformed
            if feed.bozo:
                logging.warning(f"Feed may be malformed: {feed.bozo_exception}")

            if not feed.entries:
                logging.warning(f"No entries found in: {feed_url}")
                continue

            for entry in feed.entries:
                title   = entry.get("title",   "")
                summary = entry.get("summary", "")
                text    = (title + " " + summary).lower()

                # Only keep war/conflict-related articles
                if not any(kw in text for kw in WAR_KEYWORDS):
                    continue

                pub = entry.get("published_parsed")
                if not pub:
                    logging.warning(f"Missing date for: '{title}' — skipping")
                    continue

                records.append({
                    "date":    datetime(*pub[:3]).strftime("%Y-%m-%d"),
                    "title":   title,
                    "summary": summary,
                })

        except Exception as e:
            # Log and continue — don't stop the pipeline for one bad feed
            logging.error(f"❌ Error fetching '{feed_url}': {e}")
            continue

    if not records:
        raise RuntimeError(
            "No war-related articles collected from any feed. "
            "Check your internet connection and the NYT RSS URLs."
        )

    df = pd.DataFrame(records)
    df.to_csv(NEWS_CSV, index=False)
    logging.info(f"✅ {len(df)} articles saved → {NEWS_CSV}")
