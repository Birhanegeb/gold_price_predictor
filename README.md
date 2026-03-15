# Gold Price Direction Predictor — Automated ETL & ML Pipeline with Apache Airflow

> Predicts whether gold prices will go **up or down** the next trading day by combining financial market data with real-time war/conflict news sentiment analysis — fully automated with Apache Airflow.

---

## Overview

This project builds a production-style **ETL + Machine Learning pipeline** that runs on a weekly schedule. It fetches gold futures prices from Yahoo Finance, collects war-related news from NYT RSS feeds, computes sentiment scores, and retrains a Random Forest classifier automatically every week.

```
fetch_gold_prices ──┐
                    ├──► compute_sentiment_and_merge ──► train_model
fetch_war_news    ──┘
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Pipeline orchestration | Apache Airflow |
| Financial data | yfinance (`GC=F` gold futures) |
| News data | NYT RSS feeds + feedparser |
| Sentiment analysis | TextBlob |
| ML model | scikit-learn RandomForestClassifier |
| Model persistence | joblib |
| Data processing | pandas |

---

## Repository Structure

```
gold-price-predictor/
│
├── dags/
│   └── gold_war_etl_dag.py        # Airflow DAG — wires all tasks together
│
├── tasks/
│   ├── __init__.py
│   ├── fetch_gold.py              # Task 1: download gold prices
│   ├── fetch_news.py              # Task 2: fetch war news from NYT RSS
│   ├── merge_sentiment.py         # Task 3: sentiment analysis + merge
│   └── train_model.py             # Task 4: train and save ML model
│
├── data/
│   ├── raw/                       # Fetched data (excluded from git)
│   ├── processed/                 # Merged training data (excluded from git)
│   └── samples/                   # Small sample CSVs (committed to git)
│
├── models/                        # Saved model files
│
├── notebooks/
│   └── analysis.ipynb             # Exploratory data analysis
│
├── tests/
│   └── test_model.py              # Load model and report accuracy
│
├── config.py                      # All settings: paths, symbols, keywords
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How It Works

### Task 1 — `fetch_gold_prices`
Downloads daily gold futures (`GC=F`) from `2024-01-01` to today using `yfinance`.
Saves `open`, `high`, `low`, `close` to `data/raw/gold_prices.csv`.

### Task 2 — `fetch_war_news`
Fetches NYT RSS feeds (World + HomePage). Filters articles by war/conflict keywords.
Saves matching articles to `data/raw/war_news.csv`.

> **Note:** RSS feeds return only the latest ~30–50 articles per feed. Most historical training rows will have `sentiment = 0` as a result. A production version would use the NYT Article Search API for historical backfill.

### Task 3 — `compute_sentiment_and_merge`
Runs **TextBlob polarity analysis** on each article title. Aggregates daily sentiment (mean polarity + article count). Left-joins with gold prices so every trading day has a row. Creates the binary target:
- `1` → next day close **higher** (price goes UP)
- `0` → next day close **lower**  (price goes DOWN)

Saves to `data/processed/training_data.csv`.

### Task 4 — `train_model`
Trains a **Random Forest classifier** on three features: `close`, `sentiment_mean`, `news_count`. Uses `shuffle=False` to preserve time order and prevent data leakage. Saves trained model to `models/gold_model.pkl`.

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/gold-price-predictor.git
cd gold-price-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Deploy to Airflow
```bash
# Copy DAG file
cp dags/gold_war_etl_dag.py ~/airflow/dags/

# Copy tasks package and config — Airflow needs these alongside the DAG
cp -r tasks/ ~/airflow/dags/tasks/
cp config.py ~/airflow/dags/
```

### 4. Start Airflow (local development)
```bash
airflow standalone
# Open http://localhost:8080
```

### 5. Trigger the pipeline
Airflow UI → find `gold_war_etl` → click **▶ Trigger DAG**

Output files:
```
data/raw/gold_prices.csv
data/raw/war_news.csv
data/processed/training_data.csv
models/gold_model.pkl
```

---

## Testing the Model

Run with defaults from `config.py`:
```bash
python tests/test_model.py
```

Or specify paths manually:
```bash
python tests/test_model.py \
  --model models/gold_model.pkl \
  --data  data/processed/training_data.csv
```

Example output:
```
✅ Model loaded: models/gold_model.pkl
✅ Data loaded: 314 rows

==================================================
        MODEL EVALUATION RESULTS
==================================================
  Rows evaluated : 314
  Features used  : ['close', 'sentiment_mean', 'news_count']
  Accuracy       : 0.5714  (57.1%)
==================================================
```

---

## Pipeline Schedule

The DAG runs **every Sunday at midnight** (`@weekly`). Each run:
1. Fetches all gold prices from `2024-01-01` to today (full refresh)
2. Fetches the latest war news from NYT RSS
3. Retrains the model on the most up-to-date data
4. Overwrites the previous model file with the new version

For local development, use **▶ Trigger DAG** in the Airflow UI to run manually at any time without waiting for Sunday.

---

## Design Decisions

**Why `shuffle=False` in train/test split?**
This is a time-series problem. Shuffling would let the model train on future dates when evaluating past ones, giving fake high accuracy. Keeping chronological order gives an honest result.

**Why Random Forest?**
Handles small datasets (~300 rows) well, robust to noisy features, requires minimal tuning.

**Why is sentiment a weak feature here?**
NYT RSS only returns recent articles so most historical rows have `sentiment_mean = 0`. A production version would backfill using the NYT Article Search API.

---

## Gold API Used
`yfinance` — free, no API key required. Symbol: `GC=F` (gold futures)

## License
For educational use. News data sourced from NYT RSS — see [NYT Terms of Use](https://help.nytimes.com/hc/en-us/articles/115014893428).
