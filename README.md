# Gold Price Direction Predictor - Real-Data ML Pipeline with Apache Airflow

> An end-to-end, real-data machine-learning model that predicts whether gold prices will go **up or down** the next trading day using market data and war/conflict news sentiment, orchestrated with Apache Airflow.

---

## Overview

It demonstrates a complete **data engineering and machine-learning workflow** using live external data rather than a fixed toy dataset. Apache Airflow orchestrates the pipeline: it fetches gold futures prices from Yahoo Finance and war-related articles from NYT RSS feeds, prepares and joins the data, computes sentiment features, then trains and evaluates a Random Forest classifier.

Need a hands-on Apache Airflow or machine-learning project? You can clone this repository, run it with real external data, and adapt or modify the code for learning and portfolio projects under the MIT License. See [LICENSE](./LICENSE) for the terms.

### What functionality is included

- Building an Airflow DAG with independent data-fetch tasks and dependent processing/training tasks.
- Collecting and cleaning data from real-world APIs and RSS feeds.
- Creating features and next-day labels from chronological market data.
- Evaluating on a later, held-out time period instead of scoring the training rows.
- Extending the pipeline with historical news, model tracking, monitoring, or deployment.

Market and news sources can change availability, coverage, or terms. The current RSS feeds provide limited recent news, so historical sentiment coverage is sparse; see the limitation below. Treat model results as educational experiments, not investment advice.

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
│   ├── test_model.py              # CLI entry point for held-out evaluation
│   └── test_training_data.py      # Tests for time split and next-day labels
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
Trains a **Random Forest classifier** on three features: `close`, `sentiment_mean`, `news_count`. It sorts rows by date, trains on the earliest 80%, and evaluates on the latest 20% without shuffling. The saved model is fitted only on the training portion.

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Birhanegeb/gold-price-predictor.git
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

The evaluation command applies the same chronological split as training and scores only the latest 20% of the rows. These held-out dates are not used to fit the model.

Example output (your results will vary with the fetched data):
```text
==================================================
   HELD-OUT MODEL EVALUATION
==================================================
  Evaluation set : latest chronological 20%
  Held-out rows  : number depends on dataset size
  Features used  : ['close', 'sentiment_mean', 'news_count']
  Accuracy       : depends on fetched data
==================================================
```

Run the unit tests for the chronological split and target creation:
```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Pipeline Schedule

The DAG runs every Sunday at midnight (`@weekly`). Each run:
1. Fetches all gold prices from `2024-01-01` to today (full refresh)
2. Fetches the latest war news from NYT RSS
3. Retrains the model on the most up-to-date data
4. Overwrites the previous model file with the new version

For local development, use **▶ Trigger DAG** in the Airflow UI to run manually at any time without waiting for Sunday.

---

## Design Decisions

**Why a chronological train/test split?**
This is a time-series problem. The model trains on earlier dates and evaluation uses only later dates. Shuffling would let the model train on future dates when evaluating past ones, giving fake high accuracy.

**Why Random Forest?**
Handles small datasets (~300 rows) well, robust to noisy features, requires minimal tuning.

**Why is sentiment a weak feature here?**
NYT RSS only returns recent articles so most historical rows have `sentiment_mean = 0`. A production version would backfill using the NYT Article Search API.

---

## Gold API Used
`yfinance` — free, no API key required. Symbol: `GC=F` (gold futures)

## Data Sources

- Gold futures prices (`GC=F`): [Yahoo Finance](https://finance.yahoo.com/quote/GC=F/) via the `yfinance` Python package.
- War/conflict news: [NYT World RSS feed](https://rss.nytimes.com/services/xml/rss/nyt/World.xml) and [NYT Home Page RSS feed](https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml).

The pipeline retrieves data from these services at runtime; it does not include or redistribute their datasets. Data availability, usage, and redistribution are governed by the providers' terms. Review the [NYT Terms of Use](https://help.nytimes.com/hc/en-us/articles/115014893428) and [Yahoo Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html) before using or redistributing source data.

## License

This repository's code and documentation are licensed under the [MIT License](./LICENSE). This license does not cover third-party data or services.
