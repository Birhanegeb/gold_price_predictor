# =============================================================================
# dags/gold_war_etl_dag.py
# Airflow DAG — Gold Price & War News ML Pipeline
#
# This file only wires the tasks together.
# All logic lives in tasks/ and all settings live in config.py.
#
# Pipeline flow:
#   fetch_gold_prices ──┐
#                       ├──► compute_sentiment_and_merge ──► train_model
#   fetch_war_news    ──┘
#
# Schedule : every Sunday at midnight (@weekly)
# Manual   : Airflow UI → gold_war_etl → ▶ Trigger DAG
# =============================================================================

import os
import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Make project root importable ──────────────────────────────────────────────
# Airflow runs DAGs from its own dags/ folder.
# We add the project root (one level up from dags/) to sys.path so that
# both config.py and the tasks/ package can be imported correctly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tasks.fetch_gold      import fetch_gold_prices
from tasks.fetch_news      import fetch_war_news
from tasks.merge_sentiment import compute_sentiment_and_merge
from tasks.train_model     import train_model

# ── DAG ────────────────────────────────────────────────────────────────────────
default_args = {
    "owner":      "student",
    "start_date": datetime(2024, 1, 1),
    "retries":    1,   # retry once automatically if a task fails
}

with DAG(
    dag_id="gold_war_etl",
    default_args=default_args,
    schedule="@weekly",   # every Sunday at midnight
    catchup=False,        # ignore missed past runs when deploying
    description="Gold price direction predictor — ETL + ML pipeline",
    tags=["lecture6", "mid-semester", "gold", "ml", "etl"],
) as dag:

    t1_fetch_gold = PythonOperator(
        task_id="fetch_gold_prices",
        python_callable=fetch_gold_prices,
    )

    t2_fetch_news = PythonOperator(
        task_id="fetch_war_news",
        python_callable=fetch_war_news,
    )

    t3_merge = PythonOperator(
        task_id="compute_sentiment_and_merge",
        python_callable=compute_sentiment_and_merge,
    )

    t4_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    # t1 and t2 run in parallel → both must finish before t3 → then t4
    [t1_fetch_gold, t2_fetch_news] >> t3_merge >> t4_train
