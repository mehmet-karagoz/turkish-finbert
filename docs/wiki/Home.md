# Wiki Home

This wiki explains how to use, operate, and extend the Turkish FinBERT-Like BIST Sentiment Alerts project.

## Pages

1. [Project Purpose](01-Project-Purpose.md)
2. [Installation and Quick Start](02-Installation-and-Quick-Start.md)
3. [Data, Labeling, and Model](03-Data-Labeling-and-Model.md)
4. [Daily Signal Logic](04-Daily-Signal-Logic.md)
5. [GitHub Actions Deployment](05-GitHub-Actions-Deployment.md)
6. [Telegram Bot and Commands](06-Telegram-Bot-and-Commands.md)
7. [Operations and Troubleshooting](07-Operations-and-Troubleshooting.md)

## Short Summary

The project is a daily Turkish financial sentiment and alerting pipeline for BIST stocks. It is designed to answer:

```text
Did anything meaningful happen today?
Which stocks should be manually checked?
Was the signal unusual compared with recent history?
Is there a possible market-wide event?
```

It is not a trading bot and it is not investment advice. It is a research and monitoring tool.

## Main User Flow

```text
1. Collect KAP/news text.
2. Prepare and label examples.
3. Train a baseline sentiment model.
4. Score daily news.
5. Aggregate stock and event-level sentiment.
6. Generate daily reports.
7. Send Telegram alerts.
8. Trigger the pipeline manually with Telegram commands when needed.
```

## Most Important Commands

```powershell
uv sync
uv run pytest
uv run train_model --input data/labels/labeled_news_master.csv --model-out models/master_baseline_sentiment.joblib --report-dir reports/master_baseline --split-strategy stratified
uv run daily_pipeline --kap-days 7 --update-history --baseline-lookback-days 365 --baseline-min-history 5
```

## Important Output Files

```text
reports/daily_alerts/YYYY-MM-DD_daily_alerts.md
reports/daily_alerts/YYYY-MM-DD_brief.md
reports/daily_alerts/YYYY-MM-DD_telegram.html
reports/daily_alerts/YYYY-MM-DD_event_signals.csv
data/processed/kap_api_historical_daily_sentiment.csv
```
