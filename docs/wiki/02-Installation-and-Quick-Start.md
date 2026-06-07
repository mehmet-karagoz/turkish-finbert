# Installation and Quick Start

This page explains how to run the project locally.

## Requirements

- Python 3.10 or newer.
- `uv` package manager.
- PowerShell on Windows, or a compatible shell on Linux/macOS.

## Install Dependencies

```powershell
uv sync
```

For optional transformer training:

```powershell
uv sync --extra nlp
```

## Run Tests

```powershell
uv run pytest
```

Expected result:

```text
all tests pass
```

## Train The Baseline Model

The deploy workflow trains the model automatically, but local usage usually starts with:

```powershell
uv run train_model --input data/labels/labeled_news_master.csv --model-out models/master_baseline_sentiment.joblib --report-dir reports/master_baseline --split-strategy stratified
```

This creates:

```text
models/master_baseline_sentiment.joblib
reports/master_baseline/baseline_metrics.json
reports/master_baseline/baseline_test_predictions.csv
```

## Run The Daily Pipeline

```powershell
uv run daily_pipeline --kap-days 7 --update-history --baseline-lookback-days 365 --baseline-min-history 5
```

This command:

1. fetches recent KAP disclosures,
2. maps disclosures to tracked tickers,
3. prepares text,
4. scores news,
5. aggregates daily sentiment,
6. updates historical CSV files,
7. creates daily reports.

## Run A Specific Date

```powershell
uv run daily_pipeline --kap-from-date 2026-06-07 --kap-to-date 2026-06-07 --date 2026-06-07 --update-history --baseline-lookback-days 365 --baseline-min-history 5
```

Use this when you want to reproduce a past day or manually check a specific date.

## Generate Alerts From Existing Scored Files

```powershell
uv run daily_alerts --scored-news data/processed/kap_daily_scored_news.csv --daily-sentiment data/processed/kap_daily_sentiment.csv --baseline-daily-sentiment data/processed/kap_api_historical_daily_sentiment.csv --out-dir reports/daily_alerts
```

## Send Telegram Locally

Set environment variables:

```powershell
$env:TELEGRAM_BOT_TOKEN="BOT_TOKEN"
$env:TELEGRAM_CHAT_ID="CHAT_ID"
```

Then run:

```powershell
uv run daily_alerts --scored-news data/processed/kap_daily_scored_news.csv --daily-sentiment data/processed/kap_daily_sentiment.csv --baseline-daily-sentiment data/processed/kap_api_historical_daily_sentiment.csv --send-telegram
```

## Common Local Files

```text
data/raw/tickers_bist_kap.txt
data/labels/labeled_news_master.csv
models/master_baseline_sentiment.joblib
reports/daily_alerts/
```

## Recommended Local Workflow

```text
1. Pull latest repo.
2. uv sync
3. uv run pytest
4. Train model if needed.
5. Run daily_pipeline.
6. Inspect reports/daily_alerts.
```
