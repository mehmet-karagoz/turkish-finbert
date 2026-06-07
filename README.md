# Turkish FinBERT-Like BIST Sentiment Alerts

Turkish FinBERT-Like BIST Sentiment Alerts is a research-oriented pipeline for turning Turkish financial disclosures and news into daily BIST stock sentiment alerts.

The project collects KAP/news text, maps it to BIST tickers, scores each text as `negative`, `neutral`, or `positive`, aggregates the result into stock/event-level signals, and sends a compact Telegram report.

> This project is not investment advice. It is an NLP and signal research tool. Every output should be treated as a candidate event for manual review.

## What It Does

- Fetches Turkish financial disclosures/news, primarily from KAP-compatible flows.
- Cleans text and maps company names/tickers.
- Trains a lightweight baseline sentiment model from labeled Turkish financial examples.
- Scores new disclosures with class probabilities and sentiment score.
- Groups repeated or related news into event-level signals.
- Produces a daily answer to: "Is there anything meaningful today?"
- Sends a readable Telegram summary automatically.
- Can be triggered manually from Telegram with commands such as `/run` and `/run 2026-06-07`.

## Current Scope

The repository currently uses a practical baseline model:

- `TF-IDF + LogisticRegression` for the production-friendly baseline.
- Optional transformer/BERT training is scaffolded for later experiments.

The goal is not to predict prices directly. The goal is to produce a structured daily market/news briefing:

- no important event today,
- weak signal worth manual review,
- stock-specific positive/negative event,
- possible market-wide event,
- unusual signal compared with historical sentiment.

## Pipeline

```text
KAP/news fetch
  -> text cleaning
  -> ticker/company matching
  -> sentiment scoring
  -> daily stock aggregation
  -> event grouping and priority scoring
  -> Markdown/CSV/Telegram outputs
  -> GitHub Actions scheduled deploy
  -> optional Telegram command trigger via Cloudflare Worker
```

## Repository Layout

```text
data/
  labels/          labeled examples and labeling batches
  raw/             raw ticker lists, source lists, fetched raw files
  processed/       prepared/scored/historical CSV outputs

deploy/
  cloudflare-worker/
    telegram-workflow-dispatcher.js

docs/
  wiki/            detailed public documentation pages

reports/
  daily_alerts/    daily Markdown, Telegram HTML, signal CSV outputs
  figures/         plots and model diagnostics

src/turkish_fin_bert/
  fetch_news.py
  prepare_dataset.py
  train_model.py
  score_news.py
  daily_alerts.py
  daily_pipeline.py
```

## Quick Start

Install dependencies:

```powershell
uv sync
```

Run tests:

```powershell
uv run pytest
```

Train the baseline model:

```powershell
uv run train_model --input data/labels/labeled_news_master.csv --model-out models/master_baseline_sentiment.joblib --report-dir reports/master_baseline --split-strategy stratified
```

Run the daily pipeline locally:

```powershell
uv run daily_pipeline --kap-days 7 --update-history --baseline-lookback-days 365 --baseline-min-history 5
```

Generate alerts from already scored files:

```powershell
uv run daily_alerts --scored-news data/processed/kap_daily_scored_news.csv --daily-sentiment data/processed/kap_daily_sentiment.csv --baseline-daily-sentiment data/processed/kap_api_historical_daily_sentiment.csv --out-dir reports/daily_alerts
```

## Outputs

Daily runs produce files such as:

- `reports/daily_alerts/YYYY-MM-DD_daily_alerts.md`
- `reports/daily_alerts/YYYY-MM-DD_brief.md`
- `reports/daily_alerts/YYYY-MM-DD_telegram.html`
- `reports/daily_alerts/YYYY-MM-DD_event_signals.csv`
- `reports/daily_alerts/YYYY-MM-DD_signal_baseline.csv`
- `data/processed/kap_api_historical_daily_sentiment.csv`

Telegram messages use the HTML output so the report is readable on mobile.

## Automation

The repo includes a GitHub Actions workflow:

```text
.github/workflows/daily-pipeline.yml
```

It can:

- run on a schedule,
- be triggered manually from GitHub,
- train the baseline model on the runner,
- run the daily pipeline,
- commit updated historical outputs,
- send the Telegram report.

Telegram command triggering is handled by:

```text
deploy/cloudflare-worker/telegram-workflow-dispatcher.js
```

This Worker receives Telegram webhook updates and dispatches the GitHub workflow.

## Documentation

Detailed docs are split into wiki-style pages:

- [Wiki Home](docs/wiki/Home.md)
- [Project Purpose](docs/wiki/01-Project-Purpose.md)
- [Installation and Quick Start](docs/wiki/02-Installation-and-Quick-Start.md)
- [Data, Labeling, and Model](docs/wiki/03-Data-Labeling-and-Model.md)
- [Daily Signal Logic](docs/wiki/04-Daily-Signal-Logic.md)
- [GitHub Actions Deployment](docs/wiki/05-GitHub-Actions-Deployment.md)
- [Telegram Bot and Commands](docs/wiki/06-Telegram-Bot-and-Commands.md)
- [Operations and Troubleshooting](docs/wiki/07-Operations-and-Troubleshooting.md)
- [Public Repo Checklist](docs/wiki/08-Public-Repo-Checklist.md)

These files can also be copied into a GitHub Wiki if you prefer the repository Wiki UI.

## Safety Notes

- Do not commit Telegram tokens, GitHub tokens, or webhook secrets.
- Keep secrets in GitHub Actions Secrets or Cloudflare Worker Secrets.
- Historical CSV files are research artifacts; review size and content before making the repo public.
- The labels and baseline model are early-stage and should be improved before relying on outputs operationally.

## License

Add a license before publishing the repository publicly.
