# Turkish FinBERT-Like BIST Sentiment Alerts

<p align="center">
  <b>Turkish financial sentiment analysis and daily BIST stock alert pipeline.</b>
</p>

<p align="center">
  <a href="https://github.com/mehmet-karagoz/turkish-finbert/actions/workflows/daily-pipeline.yml">
    <img src="https://github.com/mehmet-karagoz/turkish-finbert/actions/workflows/daily-pipeline.yml/badge.svg" alt="Daily Pipeline">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/NLP-Turkish%20Finance-green" alt="Turkish Finance NLP">
  <img src="https://img.shields.io/badge/Market-BIST-red" alt="BIST">
  <img src="https://img.shields.io/badge/Status-Research%20Project-orange" alt="Research Project">
</p>

---

## Overview

**Turkish FinBERT-Like BIST Sentiment Alerts** is a research-oriented NLP pipeline that converts Turkish financial disclosures and news into daily stock-level sentiment signals for BIST companies.

The project collects Turkish financial text, maps it to BIST tickers, scores each item as `negative`, `neutral`, or `positive`, aggregates the results into event-level signals, and optionally sends a compact Telegram report.

> [!IMPORTANT]
> This project is **not investment advice**. It is an NLP and signal research tool. Every output should be treated as a candidate event for manual review.

---

## Why This Project Exists

Turkish financial markets produce many daily disclosures and news items. Reading all of them manually is time-consuming.

This project aims to answer one practical question:

> **“Is there anything meaningful for BIST stocks today?”**

It does not try to predict prices directly. Instead, it helps detect:

- stock-specific positive or negative events,
- unusually strong sentiment compared with historical sentiment,
- market-wide news intensity,
- repeated or related disclosures,
- daily items that deserve manual review.

---

## Features

| Area | Description |
| --- | --- |
| Turkish financial NLP | Cleans and processes Turkish KAP/news-style text |
| Ticker matching | Maps company names and aliases to BIST tickers |
| Sentiment scoring | Scores each item as `negative`, `neutral`, or `positive` |
| Baseline model | Uses a production-friendly `TF-IDF + LogisticRegression` baseline |
| Transformer experiments | Includes optional transformer/BERT training scaffold |
| Event aggregation | Groups daily news into stock/event-level signals |
| Historical comparison | Compares current sentiment with historical baselines |
| Telegram reports | Sends compact mobile-friendly summaries |
| GitHub Actions automation | Runs the daily pipeline on schedule or manually |
| Cloudflare Worker trigger | Allows Telegram command-based workflow dispatch |

---

## Pipeline

```text
KAP/news fetch
  -> text cleaning
  -> ticker/company matching
  -> sentiment scoring
  -> daily stock aggregation
  -> event grouping and priority scoring
  -> Markdown / CSV / Telegram outputs
  -> GitHub Actions scheduled run
  -> optional Telegram command trigger
```

---

## Project Status

This repository currently focuses on a practical baseline system.

| Component | Status |
| --- | --- |
| Data preparation | Available |
| Label management | Available |
| Baseline sentiment model | Available |
| Daily scoring pipeline | Available |
| Daily alert generation | Available |
| Telegram reporting | Available |
| GitHub Actions automation | Available |
| Cloudflare Worker Telegram trigger | Available |
| Transformer/BERT training | Experimental scaffold |
| Price prediction | Not the goal |

---

## Repository Layout

```text
data/
  labels/          labeled examples and labeling batches
  raw/             raw ticker lists, source lists, fetched raw files
  processed/       prepared, scored, and historical CSV outputs

deploy/
  cloudflare-worker/
    telegram-workflow-dispatcher.js

docs/
  wiki/            detailed documentation pages

models/
  *.joblib         trained baseline models

reports/
  daily_alerts/    daily Markdown, Telegram HTML, and signal CSV outputs
  figures/         plots and model diagnostics

src/turkish_fin_bert/
  fetch_news.py
  prepare_dataset.py
  train_model.py
  train_transformer.py
  score_news.py
  daily_alerts.py
  daily_pipeline.py
  backtest.py
```

---

## Installation

### Requirements

- Python `3.10+`
- `uv`
- Git
- Optional: Telegram bot token and chat ID for notifications

Clone the repository:

```bash
git clone https://github.com/mehmet-karagoz/turkish-finbert.git
cd turkish-finbert
```

Install dependencies:

```bash
uv sync
```

Install optional NLP dependencies for transformer experiments:

```bash
uv sync --extra nlp
```

Install development dependencies:

```bash
uv sync --extra dev
```

---

## Quick Start

### 1. Run tests

```bash
uv run pytest
```

### 2. Train the baseline sentiment model

```bash
uv run train_model \
  --input data/labels/labeled_news_master.csv \
  --model-out models/master_baseline_sentiment.joblib \
  --report-dir reports/master_baseline \
  --split-strategy stratified
```

### 3. Run the daily pipeline locally

```bash
uv run daily_pipeline \
  --kap-days 7 \
  --update-history \
  --baseline-lookback-days 365 \
  --baseline-min-history 5
```

### 4. Generate alerts from existing scored files

```bash
uv run daily_alerts \
  --scored-news data/processed/kap_daily_scored_news.csv \
  --daily-sentiment data/processed/kap_daily_sentiment.csv \
  --baseline-daily-sentiment data/processed/kap_api_historical_daily_sentiment.csv \
  --out-dir reports/daily_alerts
```

---

## Example Output

Daily runs generate files similar to:

```text
reports/daily_alerts/YYYY-MM-DD_daily_alerts.md
reports/daily_alerts/YYYY-MM-DD_brief.md
reports/daily_alerts/YYYY-MM-DD_telegram.html
reports/daily_alerts/YYYY-MM-DD_event_signals.csv
reports/daily_alerts/YYYY-MM-DD_signal_baseline.csv
data/processed/kap_api_historical_daily_sentiment.csv
```

A simplified daily signal may look like:

```text
BIST Daily Sentiment Brief

Market status:
- No major market-wide negative signal detected.

Stock-level signals:
- THYAO: positive event signal, above historical baseline.
- ASELS: neutral news intensity, manual review optional.
- KCHOL: repeated disclosure group detected.

Note:
Signals are generated from NLP sentiment and event grouping. They are not buy/sell recommendations.
```

---

## CLI Commands

The project exposes several command-line entry points:

| Command | Purpose |
| --- | --- |
| `fetch_news` | Fetch Turkish financial news/disclosures |
| `audit_aliases` | Review ticker/company alias mapping |
| `create_labeling_batch` | Create data batches for manual labeling |
| `auto_label_batch` | Apply automatic labeling rules |
| `mine_negative_examples` | Mine candidate negative examples |
| `mine_positive_examples` | Mine candidate positive examples |
| `merge_labels` | Merge labeled datasets |
| `prepare_dataset` | Prepare dataset for model training |
| `train_model` | Train baseline sentiment model |
| `train_transformer` | Run transformer/BERT experiments |
| `score_news` | Score new text items |
| `daily_alerts` | Generate daily signal reports |
| `daily_pipeline` | Run end-to-end daily workflow |
| `analyze_financial_effect` | Analyze financial effect signals |
| `backtest` | Run simple signal/backtest analysis |

---

## Automation

The repository includes a GitHub Actions workflow:

```text
.github/workflows/daily-pipeline.yml
```

It can:

- run automatically on schedule,
- be triggered manually from GitHub,
- train the baseline model,
- run the daily sentiment pipeline,
- upload daily output artifacts,
- commit updated historical outputs,
- send the Telegram report if secrets are configured.

Required Telegram secrets:

```text
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

---

## Telegram Command Trigger

Telegram command triggering is handled by:

```text
deploy/cloudflare-worker/telegram-workflow-dispatcher.js
```

The Worker receives Telegram webhook updates and dispatches the GitHub Actions workflow.

Supported command style:

```text
/run
/run 2026-06-07
```

---

## Model Approach

The current baseline is intentionally simple and inspectable:

```text
TF-IDF vectorization
  -> LogisticRegression classifier
  -> class probabilities
  -> sentiment score
  -> stock-level aggregation
```

This makes the first production-like version:

- lightweight,
- fast to train,
- easy to debug,
- suitable for GitHub Actions runners,
- practical for daily automation.

Transformer-based training is included for future experiments, but the baseline model is the default practical path.

---

## Data and Labeling

The project uses labeled Turkish financial examples to train the sentiment model.

Typical label classes:

| Label | Meaning |
| --- | --- |
| `negative` | Potentially adverse financial/company event |
| `neutral` | Informational or low-signal disclosure |
| `positive` | Potentially favorable financial/company event |

The labeling workflow supports:

- creating labeling batches,
- mining positive/negative candidates,
- merging labeled files,
- preparing datasets for training.

---

## Documentation

Detailed documentation is available under `docs/wiki`:

- [Wiki Home](docs/wiki/Home.md)
- [Project Purpose](docs/wiki/01-Project-Purpose.md)
- [Installation and Quick Start](docs/wiki/02-Installation-and-Quick-Start.md)
- [Data, Labeling, and Model](docs/wiki/03-Data-Labeling-and-Model.md)
- [Daily Signal Logic](docs/wiki/04-Daily-Signal-Logic.md)
- [GitHub Actions Deployment](docs/wiki/05-GitHub-Actions-Deployment.md)
- [Telegram Bot and Commands](docs/wiki/06-Telegram-Bot-and-Commands.md)
- [Operations and Troubleshooting](docs/wiki/07-Operations-and-Troubleshooting.md)

---

## Limitations

This project has several important limitations:

- Sentiment is estimated from text, not from complete financial analysis.
- The model can misclassify ambiguous or complex disclosures.
- Company/ticker matching may fail for uncommon aliases.
- News intensity does not necessarily imply price movement.
- Historical sentiment comparison is a research signal, not a trading rule.
- Outputs require manual review before any real-world decision.

---

## Contributing

Contributions are welcome, especially for:

- Turkish financial text examples,
- better ticker/company alias dictionaries,
- labeling improvements,
- model evaluation,
- transformer experiments,
- documentation,
- automation and deployment improvements.

Suggested workflow:

```bash
git checkout -b feature/your-feature-name
uv sync --extra dev
uv run pytest
```

Then open a pull request with a clear description of the change.

---

## Disclaimer

This repository is for research and educational purposes only.

It does **not** provide investment advice, trading recommendations, portfolio management, or financial consulting. The generated alerts are NLP-based research outputs and should not be used as the sole basis for investment decisions.

---

## Author

Developed by [Mehmet Karagöz](https://github.com/mehmet-karagoz).
