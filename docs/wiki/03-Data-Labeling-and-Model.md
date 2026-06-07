# Data, Labeling, and Model

This page explains how data moves through the project and how the baseline model is trained.

## Raw Data Schema

Raw news/disclosure files are expected to follow this schema:

```text
date
ticker
source
title
text
url
language
published_at
```

The most important fields are:

- `date`: report/disclosure date,
- `ticker`: BIST ticker without `.IS`,
- `title`: disclosure or news title,
- `text`: disclosure/news body,
- `source`: source name such as KAP/RSS,
- `url`: source URL.

## Fetching News

KAP API style fetch:

```powershell
uv run fetch_news --source kap-api --tickers-file data/raw/tickers_bist_kap.txt --out data/raw/kap_api_news.csv --kap-days 30 --limit 1000
```

RSS fetch:

```powershell
uv run fetch_news --source rss --rss-url-file data/raw/news_rss_sources.txt --tickers-file data/raw/tickers_bist_seed.txt --aliases data/raw/company_aliases.csv --out data/raw/news.csv --append --limit 100
```

## Ticker Matching

Ticker matching uses:

- explicit ticker lists,
- company aliases,
- KAP stock codes when available,
- text/title matching as fallback.

Important files:

```text
data/raw/tickers_bist_kap.txt
data/raw/tickers_bist_seed.txt
data/raw/company_aliases.csv
```

Coverage can be audited with:

```powershell
uv run audit_aliases --input data/raw/news_all.csv --out-unmatched reports/unmatched_news.csv --out-summary reports/alias_coverage.csv
```

## Preparing Data

For unlabeled data:

```powershell
uv run prepare_dataset --input data/raw/news.csv --output data/processed/news_prepared.csv
```

For labeled data:

```powershell
uv run prepare_dataset --input data/labels/labeled_news_master.csv --output data/processed/labeled_news_master.csv --labeled
```

Preparation does:

- date parsing,
- ticker normalization,
- title/text cleaning,
- title and body combination,
- short text filtering,
- duplicate removal,
- optional label validation.

## Label Definitions

Allowed labels:

```text
negative
neutral
positive
```

Labeling question:

```text
From an investor perspective, is this text negative, neutral, or positive for the company/stock expectation?
```

Guidance:

- `positive`: contract win, dividend, buyback, strong financial result, meaningful positive update.
- `negative`: lawsuit loss, debt stress, weak financial result, operational problem, regulatory risk.
- `neutral`: routine disclosure, administrative update, unclear or low-impact event.

## Creating Labeling Batches

Uncertain examples:

```powershell
uv run create_labeling_batch --input data/processed/real_model_scored_news.csv --output data/labels/next_labeling_batch.csv --exclude-labeled data/labels/labeled_news_master.csv --strategy uncertain --max-rows 300 --per-ticker 20 --include-model-hints
```

Negative mining:

```powershell
uv run mine_negative_examples --input data/raw/kap_api_negative_mining_2026_q2.csv --exclude-labeled data/labels/labeled_news_master.csv --output data/labels/negative_labeling_batch.csv --max-rows 100
```

Positive mining:

```powershell
uv run mine_positive_examples --input data/raw/kap_api_negative_mining_2026_q2.csv --exclude-labeled data/labels/labeled_news_master.csv --output data/labels/positive_labeling_batch.csv --max-rows 80 --per-reason 20
```

Merge labels:

```powershell
uv run merge_labels --input data/labels/negative_labeling_batch.csv --output data/labels/labeled_news_master.csv
```

## Baseline Model

The baseline model is:

```text
TfidfVectorizer(1,2 grams) + LogisticRegression(class_weight="balanced")
```

Train:

```powershell
uv run train_model --input data/labels/labeled_news_master.csv --model-out models/master_baseline_sentiment.joblib --report-dir reports/master_baseline --split-strategy stratified
```

Outputs:

```text
models/master_baseline_sentiment.joblib
reports/master_baseline/baseline_metrics.json
reports/master_baseline/baseline_test_predictions.csv
reports/figures/confusion_matrix.png
reports/figures/class_scores.png
```

## Scoring

Score prepared news:

```powershell
uv run score_news --model models/master_baseline_sentiment.joblib --input data/processed/news_prepared.csv --out data/processed/scored_news.csv --daily-out data/processed/daily_sentiment.csv
```

Main formula:

```text
sentiment_score = prob_positive - prob_negative
```

Daily aggregation groups by:

```text
date, ticker
```

and computes:

- average sentiment score,
- average class probabilities,
- news count,
- 3/7/14-day rolling sentiment.

## Model Improvement Path

Recommended next improvements:

1. Increase labeled examples per class.
2. Review false positives/false negatives.
3. Improve ticker/company alias coverage.
4. Add event-type labels if needed.
5. Fine-tune a Turkish BERT model only after labels are strong enough.
