# GitHub Actions Deployment

This page explains how to run the project without keeping a local computer open.

## Why GitHub Actions

This project does not need a 24/7 web server for the daily report. It needs a scheduled job.

GitHub Actions is suitable because it can:

- run on a schedule,
- run on manual trigger,
- train the baseline model on each run,
- fetch and score new data,
- send Telegram output,
- commit updated historical files.

## Workflow File

```text
.github/workflows/daily-pipeline.yml
```

Main triggers:

```text
schedule
workflow_dispatch
```

Scheduled times are defined in UTC. The current workflow comments map them to Turkey time.

## Workflow Inputs

Manual workflow inputs:

```text
kap_days
baseline_lookback_days
baseline_min_history
report_date
telegram_chat_id
```

If `report_date` is empty, the workflow runs the normal recent-data flow.

If `report_date` is set, the workflow passes:

```text
--kap-from-date REPORT_DATE
--kap-to-date REPORT_DATE
--date REPORT_DATE
```

If `telegram_chat_id` is set, the final Telegram report is sent to that chat. If it is empty, the workflow uses the repository secret:

```text
TELEGRAM_CHAT_ID
```

## Required GitHub Settings

Repository settings:

```text
Settings -> Actions -> General -> Workflow permissions
```

Select:

```text
Read and write permissions
```

This is needed because the workflow commits updated historical CSV/report files.

## Required GitHub Secrets

For Telegram sending:

```text
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

Add them under:

```text
Settings -> Secrets and variables -> Actions -> New repository secret
```

## Why The Model Is Trained In CI

The repository ignores model binaries:

```text
models/*.joblib
models/*/
```

This avoids committing binary model files. The workflow trains the baseline model from:

```text
data/labels/labeled_news_master.csv
```

and writes:

```text
models/master_baseline_sentiment.joblib
```

inside the runner.

## Historical Files

GitHub Actions runners are temporary. Files disappear after a run unless they are:

- uploaded as artifacts,
- committed back to the repo,
- stored externally.

This project commits selected historical outputs back to the repo:

```text
data/raw/kap_api_historical_news.csv
data/processed/kap_api_historical_prepared.csv
data/processed/kap_api_historical_scored_news.csv
data/processed/kap_api_historical_daily_sentiment.csv
reports/daily_alerts/
```

The workflow uses `git add -f` because some output paths are ignored by `.gitignore`.

## Manual Run From GitHub

Go to:

```text
Actions -> Daily BIST Sentiment Pipeline -> Run workflow
```

For today's normal flow:

```text
report_date = empty
kap_days = 7
```

For a specific date:

```text
report_date = 2026-06-07
```

## Expected Result

After a successful run:

- Telegram receives the HTML report.
- `reports/daily_alerts/` contains new files.
- historical CSV files are updated.
- an artifact is attached to the workflow run.

## Failure Checklist

If the workflow fails:

1. Check whether `data/labels/labeled_news_master.csv` exists.
2. Check whether GitHub workflow permissions are `Read and write`.
3. Check whether Telegram secrets are present.
4. Check whether KAP returned data for the selected date.
5. Check workflow logs for the exact command that failed.
