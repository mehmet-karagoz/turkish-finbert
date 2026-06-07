# Operations and Troubleshooting

This page lists practical checks for running the project.

## Daily Operating Flow

Normal operation:

```text
1. GitHub Actions runs on schedule.
2. Model trains in the runner.
3. KAP/news data is fetched.
4. Daily alerts are generated.
5. Telegram report is sent.
6. Historical outputs are committed back.
```

Manual operation:

```text
Telegram /run
Telegram /run YYYY-MM-DD
```

or:

```text
GitHub Actions -> Run workflow
```

## Health Checks

Check GitHub Actions:

- run is green,
- `Train deploy model` passed,
- `Run daily pipeline` passed,
- `Send Telegram brief` passed,
- commit step did not fail.

Check Telegram:

- command acknowledgement arrived,
- final report arrived,
- report is formatted with bold titles/code blocks.

Check repo files:

```text
reports/daily_alerts/
data/processed/kap_api_historical_daily_sentiment.csv
```

## Local Verification Commands

```powershell
uv run pytest
python -m py_compile src/turkish_fin_bert/daily_alerts.py
Get-Content reports/daily_alerts/2026-06-06_telegram.html
```

Worker syntax check:

```powershell
Get-Content -Raw deploy/cloudflare-worker/telegram-workflow-dispatcher.js | node --input-type=module --check
```

## Common Problems

### GitHub workflow does not start from Telegram

Check:

- Cloudflare Worker deployed successfully.
- Telegram webhook points to Worker URL.
- `TELEGRAM_WEBHOOK_SECRET` matches between Telegram webhook and Cloudflare.
- `GITHUB_TOKEN` has `Actions: Read and write`.
- `GITHUB_OWNER`, `GITHUB_REPO`, and `GITHUB_REF` are correct.

### Worker says unauthorized

Check:

- `TELEGRAM_ALLOWED_CHAT_ID` includes the chat ID that sent the command.
- Private chat and group IDs are different.
- Group/supergroup IDs usually start with `-100`.
- Multiple allowed IDs are comma-separated, for example `123456789,-1001234567890`.

### Telegram final report does not arrive

Check GitHub Actions secrets:

```text
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

Check the `Send Telegram brief` step logs.

### Workflow runs but no KAP data is found

Possible causes:

- no disclosure for selected date,
- ticker list too narrow,
- KAP response format changed,
- selected date is a weekend/holiday,
- selected date has data but not for tracked tickers.

Try:

```powershell
uv run daily_pipeline --kap-days 30 --update-history
```

or a wider ticker file.

### Historical comparison says data is insufficient

This means the ticker does not have enough prior daily sentiment rows.

Fix:

- collect more historical KAP/news data,
- run with a longer lookback,
- lower `baseline_min_history` only for experimentation.

### Model output looks wrong

Check:

- label distribution,
- recent examples in `baseline_test_predictions.csv`,
- whether the text is clean,
- whether company aliases mapped correctly,
- whether the event is actually neutral/routine.

## Maintenance Tasks

Recommended weekly:

- review Telegram alerts manually,
- inspect false positives and false negatives,
- add corrected labels,
- retrain baseline model,
- verify GitHub Actions success rate.

Recommended monthly:

- review historical CSV size,
- rotate long-lived tokens if needed,
- update ticker/alias lists,
- check class balance in labels.

## Public Repo Maintenance

Before making the repo public:

- remove secrets,
- review CSV files for private notes,
- add a license,
- add a clear disclaimer,
- decide whether historical data files should stay committed.
