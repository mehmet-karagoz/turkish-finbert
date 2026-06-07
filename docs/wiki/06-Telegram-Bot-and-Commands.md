# Telegram Bot and Commands

This page explains Telegram reporting and command-based workflow triggering.

## Telegram Reporting

The pipeline creates a Telegram-specific file:

```text
reports/daily_alerts/YYYY-MM-DD_telegram.html
```

The GitHub Actions workflow sends this with:

```text
parse_mode = HTML
```

This gives a more readable mobile format than plain text.

## Required Telegram Secrets In GitHub Actions

```text
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

These are used by the GitHub Actions workflow to send the final report after the pipeline finishes.

## Getting Bot Token

1. Open Telegram.
2. Message `@BotFather`.
3. Run `/newbot`.
4. Follow the prompts.
5. Copy the token.

Do not commit this token.

## Getting Chat ID

Send any message to your bot, then open:

```text
https://api.telegram.org/botBOT_TOKEN/getUpdates
```

Find:

```json
"chat": {"id": 123456789}
```

Use that number as `TELEGRAM_CHAT_ID`.

## Command Trigger Architecture

Telegram cannot directly trigger GitHub Actions. The project uses a small Cloudflare Worker:

```text
Telegram message
  -> Telegram webhook
  -> Cloudflare Worker
  -> GitHub workflow_dispatch API
  -> GitHub Actions daily pipeline
  -> Telegram report
```

Worker file:

```text
deploy/cloudflare-worker/telegram-workflow-dispatcher.js
```

## Supported Commands

```text
/run
/run 2026-06-07
/help
```

Behavior:

- `/run`: triggers normal recent-data workflow.
- `/run 2026-06-07`: triggers workflow for a selected report date.
- `/help`: shows command help.

## Cloudflare Worker Variables And Secrets

Required:

```text
TELEGRAM_BOT_TOKEN
TELEGRAM_ALLOWED_CHAT_ID
TELEGRAM_WEBHOOK_SECRET
GITHUB_TOKEN
GITHUB_OWNER
GITHUB_REPO
GITHUB_REF
```

Optional:

```text
GITHUB_WORKFLOW_FILE = daily-pipeline.yml
DEFAULT_KAP_DAYS = 7
DEFAULT_BASELINE_LOOKBACK_DAYS = 365
DEFAULT_BASELINE_MIN_HISTORY = 5
```

## What Each Value Means

`TELEGRAM_BOT_TOKEN`

The token from BotFather.

`TELEGRAM_ALLOWED_CHAT_ID`

The only Telegram chat allowed to trigger the workflow.

`TELEGRAM_WEBHOOK_SECRET`

A random string that Telegram sends in the webhook header. The Worker rejects requests with the wrong secret.

Generate with PowerShell:

```powershell
[guid]::NewGuid().ToString("N")
```

`GITHUB_TOKEN`

A fine-grained GitHub token with:

```text
Actions: Read and write
Contents: Read-only
```

`GITHUB_OWNER`

The GitHub user or organization name.

`GITHUB_REPO`

The repository name.

`GITHUB_REF`

Usually:

```text
main
```

## Set Telegram Webhook

PowerShell:

```powershell
$env:BOT_TOKEN="BOT_TOKEN"
$env:WORKER_URL="https://WORKER_NAME.ACCOUNT.workers.dev"
$env:WEBHOOK_SECRET="THE_SAME_SECRET_IN_CLOUDFLARE"

curl.exe -X POST "https://api.telegram.org/bot$env:BOT_TOKEN/setWebhook" -d "url=$env:WORKER_URL" -d "secret_token=$env:WEBHOOK_SECRET"
```

Expected response:

```json
{"ok":true}
```

## Test

In Telegram:

```text
/help
```

Then:

```text
/run
```

Expected first message:

```text
Workflow tetiklendi.
```

After the GitHub workflow finishes, the bot should send the formatted daily report.

## Security Notes

- Only allow your own chat ID.
- Rotate `GITHUB_TOKEN` if leaked.
- Rotate `TELEGRAM_BOT_TOKEN` if leaked.
- Use a strong `TELEGRAM_WEBHOOK_SECRET`.
- Do not publish `.env` files or screenshots containing secrets.
