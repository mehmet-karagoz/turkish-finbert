# Public Repo Checklist

Use this checklist before making the repository public.

## Secrets

Confirm none of these are committed:

```text
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
TELEGRAM_WEBHOOK_SECRET
GITHUB_TOKEN
github_pat_...
.env
wrangler.toml with secrets
```

Search locally:

```powershell
rg -n "github_pat|TELEGRAM_BOT_TOKEN|TELEGRAM_CHAT_ID|GITHUB_TOKEN|WEBHOOK_SECRET|BOT_TOKEN" -S . --glob "!uv.lock"
```

## Data Review

Review:

```text
data/labels/
data/raw/
data/processed/
reports/daily_alerts/
```

Questions:

- Is every file safe to publish?
- Are there any private notes?
- Are there any paid-provider data exports?
- Are there any credentials in CSV/text files?
- Are generated historical files too large?

## Model Artifacts

Current `.gitignore` excludes:

```text
models/*.joblib
models/*/
```

This is good for public release unless you intentionally want to publish model binaries.

## License

Add a license before public release.

Common options:

- MIT: permissive, simple.
- Apache-2.0: permissive with patent language.
- GPL: copyleft.

Choose intentionally.

## Disclaimer

The README already says the project is not investment advice. Keep that visible.

Recommended wording:

```text
This project is for research and monitoring. It is not investment advice and should not be used as the sole basis for trading decisions.
```

## GitHub Actions

Check:

- workflow permissions are clear in docs,
- secrets are documented but not committed,
- scheduled jobs are not too frequent,
- committed historical outputs are expected.

## Cloudflare Worker

Check:

- Worker source is committed,
- secrets are only in Cloudflare,
- `TELEGRAM_ALLOWED_CHAT_ID` is set,
- webhook secret is strong,
- no real token appears in docs.

## README Quality

Main README should answer:

- What is this?
- Why does it exist?
- What does it output?
- How do I run it?
- Where is the detailed documentation?
- What are the limitations?

## Final Local Checks

```powershell
uv run pytest
git diff --check
rg -n "github_pat|TELEGRAM_BOT_TOKEN=|GITHUB_TOKEN=|BOT_TOKEN=real" -S . --glob "!uv.lock"
```

## Suggested Release Steps

1. Add license.
2. Review public data files.
3. Run tests.
4. Push docs.
5. Make repository public.
6. Run GitHub Actions manually once.
7. Test Telegram `/run`.
