# Daily Signal Logic

This page explains how the project turns scored news into daily alerts.

## Input Files

Daily alert generation uses:

```text
scored news CSV
daily sentiment CSV
optional historical daily sentiment CSV
```

Typical command:

```powershell
uv run daily_alerts --scored-news data/processed/kap_daily_scored_news.csv --daily-sentiment data/processed/kap_daily_sentiment.csv --baseline-daily-sentiment data/processed/kap_api_historical_daily_sentiment.csv --out-dir reports/daily_alerts
```

## Stock-Level Sentiment

For each ticker/day:

```text
sentiment_score = average(prob_positive - prob_negative)
```

If `sentiment_score >= min_abs_score`, the stock can appear in positive candidates.

If `sentiment_score <= -min_abs_score`, the stock can appear in negative candidates.

Default threshold:

```text
min_abs_score = 0.20
```

## Event-Level Grouping

The system does not only list best/worst stocks. It also groups related news into events.

This reduces noise from:

- repeated disclosures,
- same URL appearing for multiple tickers,
- multiple rows from one underlying event,
- routine announcements.

Event signals include fields such as:

```text
tickers
title
avg_impact_score
materiality_score
signal_strength
event_type
market_scope
priority_score
priority_reason
```

## Event Types

Common event categories:

```text
capital_action
contract_order
debt_credit
financial_result
legal_regulatory
management
market
routine
other
```

These are mapped into human-readable labels in reports.

## Signal Strength

Signals are described as:

```text
none
weak
medium
strong
```

Weak signals are not automatically important. They usually mean:

```text
manual review only, no strong conclusion
```

## Historical Baseline

The project compares current stock sentiment with previous days.

Important rule:

```text
The baseline never uses future dates relative to the report date.
```

This prevents look-ahead bias.

Possible historical labels:

```text
insufficient_history
normal
elevated
unusual
```

Report language:

- `gecmis veri az`,
- `gecmise gore olagan`,
- `gecmise gore dikkat cekici`,
- `gecmise gore sira disi`.

## Priority Score

`priority_score` is a 0-100 score combining:

- event materiality,
- absolute impact score,
- signal strength,
- historical anomaly,
- action label.

This score helps sort daily watchlist items.

## Action Labels

The report can assign actions such as:

```text
takip et
piyasa geneli dikkat
detay kontrol et
gecmis veri yetersiz
zayif sinyal
rutin / ignore
```

These are not trading actions. They are review labels.

## Daily Decision

The brief and Telegram report focus on:

```text
Karar
Sonuc
Durum
Akis
Izlenecekler
```

Example no-event language:

```text
Bugun aksiyon gerektiren olay yok.
Onemli olay yok; takip listesi bos.
```

Example weak-event language:

```text
Bugun dusuk oncelikli manuel kontrol sinyali var.
Zayif sinyaller var; manuel kontrol disinda aksiyon gerektirmiyor.
```

## Output Files

```text
YYYY-MM-DD_daily_alerts.md
YYYY-MM-DD_brief.md
YYYY-MM-DD_telegram.html
YYYY-MM-DD_event_signals.csv
YYYY-MM-DD_top_stocks.csv
YYYY-MM-DD_bottom_stocks.csv
YYYY-MM-DD_signal_baseline.csv
```

## How To Read Telegram Output

Telegram output is optimized for a quick mobile check:

```text
BIST Gunluk Ozet
Karar
Sonuc
Durum
Akis
Gecmis
Aksiyonlar
Izlenecekler
Pozitif / Negatif
```

If the watchlist is empty, that is a valid useful result.
