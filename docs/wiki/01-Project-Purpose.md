# Project Purpose

This project builds a Turkish financial sentiment pipeline for BIST stocks. It focuses on official disclosures and finance-related text, then converts those texts into structured daily alerts.

## Problem

Investors and researchers face a large amount of daily text:

- KAP disclosures,
- company announcements,
- news titles and summaries,
- financial event descriptions,
- repeated copies of the same disclosure.

Reading every item manually is slow. A simple positive/negative text classifier is also not enough, because the useful question is usually:

```text
Is this event important enough to check today?
```

## Project Goal

The goal is to produce a daily research briefing that says:

- there is no important event today,
- there are weak signals but no urgent action,
- one or more stocks deserve manual review,
- a market-wide event may be present,
- a signal is unusual compared with recent history.

The project is built as a pipeline, not as a one-off notebook.

## What The System Produces

For each daily run, the system produces:

- a detailed Markdown report,
- a short Markdown brief,
- a Telegram HTML message,
- stock-level ranking CSVs,
- event-level signal CSVs,
- historical sentiment CSVs for baseline comparison.

## What The System Does Not Do

The system does not:

- give investment advice,
- guarantee price movement prediction,
- replace manual review,
- execute trades,
- optimize a portfolio by default,
- claim that sentiment alone is enough for trading decisions.

## Current Model Strategy

The project name says FinBERT-like because the long-term direction is a Turkish financial language model. The current practical baseline is:

```text
TF-IDF + LogisticRegression
```

This is intentional. It keeps the first working version:

- fast,
- cheap,
- easy to run in GitHub Actions,
- easy to debug,
- suitable for active labeling and iteration.

Transformer fine-tuning can be added later after the labeled data set is stronger.

## Typical Use Cases

- Daily KAP/news monitoring.
- Researching sentiment-event relationships.
- Building a labeled Turkish financial text data set.
- Comparing current sentiment against historical behavior.
- Sending compact Telegram alerts to a private channel.
- Triggering reports manually for a specific date.

## Interpretation Rule

The output should be read as:

```text
"This event may be worth checking."
```

not as:

```text
"Buy or sell this stock."
```
