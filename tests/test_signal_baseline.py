import pandas as pd

from turkish_fin_bert.signal_baseline import build_signal_baseline, select_notable_anomalies


def test_build_signal_baseline_uses_only_prior_days():
    daily = pd.DataFrame(
        [
            {"date": "2026-05-27", "ticker": "AAA", "sentiment_score": 0.04, "news_count": 1},
            {"date": "2026-05-28", "ticker": "AAA", "sentiment_score": 0.05, "news_count": 1},
            {"date": "2026-05-29", "ticker": "AAA", "sentiment_score": 0.06, "news_count": 1},
            {"date": "2026-05-30", "ticker": "AAA", "sentiment_score": 0.04, "news_count": 1},
            {"date": "2026-05-31", "ticker": "AAA", "sentiment_score": 0.05, "news_count": 1},
            {"date": "2026-06-01", "ticker": "AAA", "sentiment_score": 0.80, "news_count": 2},
            {"date": "2026-06-02", "ticker": "AAA", "sentiment_score": -0.80, "news_count": 2},
        ]
    )

    baseline = build_signal_baseline(daily, "2026-06-01", lookback_days=30, min_history=5)
    row = baseline.iloc[0]

    assert row["ticker"] == "AAA"
    assert row["history_days"] == 5
    assert row["baseline_mean"] < 0.06
    assert row["anomaly_level"] == "unusual"


def test_build_signal_baseline_marks_short_history():
    daily = pd.DataFrame(
        [
            {"date": "2026-05-31", "ticker": "AAA", "sentiment_score": 0.05, "news_count": 1},
            {"date": "2026-06-01", "ticker": "AAA", "sentiment_score": 0.70, "news_count": 2},
        ]
    )

    baseline = build_signal_baseline(daily, "2026-06-01", lookback_days=30, min_history=5)

    assert baseline.iloc[0]["history_days"] == 1
    assert baseline.iloc[0]["anomaly_level"] == "insufficient_history"


def test_select_notable_anomalies_returns_only_elevated_or_unusual_scores():
    baseline = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-06-01"),
                "ticker": "AAA",
                "sentiment_score": 0.70,
                "news_count": 1,
                "history_days": 10,
                "baseline_mean": 0.0,
                "baseline_std": 0.1,
                "baseline_abs_p90": 0.1,
                "delta_from_mean": 0.7,
                "anomaly_z": 7.0,
                "anomaly_level": "unusual",
            },
            {
                "date": pd.Timestamp("2026-06-01"),
                "ticker": "BBB",
                "sentiment_score": 0.05,
                "news_count": 1,
                "history_days": 10,
                "baseline_mean": 0.0,
                "baseline_std": 0.1,
                "baseline_abs_p90": 0.1,
                "delta_from_mean": 0.05,
                "anomaly_z": 0.5,
                "anomaly_level": "normal",
            },
        ]
    )

    notable = select_notable_anomalies(baseline, min_abs_score=0.20)

    assert notable["ticker"].tolist() == ["AAA"]
