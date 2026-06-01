import pandas as pd

from turkish_fin_bert.backtest import _rebalance_dates, run_backtest


def test_rebalance_dates_supports_daily_mode():
    dates = pd.Series(pd.to_datetime(["2026-05-01", "2026-05-02", "2026-05-03"]))

    out = _rebalance_dates(dates, months=0)

    assert len(out) == 3


def test_run_backtest_ignores_sentiment_without_price_data(tmp_path):
    sentiment_csv = tmp_path / "sentiment.csv"
    prices_csv = tmp_path / "prices.csv"
    out_csv = tmp_path / "backtest.csv"
    pd.DataFrame(
        [
            {"date": "2026-01-01", "ticker": "AAA", "sentiment_score": -0.5},
            {"date": "2026-01-01", "ticker": "MISSING", "sentiment_score": 1.0},
            {"date": "2026-01-02", "ticker": "AAA", "sentiment_score": 0.2},
        ]
    ).to_csv(sentiment_csv, index=False)
    pd.DataFrame(
        [
            {"date": "2026-01-01", "ticker": "AAA", "open": 10, "high": 10, "low": 10, "close": 10, "volume": 100},
            {"date": "2026-01-02", "ticker": "AAA", "open": 11, "high": 11, "low": 11, "close": 11, "volume": 100},
            {"date": "2026-01-03", "ticker": "AAA", "open": 12, "high": 12, "low": 12, "close": 12, "volume": 100},
        ]
    ).to_csv(prices_csv, index=False)

    result = run_backtest(sentiment_csv, prices_csv, top_n=1, rebalance_months=0, out_csv=out_csv)

    assert "MISSING" not in ",".join(result["holdings"].fillna("").astype(str))
    assert out_csv.exists()
