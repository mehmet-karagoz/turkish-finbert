import pandas as pd

from turkish_fin_bert.daily_alerts import generate_daily_alerts


def test_generate_daily_alerts_writes_rankings_and_market_news(tmp_path):
    scored = tmp_path / "scored.csv"
    daily = tmp_path / "daily.csv"
    out_dir = tmp_path / "alerts"

    pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "ticker": "AAA",
                "title": "AAA yeni sozlesme imzaladi",
                "text": "Sirket yeni is iliskisi acikladi.",
                "url": "u1",
                "prediction": "positive",
                "prob_negative": 0.05,
                "prob_neutral": 0.10,
                "prob_positive": 0.85,
                "sentiment_score": 0.80,
            },
            {
                "date": "2026-06-01",
                "ticker": "BBB",
                "title": "BBB dava kaybetti",
                "text": "Sirket aleyhine onemli karar aciklandi.",
                "url": "u2",
                "prediction": "negative",
                "prob_negative": 0.90,
                "prob_neutral": 0.05,
                "prob_positive": 0.05,
                "sentiment_score": -0.85,
            },
            {
                "date": "2026-06-01",
                "ticker": "",
                "title": "BIST genelinde sert yukselis",
                "text": "Borsa Istanbul ve piyasa geneli icin guclu haber.",
                "url": "u3",
                "prediction": "positive",
                "prob_negative": 0.10,
                "prob_neutral": 0.10,
                "prob_positive": 0.80,
                "sentiment_score": 0.70,
            },
        ]
    ).to_csv(scored, index=False)

    pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "ticker": "AAA",
                "sentiment_score": 0.80,
                "prob_negative": 0.05,
                "prob_neutral": 0.10,
                "prob_positive": 0.85,
                "news_count": 1,
            },
            {
                "date": "2026-06-01",
                "ticker": "BBB",
                "sentiment_score": -0.85,
                "prob_negative": 0.90,
                "prob_neutral": 0.05,
                "prob_positive": 0.05,
                "news_count": 1,
            },
        ]
    ).to_csv(daily, index=False)

    result = generate_daily_alerts(scored, daily, out_dir, date="2026-06-01", top_n=1)

    assert result["top_stocks"]["ticker"].tolist() == ["AAA"]
    assert result["bottom_stocks"]["ticker"].tolist() == ["BBB"]
    assert (result["top_stocks"]["sentiment_score"] > 0).all()
    assert (result["bottom_stocks"]["sentiment_score"] < 0).all()
    assert result["positive_news"]["ticker"].tolist() == ["AAA"]
    assert result["negative_news"]["ticker"].tolist() == ["BBB"]
    assert result["market_news"]["title"].tolist() == ["BIST genelinde sert yukselis"]
    assert not result["event_signals"].empty
    assert (out_dir / "2026-06-01_daily_alerts.md").exists()
    assert (out_dir / "2026-06-01_event_signals.csv").exists()
    report_text = (out_dir / "2026-06-01_daily_alerts.md").read_text(encoding="utf-8")
    assert "Onemli Olay Ozeti" in report_text
    assert "En Iyi Hisseler" in report_text


def test_generate_daily_alerts_filters_weak_signals_and_reports_no_signal(tmp_path):
    scored = tmp_path / "scored.csv"
    daily = tmp_path / "daily.csv"
    out_dir = tmp_path / "alerts"

    pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "ticker": "AAA",
                "title": "AAA hafif pozitif haber",
                "text": "Sirket icin zayif pozitif haber.",
                "url": "u1",
                "prediction": "positive",
                "prob_negative": 0.40,
                "prob_neutral": 0.10,
                "prob_positive": 0.50,
                "sentiment_score": 0.10,
            },
            {
                "date": "2026-06-01",
                "ticker": "BBB",
                "title": "BBB hafif negatif haber",
                "text": "Sirket icin zayif negatif haber.",
                "url": "u2",
                "prediction": "negative",
                "prob_negative": 0.52,
                "prob_neutral": 0.10,
                "prob_positive": 0.48,
                "sentiment_score": -0.04,
            },
        ]
    ).to_csv(scored, index=False)
    pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "ticker": "AAA",
                "sentiment_score": 0.10,
                "prob_negative": 0.40,
                "prob_neutral": 0.10,
                "prob_positive": 0.50,
                "news_count": 1,
            },
            {
                "date": "2026-06-01",
                "ticker": "BBB",
                "sentiment_score": -0.04,
                "prob_negative": 0.52,
                "prob_neutral": 0.10,
                "prob_positive": 0.48,
                "news_count": 1,
            },
        ]
    ).to_csv(daily, index=False)

    result = generate_daily_alerts(scored, daily, out_dir, date="2026-06-01", top_n=10, min_abs_score=0.20)
    text = (out_dir / "2026-06-01_daily_alerts.md").read_text(encoding="utf-8")

    assert result["top_stocks"].empty
    assert result["bottom_stocks"].empty
    assert result["positive_news"].empty
    assert result["negative_news"].empty
    assert result["event_signals"].empty
    assert "Bugun esigi asan anlamli haber/hisse sinyali yok." in text
    assert "Sinyal esigi: |skor| >= 0.20" in text


def test_generate_daily_alerts_groups_same_url_as_one_event(tmp_path):
    scored = tmp_path / "scored.csv"
    daily = tmp_path / "daily.csv"
    out_dir = tmp_path / "alerts"

    pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "ticker": "AAA",
                "title": "AAA ve BBB yeni sozlesme imzaladi",
                "text": "Iki sirket onemli is iliskisi acikladi.",
                "url": "same-url",
                "prediction": "positive",
                "prob_negative": 0.05,
                "prob_neutral": 0.10,
                "prob_positive": 0.85,
                "sentiment_score": 0.80,
            },
            {
                "date": "2026-06-01",
                "ticker": "BBB",
                "title": "AAA ve BBB yeni sozlesme imzaladi",
                "text": "Iki sirket onemli is iliskisi acikladi.",
                "url": "same-url",
                "prediction": "positive",
                "prob_negative": 0.10,
                "prob_neutral": 0.10,
                "prob_positive": 0.80,
                "sentiment_score": 0.70,
            },
        ]
    ).to_csv(scored, index=False)
    pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "ticker": "AAA",
                "sentiment_score": 0.80,
                "prob_negative": 0.05,
                "prob_neutral": 0.10,
                "prob_positive": 0.85,
                "news_count": 1,
            },
            {
                "date": "2026-06-01",
                "ticker": "BBB",
                "sentiment_score": 0.70,
                "prob_negative": 0.10,
                "prob_neutral": 0.10,
                "prob_positive": 0.80,
                "news_count": 1,
            },
        ]
    ).to_csv(daily, index=False)

    result = generate_daily_alerts(scored, daily, out_dir, date="2026-06-01", top_n=10)
    events = result["event_signals"]

    assert len(events) == 1
    assert events.iloc[0]["tickers"] == "AAA,BBB"
    assert events.iloc[0]["news_count"] == 2
    assert events.iloc[0]["market_scope"] == "multi_stock"
    assert events.iloc[0]["signal_strength"] in {"medium", "strong"}
