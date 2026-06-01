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
    assert result["market_news"]["title"].tolist() == ["BIST genelinde sert yukselis"]
    assert (out_dir / "2026-06-01_daily_alerts.md").exists()
    assert "En Iyi Hisseler" in (out_dir / "2026-06-01_daily_alerts.md").read_text(encoding="utf-8")
