import pandas as pd

from turkish_fin_bert import daily_pipeline


def test_run_daily_pipeline_creates_outputs(monkeypatch, tmp_path):
    tickers_file = tmp_path / "tickers.txt"
    tickers_file.write_text("AAA\nBBB\n", encoding="utf-8")
    raw_out = tmp_path / "raw.csv"
    prepared_out = tmp_path / "prepared.csv"
    scored_out = tmp_path / "scored.csv"
    daily_out = tmp_path / "daily.csv"
    alerts_dir = tmp_path / "alerts"

    def fake_fetch_kap_api(args):
        assert args.kap_days == 7
        return [
            {
                "publishDate": "01.06.2026 10:00:00",
                "kapTitle": "AAA A.S.",
                "subject": "Yeni Is Iliskisi",
                "summary": "Sirket yeni sozlesme imzaladi.",
                "stockCodes": "AAA",
                "disclosureIndex": 1,
            },
            {
                "publishDate": "01.06.2026 11:00:00",
                "kapTitle": "BBB A.S.",
                "subject": "Ozel Durum Aciklamasi",
                "summary": "Sirket aleyhine onemli gelisme acikladi.",
                "stockCodes": "BBB",
                "disclosureIndex": 2,
            },
        ]

    def fake_score_news(model_path, input_csv, out_csv, daily_out_csv):
        prepared = pd.read_csv(input_csv)
        scored = prepared.copy()
        scored["prob_negative"] = [0.05, 0.90]
        scored["prob_neutral"] = [0.10, 0.05]
        scored["prob_positive"] = [0.85, 0.05]
        scored["prediction"] = ["positive", "negative"]
        scored["sentiment_score"] = scored["prob_positive"] - scored["prob_negative"]
        scored.to_csv(out_csv, index=False)
        daily = (
            scored.groupby(["date", "ticker"], as_index=False)
            .agg(
                sentiment_score=("sentiment_score", "mean"),
                prob_negative=("prob_negative", "mean"),
                prob_neutral=("prob_neutral", "mean"),
                prob_positive=("prob_positive", "mean"),
                news_count=("text", "size"),
            )
            .sort_values(["ticker", "date"])
        )
        daily.to_csv(daily_out_csv, index=False)
        return scored, daily

    monkeypatch.setattr(daily_pipeline, "fetch_kap_api", fake_fetch_kap_api)
    monkeypatch.setattr(daily_pipeline, "score_news", fake_score_news)

    result = daily_pipeline.run_daily_pipeline(
        model_path=tmp_path / "model.joblib",
        tickers_file=tickers_file,
        raw_out=raw_out,
        prepared_out=prepared_out,
        scored_out=scored_out,
        daily_out=daily_out,
        alerts_dir=alerts_dir,
        kap_days=7,
        top_n=1,
        min_abs_score=0.20,
    )

    assert result["raw_rows"] == 2
    assert result["prepared_rows"] == 2
    assert result["scored_rows"] == 2
    assert result["daily_rows"] == 2
    assert raw_out.exists()
    assert prepared_out.exists()
    assert scored_out.exists()
    assert daily_out.exists()
    assert (alerts_dir / "2026-06-01_daily_alerts.md").exists()


def test_run_daily_pipeline_updates_history_without_duplicates(monkeypatch, tmp_path):
    tickers_file = tmp_path / "tickers.txt"
    tickers_file.write_text("AAA\nBBB\n", encoding="utf-8")
    raw_out = tmp_path / "raw.csv"
    prepared_out = tmp_path / "prepared.csv"
    scored_out = tmp_path / "scored.csv"
    daily_out = tmp_path / "daily.csv"
    historical_raw = tmp_path / "historical_raw.csv"
    historical_prepared = tmp_path / "historical_prepared.csv"
    historical_scored = tmp_path / "historical_scored.csv"
    historical_daily = tmp_path / "historical_daily.csv"
    alerts_dir = tmp_path / "alerts"

    def fake_fetch_kap_api(args):
        return [
            {
                "publishDate": "01.06.2026 10:00:00",
                "kapTitle": "AAA A.S.",
                "subject": "Yeni Is Iliskisi",
                "summary": "Sirket yeni sozlesme imzaladi.",
                "stockCodes": "AAA",
                "disclosureIndex": 1,
            },
            {
                "publishDate": "01.06.2026 11:00:00",
                "kapTitle": "BBB A.S.",
                "subject": "Ozel Durum Aciklamasi",
                "summary": "Sirket aleyhine onemli gelisme acikladi.",
                "stockCodes": "BBB",
                "disclosureIndex": 2,
            },
        ]

    def fake_score_news(model_path, input_csv, out_csv, daily_out_csv):
        prepared = pd.read_csv(input_csv)
        scored = prepared.copy()
        scored["prob_negative"] = [0.05, 0.90]
        scored["prob_neutral"] = [0.10, 0.05]
        scored["prob_positive"] = [0.85, 0.05]
        scored["prediction"] = ["positive", "negative"]
        scored["sentiment_score"] = scored["prob_positive"] - scored["prob_negative"]
        scored.to_csv(out_csv, index=False)
        daily = (
            scored.groupby(["date", "ticker"], as_index=False)
            .agg(
                sentiment_score=("sentiment_score", "mean"),
                prob_negative=("prob_negative", "mean"),
                prob_neutral=("prob_neutral", "mean"),
                prob_positive=("prob_positive", "mean"),
                news_count=("text", "size"),
            )
            .sort_values(["ticker", "date"])
        )
        daily.to_csv(daily_out_csv, index=False)
        return scored, daily

    monkeypatch.setattr(daily_pipeline, "fetch_kap_api", fake_fetch_kap_api)
    monkeypatch.setattr(daily_pipeline, "score_news", fake_score_news)

    kwargs = dict(
        model_path=tmp_path / "model.joblib",
        tickers_file=tickers_file,
        raw_out=raw_out,
        prepared_out=prepared_out,
        scored_out=scored_out,
        daily_out=daily_out,
        alerts_dir=alerts_dir,
        update_history=True,
        historical_raw_out=historical_raw,
        historical_prepared_out=historical_prepared,
        historical_scored_out=historical_scored,
        historical_daily_out=historical_daily,
        kap_days=7,
        top_n=1,
        min_abs_score=0.20,
    )

    first = daily_pipeline.run_daily_pipeline(**kwargs)
    second = daily_pipeline.run_daily_pipeline(**kwargs)

    assert first["baseline_daily_sentiment"] == historical_daily
    assert second["historical_raw_rows"] == 2
    assert second["historical_prepared_rows"] == 2
    assert second["historical_scored_rows"] == 2
    assert second["historical_daily_rows"] == 2
    assert len(pd.read_csv(historical_raw)) == 2
    assert len(pd.read_csv(historical_prepared)) == 2
    assert len(pd.read_csv(historical_scored)) == 2
    assert len(pd.read_csv(historical_daily)) == 2
