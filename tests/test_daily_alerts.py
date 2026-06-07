import pandas as pd

from turkish_fin_bert.daily_alerts import build_baseline_daily_frame, generate_daily_alerts, main


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
    assert "priority_score" in result["event_signals"].columns
    assert "priority_reason" in result["event_signals"].columns
    assert result["event_signals"]["priority_score"].between(0, 100).all()
    assert (out_dir / "2026-06-01_daily_alerts.md").exists()
    assert (out_dir / "2026-06-01_brief.md").exists()
    assert (out_dir / "2026-06-01_telegram.html").exists()
    assert (out_dir / "2026-06-01_event_signals.csv").exists()
    assert (out_dir / "2026-06-01_signal_baseline.csv").exists()
    report_text = (out_dir / "2026-06-01_daily_alerts.md").read_text(encoding="utf-8")
    brief_text = (out_dir / "2026-06-01_brief.md").read_text(encoding="utf-8")
    telegram_text = (out_dir / "2026-06-01_telegram.html").read_text(encoding="utf-8")
    assert "Onemli Olay Ozeti" in report_text
    assert "Gecmis Karsilastirma" in report_text
    assert "En Iyi Hisseler" in report_text
    assert "BIST Kisa Ozet - 2026-06-01" in brief_text
    assert "Gecmis karsilastirma:" in brief_text
    assert "Izlenecekler:" in brief_text
    assert "Karar:" in brief_text
    assert "Sonuc:" in brief_text
    assert "Not:" in brief_text
    assert "Akis:" in brief_text
    assert "Oncelik seviyesi:" in brief_text
    assert "Aksiyon:" in brief_text
    assert "Aksiyonlar:" in brief_text
    assert "Oncelik:" in brief_text
    assert "Dagilim:" in report_text
    assert "oncelik" in report_text
    assert "Neden:" in brief_text
    assert all(len(line) <= 180 for line in brief_text.splitlines())
    assert "(weak," not in brief_text
    assert "(strong," not in brief_text
    assert telegram_text.startswith("<b>BIST Gunluk Ozet</b>")
    assert "<b>Karar</b>" in telegram_text
    assert "<b>Izlenecekler</b>" in telegram_text
    assert "AAA" in telegram_text


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
    brief_text = (out_dir / "2026-06-01_brief.md").read_text(encoding="utf-8")

    assert result["top_stocks"].empty
    assert result["bottom_stocks"].empty
    assert result["positive_news"].empty
    assert result["negative_news"].empty
    assert result["event_signals"].empty
    assert "priority_score" in result["event_signals"].columns
    assert "Bugun esigi asan anlamli haber/hisse sinyali yok." in text
    assert "Durum: Bugun esigi asan anlamli haber/hisse sinyali yok." in brief_text
    assert "Karar: Bugun aksiyon gerektiren olay yok." in brief_text
    assert "Sonuc: Onemli olay yok; takip listesi bos." in brief_text
    assert "Akis: 2 haber, 2 hisse, 0 esik ustu hisse, 0 olay, 0 piyasa geneli aday" in brief_text
    assert "Oncelik seviyesi: yok" in brief_text
    assert "Aksiyonlar: aksiyon gerektiren olay yok" in brief_text
    assert "- Yok" in brief_text
    assert "Sinyal esigi: |skor| >= 0.20" in text
    assert "Gunluk Karar" in text


def test_generate_daily_alerts_adds_historical_anomaly_context(tmp_path):
    scored = tmp_path / "scored.csv"
    daily = tmp_path / "daily.csv"
    out_dir = tmp_path / "alerts"

    pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "ticker": "AAA",
                "title": "AAA yeni sozlesme imzaladi",
                "text": "Sirket onemli yeni is iliskisi acikladi.",
                "url": "u1",
                "prediction": "positive",
                "prob_negative": 0.05,
                "prob_neutral": 0.10,
                "prob_positive": 0.85,
                "sentiment_score": 0.80,
            },
        ]
    ).to_csv(scored, index=False)
    rows = [
        {
            "date": date,
            "ticker": "AAA",
            "sentiment_score": score,
            "prob_negative": 0.30,
            "prob_neutral": 0.40,
            "prob_positive": 0.35,
            "news_count": 1,
        }
        for date, score in [
            ("2026-05-27", 0.04),
            ("2026-05-28", 0.05),
            ("2026-05-29", 0.06),
            ("2026-05-30", 0.04),
            ("2026-05-31", 0.05),
        ]
    ]
    rows.append(
        {
            "date": "2026-06-01",
            "ticker": "AAA",
            "sentiment_score": 0.80,
            "prob_negative": 0.05,
            "prob_neutral": 0.10,
            "prob_positive": 0.85,
            "news_count": 1,
        }
    )
    rows.append(
        {
            "date": "2026-06-02",
            "ticker": "AAA",
            "sentiment_score": -0.80,
            "prob_negative": 0.85,
            "prob_neutral": 0.10,
            "prob_positive": 0.05,
            "news_count": 1,
        }
    )
    pd.DataFrame(rows).to_csv(daily, index=False)

    result = generate_daily_alerts(scored, daily, out_dir, date="2026-06-01", top_n=10)
    brief_text = (out_dir / "2026-06-01_brief.md").read_text(encoding="utf-8")

    assert result["signal_baseline"].iloc[0]["history_days"] == 5
    assert result["signal_baseline"].iloc[0]["anomaly_level"] == "unusual"
    assert result["event_signals"].iloc[0]["priority_score"] >= 60
    assert "1 sira disi" in brief_text
    assert "Oncelik:" in brief_text
    assert "AAA gecmise gore sira disi" in brief_text
    assert all(len(line) <= 180 for line in brief_text.splitlines())


def test_generate_daily_alerts_marks_weak_elevated_management_event_for_review(tmp_path):
    scored = tmp_path / "scored.csv"
    daily = tmp_path / "daily.csv"
    baseline_daily = tmp_path / "baseline_daily.csv"
    out_dir = tmp_path / "alerts"

    pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "ticker": "AAA",
                "title": "AAA teknik yonetime iliskin aciklama",
                "text": "Sirket teknik yonetim tarafinda yeni atama acikladi.",
                "url": "u1",
                "prediction": "positive",
                "prob_negative": 0.40,
                "prob_neutral": 0.00,
                "prob_positive": 0.62,
                "sentiment_score": 0.22,
            },
        ]
    ).to_csv(scored, index=False)
    pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "ticker": "AAA",
                "sentiment_score": 0.22,
                "prob_negative": 0.40,
                "prob_neutral": 0.00,
                "prob_positive": 0.62,
                "news_count": 1,
            }
        ]
    ).to_csv(daily, index=False)
    pd.DataFrame(
        [
            {
                "date": date,
                "ticker": "AAA",
                "sentiment_score": score,
                "prob_negative": 0.33,
                "prob_neutral": 0.34,
                "prob_positive": 0.32,
                "news_count": 1,
            }
            for date, score in [
                ("2026-05-27", -0.02),
                ("2026-05-28", -0.01),
                ("2026-05-29", 0.00),
                ("2026-05-30", 0.01),
                ("2026-05-31", -0.02),
            ]
        ]
    ).to_csv(baseline_daily, index=False)

    result = generate_daily_alerts(
        scored,
        daily,
        out_dir,
        baseline_daily_sentiment_csv=baseline_daily,
        date="2026-06-01",
        top_n=10,
    )
    brief_text = (out_dir / "2026-06-01_brief.md").read_text(encoding="utf-8")

    assert result["event_signals"].iloc[0]["signal_strength"] == "weak"
    assert result["signal_baseline"].iloc[0]["anomaly_level"] == "elevated"
    assert result["event_signals"].iloc[0]["priority_score"] > 0
    assert "Aksiyon: detay kontrol et | pozitif zayif +0.22" in brief_text
    assert "Karar: Bugun dusuk oncelikli manuel kontrol sinyali var." in brief_text
    assert "Sonuc: Zayif sinyaller var; manuel kontrol disinda aksiyon gerektirmiyor." in brief_text
    assert "Oncelik seviyesi: dusuk" in brief_text
    assert "Aksiyonlar: 1 detay kontrol et" in brief_text
    assert "Neden:" in brief_text
    assert "yonetim haberi; AAA gecmise gore dikkat cekici" in brief_text
    assert all(len(line) <= 180 for line in brief_text.splitlines())


def test_generate_daily_alerts_can_use_separate_baseline_daily_file(tmp_path):
    scored = tmp_path / "scored.csv"
    daily = tmp_path / "daily.csv"
    baseline_daily = tmp_path / "baseline_daily.csv"
    out_dir = tmp_path / "alerts"

    pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "ticker": "AAA",
                "title": "AAA yeni sozlesme imzaladi",
                "text": "Sirket onemli yeni is iliskisi acikladi.",
                "url": "u1",
                "prediction": "positive",
                "prob_negative": 0.05,
                "prob_neutral": 0.10,
                "prob_positive": 0.85,
                "sentiment_score": 0.80,
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
            }
        ]
    ).to_csv(daily, index=False)
    rows = [
        {
            "date": date,
            "ticker": "AAA",
            "sentiment_score": score,
            "prob_negative": 0.30,
            "prob_neutral": 0.40,
            "prob_positive": 0.35,
            "news_count": 1,
        }
        for date, score in [
            ("2026-05-27", 0.04),
            ("2026-05-28", 0.05),
            ("2026-05-29", 0.06),
            ("2026-05-30", 0.04),
            ("2026-05-31", 0.05),
            ("2026-06-02", -0.80),
        ]
    ]
    pd.DataFrame(rows).to_csv(baseline_daily, index=False)

    result = generate_daily_alerts(
        scored,
        daily,
        out_dir,
        baseline_daily_sentiment_csv=baseline_daily,
        date="2026-06-01",
        top_n=10,
    )

    assert result["signal_baseline"].iloc[0]["history_days"] == 5
    assert result["signal_baseline"].iloc[0]["anomaly_level"] == "unusual"


def test_build_baseline_daily_frame_uses_current_day_from_current_file_only():
    current = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-06-01"), "ticker": "AAA", "sentiment_score": 0.80, "news_count": 1},
        ]
    )
    baseline = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-05-31"), "ticker": "AAA", "sentiment_score": 0.05, "news_count": 1},
            {"date": pd.Timestamp("2026-06-01"), "ticker": "BBB", "sentiment_score": -0.90, "news_count": 1},
            {"date": pd.Timestamp("2026-06-02"), "ticker": "AAA", "sentiment_score": -0.80, "news_count": 1},
        ]
    )

    combined = build_baseline_daily_frame(current, pd.Timestamp("2026-06-01"), baseline)

    assert combined["ticker"].tolist() == ["AAA", "AAA"]
    assert pd.Timestamp("2026-06-02") not in combined["date"].tolist()


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


def test_daily_alerts_sends_html_text_to_telegram(tmp_path, monkeypatch):
    scored = tmp_path / "scored.csv"
    daily = tmp_path / "daily.csv"
    out_dir = tmp_path / "alerts"
    sent_messages = []

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
            }
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
            }
        ]
    ).to_csv(daily, index=False)

    def fake_send(token, chat_id, text, parse_mode=None):
        sent_messages.append((token, chat_id, text, parse_mode))

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    monkeypatch.setattr("turkish_fin_bert.daily_alerts.send_telegram_message", fake_send)
    monkeypatch.setattr(
        "sys.argv",
        [
            "daily_alerts",
            "--scored-news",
            str(scored),
            "--daily-sentiment",
            str(daily),
            "--date",
            "2026-06-01",
            "--out-dir",
            str(out_dir),
            "--send-telegram",
        ],
    )

    main()

    assert len(sent_messages) == 1
    assert sent_messages[0][0] == "token"
    assert sent_messages[0][1] == "chat"
    assert sent_messages[0][2].startswith("<b>BIST Gunluk Ozet</b>")
    assert "<b>Karar</b>" in sent_messages[0][2]
    assert "## En Iyi Hisseler" not in sent_messages[0][2]
    assert "(weak," not in sent_messages[0][2]
    assert sent_messages[0][3] == "HTML"
