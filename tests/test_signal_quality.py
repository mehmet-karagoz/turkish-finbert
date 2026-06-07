import pandas as pd

from turkish_fin_bert.signal_quality import build_event_signals, classify_event


def test_classify_event_handles_routine_before_management():
    event_type, weight = classify_event("Kurumsal Yonetim Bilgi Formu Guncelleme - Yonetim Kurulu")

    assert event_type == "routine"
    assert weight < 0.20


def test_classify_event_does_not_treat_company_name_yatirim_as_contract():
    event_type, _ = classify_event("BORUSAN YATIRIM VE PAZARLAMA A.S. - Ozel Durum Aciklamasi Genel")

    assert event_type == "other"


def test_classify_event_detects_material_categories():
    assert classify_event("Sirket pay geri alim programi acikladi")[0] == "capital_action"
    assert classify_event("Yeni is iliskisi ve sozlesme imzalandi")[0] == "contract_order"
    assert classify_event("Sirket aleyhine dava ve ceza karari aciklandi")[0] == "legal_regulatory"


def test_build_event_signals_suppresses_routine_events_even_with_sentiment_score():
    news = pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "ticker": "AAA",
                "title": "AAA - Kurumsal Yonetim Bilgi Formu Guncelleme",
                "text": "Kurumsal Yonetim Bilgi Formu Guncelleme - Yonetim Kurulu",
                "url": "u1",
                "prob_negative": 0.10,
                "prob_positive": 0.60,
                "impact_score": 0.50,
                "confidence": 0.60,
            }
        ]
    )

    events = build_event_signals(news, min_abs_score=0.20, top_n=10)

    assert events.empty
