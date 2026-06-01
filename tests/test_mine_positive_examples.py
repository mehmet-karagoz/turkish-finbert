import pandas as pd

from turkish_fin_bert.mine_positive_examples import high_confidence_positive_reason, mine_positive_examples


def test_high_confidence_positive_reason_detects_share_buyback():
    reason = high_confidence_positive_reason(
        "Payların Geri Alınmasına İlişkin Bildirim",
        "Pay geri alım işlemleri hakkında.",
    )

    assert reason == "share_buyback"


def test_high_confidence_positive_reason_excludes_cancelled_buyback():
    reason = high_confidence_positive_reason(
        "Özel Durum Açıklaması",
        "Pay geri alım iptali hakkında açıklama.",
    )

    assert reason == ""


def test_high_confidence_positive_reason_detects_contract():
    reason = high_confidence_positive_reason("Yeni İş İlişkisi", "Sözleşme imzalanması hakkında.")

    assert reason == "new_business_contract"


def test_mine_positive_examples_writes_positive_batch_and_excludes_labeled(tmp_path):
    input_csv = tmp_path / "news.csv"
    labeled_csv = tmp_path / "labeled.csv"
    output_csv = tmp_path / "positive.csv"
    rows = [
        {
            "date": "2026-01-01",
            "ticker": "AAA",
            "source": "KAP",
            "title": "AAA - Yeni İş İlişkisi",
            "text": "Yeni iş ilişkisi açıklandı.",
            "url": "u1",
        },
        {
            "date": "2026-01-02",
            "ticker": "BBB",
            "source": "KAP",
            "title": "BBB - Payların Geri Alınması",
            "text": "Pay geri alım işlemleri.",
            "url": "u2",
        },
    ]
    pd.DataFrame(rows).to_csv(input_csv, index=False)
    pd.DataFrame([{**rows[0], "label": "positive"}]).to_csv(labeled_csv, index=False)

    out = mine_positive_examples([input_csv], output_csv, exclude_labeled=[labeled_csv])

    assert out["ticker"].tolist() == ["BBB"]
    assert out["label"].tolist() == ["positive"]
    assert output_csv.exists()
