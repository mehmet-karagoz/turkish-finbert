import pandas as pd

from turkish_fin_bert.mine_negative_examples import high_confidence_negative_reason, mine_negative_examples


def test_high_confidence_negative_reason_excludes_company_favorable_legal_result():
    reason = high_confidence_negative_reason(
        "Özel Durum Açıklaması",
        "Ceza ihbarnamesinin iptaline ilişkin karar şirketimiz lehine sonuçlandı.",
    )

    assert reason == ""


def test_high_confidence_negative_reason_detects_non_distribution():
    reason = high_confidence_negative_reason(
        "Kar Payı Dağıtım İşlemleri",
        "Kar payı dağıtılmamasına ilişkin genel kurul sonucu.",
    )

    assert reason == "dividend_not_distributed"


def test_mine_negative_examples_writes_negative_batch_and_excludes_labeled(tmp_path):
    input_csv = tmp_path / "news.csv"
    labeled_csv = tmp_path / "labeled.csv"
    output_csv = tmp_path / "negative.csv"
    rows = [
        {
            "date": "2026-01-01",
            "ticker": "AAA",
            "source": "KAP",
            "title": "AAA - Ortaklık Aleyhine Dava Açılması",
            "text": "Şirket aleyhine dava açılması hakkında.",
            "url": "u1",
        },
        {
            "date": "2026-01-02",
            "ticker": "BBB",
            "source": "KAP",
            "title": "BBB - Kar Payı",
            "text": "Kar payı dağıtılmamasına karar verilmiştir.",
            "url": "u2",
        },
    ]
    pd.DataFrame(rows).to_csv(input_csv, index=False)
    pd.DataFrame([{**rows[0], "label": "negative"}]).to_csv(labeled_csv, index=False)

    out = mine_negative_examples([input_csv], output_csv, exclude_labeled=[labeled_csv])

    assert out["ticker"].tolist() == ["BBB"]
    assert out["label"].tolist() == ["negative"]
    assert output_csv.exists()
