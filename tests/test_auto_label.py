import pandas as pd

from turkish_fin_bert.auto_label import auto_label_batch, infer_label, normalize_for_rules


def test_normalize_for_rules_handles_turkish_letters():
    assert normalize_for_rules("Payların Geri Alınması") == "paylarin geri alinmasi"


def test_infer_label_detects_positive_share_buyback():
    label, rule = infer_label("Payların Geri Alınmasına İlişkin Bildirim", "Pay geri alım işlemleri")

    assert label == "positive"
    assert rule == "share_buyback"


def test_infer_label_detects_negative_non_distribution_before_dividend():
    label, rule = infer_label("Kar Payı Dağıtım İşlemleri", "Kar payı dağıtılmamasına karar verildi.")

    assert label == "negative"
    assert rule == "dividend_not_distributed"


def test_infer_label_defaults_routine_disclosure_to_neutral():
    label, rule = infer_label("Bağımsız Denetim Kuruluşunun Belirlenmesi", "")

    assert label == "neutral"
    assert rule == "routine_governance"


def test_auto_label_batch_writes_empty_labels_and_keeps_existing_without_overwrite(tmp_path):
    input_csv = tmp_path / "batch.csv"
    pd.DataFrame(
        [
            {"title": "Yeni İş İlişkisi", "text": "Yeni iş ilişkisi açıklandı.", "label": "", "notes": ""},
            {"title": "Pay geri alımı", "text": "Pay geri alım işlemleri.", "label": "neutral", "notes": "manual"},
        ]
    ).to_csv(input_csv, index=False)

    out = auto_label_batch(input_csv)

    assert out["label"].tolist() == ["positive", "neutral"]
    assert out["notes"].tolist() == ["weak_label:new_business_or_tender", "manual"]
