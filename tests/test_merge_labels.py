import pandas as pd

from turkish_fin_bert.merge_labels import merge_labeled_batches


def test_merge_labeled_batches_keeps_only_valid_filled_labels_and_dedupes(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    out_csv = tmp_path / "master.csv"
    base = {
        "date": "2026-05-26",
        "ticker": "THYAO",
        "source": "rss",
        "title": "THY haber",
        "text": "Turk Hava Yollari icin yeterince uzun haber metni.",
        "url": "u1",
    }
    pd.DataFrame([{**base, "label": "positive"}, {**base, "url": "u2", "label": ""}]).to_csv(first, index=False)
    pd.DataFrame([{**base, "label": "neutral", "notes": "duzeltildi"}]).to_csv(second, index=False)

    merged = merge_labeled_batches([first, second], out_csv)

    assert len(merged) == 1
    assert merged.loc[0, "label"] == "neutral"
    assert merged.loc[0, "label_id"] == "L000001"
    assert out_csv.exists()
