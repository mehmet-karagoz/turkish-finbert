import pandas as pd
import pytest

from turkish_fin_bert.create_labeling_batch import create_labeling_batch, validate_labels


def test_create_labeling_batch_filters_and_adds_empty_label(tmp_path):
    input_csv = tmp_path / "news.csv"
    output_csv = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-05-26",
                "ticker": "THYAO",
                "source": "rss",
                "title": "THY iyi trafik acikladi",
                "text": "Turk Hava Yollari yolcu sayisi ve doluluk oraninda artis acikladi.",
                "url": "u1",
                "prediction": "positive",
                "sentiment_score": 0.5,
            },
            {
                "date": "2026-05-27",
                "ticker": "",
                "source": "rss",
                "title": "Genel ekonomi",
                "text": "Genel piyasa haberi ve kisa yorum.",
                "url": "u2",
                "prediction": "neutral",
                "sentiment_score": 0.0,
            },
        ]
    ).to_csv(input_csv, index=False)

    out = create_labeling_batch(input_csv, output_csv, include_model_hints=True)

    assert len(out) == 1
    assert out.loc[0, "label_id"] == "N000001"
    assert out.loc[0, "label"] == ""
    assert out.loc[0, "model_prediction"] == "positive"
    assert output_csv.exists()


def test_validate_labels_rejects_unknown_label(tmp_path):
    input_csv = tmp_path / "labels.csv"
    pd.DataFrame([{"label": "good"}]).to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="Geçersiz label"):
        validate_labels(input_csv)
