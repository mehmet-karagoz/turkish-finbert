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


def test_create_labeling_batch_excludes_existing_labels(tmp_path):
    input_csv = tmp_path / "news.csv"
    labeled_csv = tmp_path / "labeled.csv"
    output_csv = tmp_path / "next.csv"
    rows = [
        {
            "date": "2026-05-26",
            "ticker": "THYAO",
            "source": "rss",
            "title": "THY haber",
            "text": "Turk Hava Yollari icin yeterince uzun haber metni.",
            "url": "u1",
        },
        {
            "date": "2026-05-27",
            "ticker": "ASELS",
            "source": "rss",
            "title": "ASELS haber",
            "text": "Aselsan icin yeterince uzun haber metni.",
            "url": "u2",
        },
    ]
    pd.DataFrame(rows).to_csv(input_csv, index=False)
    pd.DataFrame([{**rows[0], "label": "positive"}]).to_csv(labeled_csv, index=False)

    out = create_labeling_batch(input_csv, output_csv, exclude_labeled=[labeled_csv])

    assert out["ticker"].tolist() == ["ASELS"]


def test_create_labeling_batch_all_excluded_writes_empty_file(tmp_path):
    input_csv = tmp_path / "news.csv"
    labeled_csv = tmp_path / "labeled.csv"
    output_csv = tmp_path / "next.csv"
    row = {
        "date": "2026-05-26",
        "ticker": "THYAO",
        "source": "rss",
        "title": "THY haber",
        "text": "Turk Hava Yollari icin yeterince uzun haber metni.",
        "url": "u1",
    }
    pd.DataFrame([row]).to_csv(input_csv, index=False)
    pd.DataFrame([{**row, "label": "positive"}]).to_csv(labeled_csv, index=False)

    out = create_labeling_batch(input_csv, output_csv, exclude_labeled=[labeled_csv])

    assert out.empty
    assert output_csv.exists()


def test_create_labeling_batch_can_prioritize_uncertain_rows(tmp_path):
    input_csv = tmp_path / "scored.csv"
    output_csv = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-05-26",
                "ticker": "A",
                "source": "rss",
                "title": "A haber",
                "text": "A hissesi icin yeterince uzun haber metni.",
                "url": "u1",
                "prob_negative": 0.90,
                "prob_neutral": 0.05,
                "prob_positive": 0.05,
            },
            {
                "date": "2026-05-26",
                "ticker": "B",
                "source": "rss",
                "title": "B haber",
                "text": "B hissesi icin yeterince uzun haber metni.",
                "url": "u2",
                "prob_negative": 0.34,
                "prob_neutral": 0.33,
                "prob_positive": 0.33,
            },
        ]
    ).to_csv(input_csv, index=False)

    out = create_labeling_batch(input_csv, output_csv, max_rows=1, strategy="uncertain")

    assert out["ticker"].tolist() == ["B"]


def test_validate_labels_rejects_unknown_label(tmp_path):
    input_csv = tmp_path / "labels.csv"
    pd.DataFrame([{"label": "good"}]).to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="Geçersiz label"):
        validate_labels(input_csv)
