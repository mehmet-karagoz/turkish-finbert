import pandas as pd

from turkish_fin_bert.prepare_dataset import prepare_dataset


def test_prepare_dataset_accepts_multiple_input_files(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    output = tmp_path / "prepared.csv"
    rows = [
        {
            "date": "2026-05-26",
            "ticker": "THYAO",
            "source": "kap",
            "title": "THYAO haber",
            "text": "Turk Hava Yollari icin yeterince uzun haber metni.",
            "url": "u1",
        },
        {
            "date": "2026-05-27",
            "ticker": "ASELS.IS",
            "source": "kap",
            "title": "ASELS haber",
            "text": "Aselsan icin yeterince uzun haber metni.",
            "url": "u2",
        },
    ]
    pd.DataFrame([rows[0]]).to_csv(first, index=False)
    pd.DataFrame([rows[0], rows[1]]).to_csv(second, index=False)

    out = prepare_dataset([first, second], output)

    assert len(out) == 2
    assert out["ticker"].tolist() == ["THYAO", "ASELS"]
    assert output.exists()
