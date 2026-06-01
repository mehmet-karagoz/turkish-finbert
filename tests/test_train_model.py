import pandas as pd

from turkish_fin_bert.train_model import train_baseline


def test_train_baseline_stratified_split_keeps_each_label_in_test(tmp_path):
    input_csv = tmp_path / "labeled.csv"
    model_out = tmp_path / "model.joblib"
    report_dir = tmp_path / "reports"
    rows = []
    for label in ["negative", "neutral", "positive"]:
        for i in range(6):
            rows.append(
                {
                    "date": f"2026-01-{i + 1:02d}",
                    "ticker": label[:3].upper(),
                    "title": f"{label} title {i}",
                    "text": f"{label} finans haberi {i}",
                    "label": label,
                }
            )
    pd.DataFrame(rows).to_csv(input_csv, index=False)

    metrics = train_baseline(input_csv, model_out, report_dir, split_strategy="stratified", test_size=0.34)

    assert metrics["split_strategy"] == "stratified"
    assert metrics["fit_rows"] == 18
    assert metrics["classification_report"]["negative"]["support"] > 0
    assert metrics["classification_report"]["neutral"]["support"] > 0
    assert metrics["classification_report"]["positive"]["support"] > 0
    assert model_out.exists()
