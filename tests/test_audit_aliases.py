import pandas as pd

from turkish_fin_bert.audit_aliases import audit_aliases


def test_audit_aliases_reports_unmatched_rows(tmp_path):
    input_csv = tmp_path / "news.csv"
    unmatched_csv = tmp_path / "unmatched.csv"
    summary_csv = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            {"date": "2024-01-01", "ticker": "THYAO", "source": "rss", "title": "THYAO haber", "url": "u1"},
            {"date": "2024-01-02", "ticker": "", "source": "rss", "title": "Genel haber", "url": "u2"},
        ]
    ).to_csv(input_csv, index=False)

    unmatched, summary = audit_aliases(input_csv, unmatched_csv, summary_csv)

    assert len(unmatched) == 1
    assert summary.loc[0, "matched_rows"] == 1
    assert summary.loc[0, "match_rate"] == 0.5
