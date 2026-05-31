import argparse
from pathlib import Path

import pandas as pd

from .console import configure_console
from .paths import ensure_project_dirs


def audit_aliases(input_csv: Path, out_unmatched: Path, out_summary: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(input_csv)
    if "ticker" not in df.columns:
        raise ValueError("CSV içinde ticker kolonu olmalı.")

    df["ticker"] = df["ticker"].fillna("").astype(str).str.strip()
    df["is_matched"] = df["ticker"].ne("")

    summary = (
        df.groupby("source", dropna=False)
        .agg(
            total_rows=("title", "size"),
            matched_rows=("is_matched", "sum"),
        )
        .reset_index()
    )
    summary["unmatched_rows"] = summary["total_rows"] - summary["matched_rows"]
    summary["match_rate"] = summary["matched_rows"] / summary["total_rows"]

    unmatched_cols = [col for col in ["date", "source", "title", "url", "text"] if col in df.columns]
    unmatched = df[df["ticker"].eq("")][unmatched_cols].copy()

    out_unmatched.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    unmatched.to_csv(out_unmatched, index=False)
    summary.to_csv(out_summary, index=False)
    return unmatched, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Haberlerde ticker eşleşme oranını ve eşleşmeyen başlıkları raporlar.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/news_all.csv"), help="Ham haber CSV.")
    parser.add_argument("--out-unmatched", type=Path, default=Path("reports/unmatched_news.csv"), help="Ticker bulunamayan haberler.")
    parser.add_argument("--out-summary", type=Path, default=Path("reports/alias_coverage.csv"), help="Kaynak bazlı eşleşme özeti.")
    return parser


def main() -> None:
    configure_console()
    ensure_project_dirs()
    args = build_parser().parse_args()
    unmatched, summary = audit_aliases(args.input, args.out_unmatched, args.out_summary)
    total = int(summary["total_rows"].sum()) if not summary.empty else 0
    matched = int(summary["matched_rows"].sum()) if not summary.empty else 0
    rate = matched / total if total else 0
    print(f"Eşleşen haber: {matched}/{total} ({rate:.1%})")
    print(f"Eşleşmeyen haber raporu: {args.out_unmatched} ({len(unmatched)} satır)")


if __name__ == "__main__":
    main()

