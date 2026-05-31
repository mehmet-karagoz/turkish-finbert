import argparse
from pathlib import Path

import pandas as pd

from .console import configure_console
from .paths import FIGURE_DIR, ensure_project_dirs
from .text import combine_title_text, clean_text
from .visualize import plot_daily_counts, plot_label_distribution, plot_source_counts, plot_text_lengths, plot_ticker_label_distribution


LABELS = {"negative", "neutral", "positive"}
REQUIRED_COLUMNS = {"date", "ticker", "title", "text", "source", "url"}


def prepare_dataset(input_csv: Path, output_csv: Path, labeled: bool = False, min_chars: int = 20) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Eksik kolonlar: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["ticker"] = df["ticker"].astype(str).str.upper().str.replace(".IS", "", regex=False).str.strip()
    df["title"] = df["title"].map(clean_text)
    df["text"] = [combine_title_text(title, text) for title, text in zip(df["title"], df["text"])]
    df["source"] = df["source"].fillna("unknown").map(clean_text)
    df["url"] = df["url"].fillna("").astype(str).str.strip()

    df = df[df["text"].str.len() >= min_chars].copy()
    df = df.drop_duplicates(subset=["date", "ticker", "title", "text"])

    if labeled:
        if "label" not in df.columns:
            raise ValueError("labeled=True için CSV içinde label kolonu olmalı.")
        df["label"] = df["label"].astype(str).str.lower().str.strip()
        bad_labels = sorted(set(df["label"]) - LABELS)
        if bad_labels:
            raise ValueError(f"Geçersiz label değerleri: {bad_labels}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    plot_text_lengths(df, FIGURE_DIR / "text_length_distribution.png")
    plot_daily_counts(df, FIGURE_DIR / "daily_text_counts.png")
    plot_source_counts(df, FIGURE_DIR / "source_distribution.png")
    if labeled:
        plot_label_distribution(df, FIGURE_DIR / "label_distribution.png")
        plot_ticker_label_distribution(df, FIGURE_DIR / "ticker_label_distribution.png")

    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ham Türkçe finans metinlerini temizler ve model veri setine çevirir.")
    parser.add_argument("--input", type=Path, required=True, help="Ham CSV yolu.")
    parser.add_argument("--output", type=Path, required=True, help="Temizlenmiş CSV yolu.")
    parser.add_argument("--labeled", action="store_true", help="CSV içinde label kolonu varsa açılır.")
    parser.add_argument("--min-chars", type=int, default=20, help="Minimum metin uzunluğu.")
    return parser


def main() -> None:
    configure_console()
    ensure_project_dirs()
    args = build_parser().parse_args()
    df = prepare_dataset(args.input, args.output, labeled=args.labeled, min_chars=args.min_chars)
    print(f"{len(df)} satır hazırlandı: {args.output}")


if __name__ == "__main__":
    main()
