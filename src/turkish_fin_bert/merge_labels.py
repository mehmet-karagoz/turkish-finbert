import argparse
from pathlib import Path

import pandas as pd

from .console import configure_console
from .create_labeling_batch import BASE_COLUMNS, LABELS, row_key
from .paths import FIGURE_DIR, ensure_project_dirs
from .text import clean_text
from .visualize import plot_daily_counts, plot_label_distribution, plot_text_lengths, plot_ticker_label_distribution


def _normalize_labeled_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "ticker", "title", "text", "source", "url", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} içinde eksik kolonlar: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["ticker"] = df["ticker"].fillna("").astype(str).str.upper().str.replace(".IS", "", regex=False).str.strip()
    df["title"] = df["title"].map(clean_text)
    df["text"] = df["text"].map(clean_text)
    df["source"] = df["source"].fillna("unknown").map(clean_text)
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].fillna("").astype(str).str.lower().str.strip()
    df = df[df["label"].ne("")].copy()

    bad = sorted(set(df["label"]) - LABELS)
    if bad:
        raise ValueError(f"{path} içinde geçersiz label değerleri: {bad}")
    if "notes" not in df.columns:
        df["notes"] = ""
    if "label_id" not in df.columns:
        df["label_id"] = ""
    return df


def merge_labeled_batches(inputs: list[Path], output_csv: Path) -> pd.DataFrame:
    frames = [_normalize_labeled_frame(path) for path in inputs]
    if not frames:
        raise ValueError("En az bir etiketli CSV verilmeli.")

    merged = pd.concat(frames, ignore_index=True)
    merged["_row_key"] = row_key(merged)
    merged = merged.drop_duplicates(subset=["_row_key"], keep="last").drop(columns=["_row_key"])
    merged = merged.sort_values(["date", "ticker", "source", "title"]).reset_index(drop=True)
    merged["label_id"] = [f"L{i:06d}" for i in range(1, len(merged) + 1)]

    columns = BASE_COLUMNS + [col for col in merged.columns if col not in BASE_COLUMNS]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out = merged[columns].copy()
    out.to_csv(output_csv, index=False)

    plot_label_distribution(out, FIGURE_DIR / "master_label_distribution.png")
    plot_text_lengths(out, FIGURE_DIR / "master_text_length_distribution.png")
    plot_daily_counts(out, FIGURE_DIR / "master_daily_text_counts.png")
    plot_ticker_label_distribution(out, FIGURE_DIR / "master_ticker_label_distribution.png")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Etiketlenmiş haber batch dosyalarını tek master CSV içinde birleştirir.")
    parser.add_argument("--input", action="append", type=Path, required=True, help="Etiketlenmiş CSV. Birden fazla kez verilebilir.")
    parser.add_argument("--output", type=Path, default=Path("data/labels/labeled_news_master.csv"), help="Master etiketli CSV çıktısı.")
    return parser


def main() -> None:
    configure_console()
    ensure_project_dirs()
    args = build_parser().parse_args()
    df = merge_labeled_batches(args.input, args.output)
    print(f"{len(df)} etiketli haber master dosyaya yazıldı: {args.output}")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
