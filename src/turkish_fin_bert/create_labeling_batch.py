import argparse
from pathlib import Path

import pandas as pd

from .console import configure_console
from .paths import FIGURE_DIR, ensure_project_dirs
from .text import clean_text
from .visualize import plot_daily_counts, plot_source_counts, plot_text_lengths, plot_ticker_counts


BASE_COLUMNS = ["label_id", "label", "date", "ticker", "source", "title", "text", "url", "notes"]
LABELS = {"negative", "neutral", "positive"}


def _sample_rows(df: pd.DataFrame, max_rows: int, per_ticker: int, seed: int) -> pd.DataFrame:
    if per_ticker > 0 and "ticker" in df:
        parts = [group.sample(n=min(per_ticker, len(group)), random_state=seed) for _, group in df.groupby("ticker", sort=False)]
        sampled = pd.concat(parts, ignore_index=True) if parts else df.head(0).copy()
    else:
        sampled = df.copy()

    if max_rows > 0 and len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=seed)
    return sampled.sort_values(["date", "ticker", "source", "title"], ascending=[False, True, True, True]).reset_index(drop=True)


def create_labeling_batch(
    input_csv: Path,
    output_csv: Path,
    max_rows: int = 300,
    per_ticker: int = 0,
    seed: int = 42,
    min_chars: int = 20,
    include_untagged: bool = False,
    include_model_hints: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required = {"date", "ticker", "title", "text", "source", "url"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Eksik kolonlar: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["ticker"] = df["ticker"].fillna("").astype(str).str.upper().str.replace(".IS", "", regex=False).str.strip()
    df["title"] = df["title"].map(clean_text)
    df["text"] = df["text"].map(clean_text)
    df["source"] = df["source"].fillna("unknown").map(clean_text)
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() >= min_chars].copy()
    if not include_untagged:
        df = df[df["ticker"].ne("")].copy()
    df = df.drop_duplicates(subset=["ticker", "title", "text", "url"])

    sampled = _sample_rows(df, max_rows=max_rows, per_ticker=per_ticker, seed=seed)
    sampled.insert(0, "label_id", [f"N{i:06d}" for i in range(1, len(sampled) + 1)])
    sampled.insert(1, "label", "")
    sampled["notes"] = ""

    columns = BASE_COLUMNS.copy()
    if include_model_hints:
        hint_map = {
            "prediction": "model_prediction",
            "sentiment_score": "model_sentiment_score",
            "prob_negative": "model_prob_negative",
            "prob_neutral": "model_prob_neutral",
            "prob_positive": "model_prob_positive",
        }
        for source_col, target_col in hint_map.items():
            if source_col in sampled.columns:
                sampled[target_col] = sampled[source_col]
                columns.append(target_col)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out = sampled[columns].copy()
    out.to_csv(output_csv, index=False)

    plot_text_lengths(out, FIGURE_DIR / "labeling_text_length_distribution.png")
    plot_daily_counts(out, FIGURE_DIR / "labeling_daily_text_counts.png")
    plot_source_counts(out, FIGURE_DIR / "labeling_source_distribution.png")
    plot_ticker_counts(out, FIGURE_DIR / "labeling_ticker_distribution.png")
    return out


def validate_labels(input_csv: Path) -> tuple[int, int]:
    df = pd.read_csv(input_csv)
    if "label" not in df.columns:
        raise ValueError("CSV içinde label kolonu olmalı.")
    labels = df["label"].fillna("").astype(str).str.lower().str.strip()
    filled = labels.ne("")
    bad = sorted(set(labels[filled]) - LABELS)
    if bad:
        raise ValueError(f"Geçersiz label değerleri: {bad}")
    return int(filled.sum()), int(len(df))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gerçek haberlerden manuel etiketleme için sade CSV üretir.")
    parser.add_argument("--input", type=Path, required=True, help="Hazırlanmış veya skorlanmış haber CSV yolu.")
    parser.add_argument("--output", type=Path, default=Path("data/labels/real_labeling_batch.csv"), help="Etiketleme CSV çıktısı.")
    parser.add_argument("--max-rows", type=int, default=300, help="Seçilecek maksimum haber sayısı. 0 sınırsız.")
    parser.add_argument("--per-ticker", type=int, default=0, help="Her ticker için maksimum haber sayısı. 0 kapalı.")
    parser.add_argument("--seed", type=int, default=42, help="Tekrarlanabilir örnekleme tohumu.")
    parser.add_argument("--min-chars", type=int, default=20, help="Minimum metin uzunluğu.")
    parser.add_argument("--include-untagged", action="store_true", help="Ticker bulunmayan haberleri de etiketleme havuzuna al.")
    parser.add_argument("--include-model-hints", action="store_true", help="Varsa model tahmini ve olasılıklarını yardımcı kolon olarak ekle.")
    parser.add_argument("--validate-only", action="store_true", help="Yeni CSV üretmeden mevcut label kolonunu kontrol eder.")
    return parser


def main() -> None:
    configure_console()
    ensure_project_dirs()
    args = build_parser().parse_args()
    if args.validate_only:
        filled, total = validate_labels(args.input)
        print(f"Etiket kontrolü tamamlandı: {filled}/{total} satır etiketli.")
        return

    df = create_labeling_batch(
        input_csv=args.input,
        output_csv=args.output,
        max_rows=args.max_rows,
        per_ticker=args.per_ticker,
        seed=args.seed,
        min_chars=args.min_chars,
        include_untagged=args.include_untagged,
        include_model_hints=args.include_model_hints,
    )
    print(f"{len(df)} haber etiketleme için hazırlandı: {args.output}")


if __name__ == "__main__":
    main()
