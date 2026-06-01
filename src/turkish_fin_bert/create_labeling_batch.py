import argparse
from pathlib import Path

import pandas as pd

from .console import configure_console
from .paths import FIGURE_DIR, ensure_project_dirs
from .text import clean_text
from .visualize import plot_daily_counts, plot_source_counts, plot_text_lengths, plot_ticker_counts


BASE_COLUMNS = ["label_id", "label", "date", "ticker", "source", "title", "text", "url", "notes"]
LABELS = {"negative", "neutral", "positive"}
PROB_COLUMNS = ["prob_negative", "prob_neutral", "prob_positive"]
MODEL_PROB_COLUMNS = ["model_prob_negative", "model_prob_neutral", "model_prob_positive"]


def row_key(df: pd.DataFrame) -> pd.Series:
    url = df.get("url", pd.Series("", index=df.index)).fillna("").astype(str).str.strip().str.lower()
    fallback = (
        df.get("ticker", pd.Series("", index=df.index)).fillna("").astype(str).str.upper().str.strip()
        + "|"
        + df.get("title", pd.Series("", index=df.index)).fillna("").astype(str).map(clean_text).str.lower()
        + "|"
        + df.get("text", pd.Series("", index=df.index)).fillna("").astype(str).map(clean_text).str.lower()
    )
    return url.where(url.ne(""), fallback)


def _exclude_labeled_rows(df: pd.DataFrame, labeled_paths: list[Path]) -> pd.DataFrame:
    if not labeled_paths:
        return df
    keys: set[str] = set()
    for path in labeled_paths:
        if not path.exists():
            continue
        labeled = pd.read_csv(path)
        if "label" in labeled.columns:
            labels = labeled["label"].fillna("").astype(str).str.strip()
            labeled = labeled[labels.ne("")]
        keys.update(row_key(labeled).dropna().astype(str).tolist())
    if not keys:
        return df
    return df[~row_key(df).isin(keys)].copy()


def _rank_rows(df: pd.DataFrame, strategy: str, seed: int) -> pd.DataFrame:
    if strategy == "random":
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if strategy == "uncertain":
        prob_cols = [col for col in PROB_COLUMNS if col in df.columns]
        if len(prob_cols) != 3:
            prob_cols = [col for col in MODEL_PROB_COLUMNS if col in df.columns]
        if len(prob_cols) == 3:
            ranked = df.copy()
            ranked["_uncertainty"] = 1 - ranked[prob_cols].max(axis=1)
            return ranked.sort_values(["_uncertainty", "date"], ascending=[False, False]).drop(columns=["_uncertainty"])
    return df.sort_values(["date", "ticker", "source", "title"], ascending=[False, True, True, True])


def _sample_rows(df: pd.DataFrame, max_rows: int, per_ticker: int, seed: int, strategy: str) -> pd.DataFrame:
    df = _rank_rows(df, strategy=strategy, seed=seed)
    if per_ticker > 0 and "ticker" in df:
        parts = [group.head(per_ticker) for _, group in df.groupby("ticker", sort=False)]
        sampled = pd.concat(parts, ignore_index=True) if parts else df.head(0).copy()
    else:
        sampled = df.copy()

    if max_rows > 0 and len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=seed) if strategy == "random" else sampled.head(max_rows)
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
    exclude_labeled: list[Path] | None = None,
    strategy: str = "recent",
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
    df = _exclude_labeled_rows(df, exclude_labeled or [])

    sampled = _sample_rows(df, max_rows=max_rows, per_ticker=per_ticker, seed=seed, strategy=strategy)
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
    parser.add_argument("--exclude-labeled", action="append", type=Path, default=[], help="Daha önce etiketlenmiş CSV. Birden fazla kez verilebilir.")
    parser.add_argument("--strategy", choices=["recent", "random", "uncertain"], default="recent", help="Aday seçimi: yeni, rastgele veya modelin kararsız kaldığı haberler.")
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
        exclude_labeled=args.exclude_labeled,
        strategy=args.strategy,
    )
    print(f"{len(df)} haber etiketleme için hazırlandı: {args.output}")


if __name__ == "__main__":
    main()
