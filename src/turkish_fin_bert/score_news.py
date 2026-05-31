import argparse
from pathlib import Path

import joblib
import pandas as pd

from .console import configure_console
from .paths import ensure_project_dirs
from .text import combine_title_text


LABEL_ORDER = ["negative", "neutral", "positive"]


def _ensure_text(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df:
        raise ValueError("CSV içinde text kolonu olmalı.")
    if "title" in df:
        df["text"] = [combine_title_text(title, text) for title, text in zip(df["title"], df["text"])]
    return df


def score_news(model_path: Path, input_csv: Path, out_csv: Path, daily_out_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = joblib.load(model_path)
    df = pd.read_csv(input_csv)
    df = _ensure_text(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["ticker"] = df["ticker"].astype(str).str.upper().str.replace(".IS", "", regex=False).str.strip()

    proba = model.predict_proba(df["text"])
    classes = list(model.classes_)
    for label in LABEL_ORDER:
        if label in classes:
            df[f"prob_{label}"] = proba[:, classes.index(label)]
        else:
            df[f"prob_{label}"] = 0.0
    df["prediction"] = model.predict(df["text"])
    df["sentiment_score"] = df["prob_positive"] - df["prob_negative"]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    daily = (
        df.dropna(subset=["date", "ticker"])
        .groupby(["date", "ticker"], as_index=False)
        .agg(
            sentiment_score=("sentiment_score", "mean"),
            prob_negative=("prob_negative", "mean"),
            prob_neutral=("prob_neutral", "mean"),
            prob_positive=("prob_positive", "mean"),
            news_count=("text", "size"),
        )
        .sort_values(["ticker", "date"])
    )
    for window in [3, 7, 14]:
        daily[f"sentiment_{window}d"] = daily.groupby("ticker")["sentiment_score"].transform(lambda s: s.rolling(window, min_periods=1).mean())

    daily_out_csv.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(daily_out_csv, index=False)
    return df, daily


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eğitilmiş sentiment modeliyle haberleri skorlar.")
    parser.add_argument("--model", type=Path, required=True, help="Joblib model yolu.")
    parser.add_argument("--input", type=Path, required=True, help="Skorlanacak haber CSV.")
    parser.add_argument("--out", type=Path, default=Path("data/processed/scored_news.csv"), help="Satır bazlı skor CSV.")
    parser.add_argument("--daily-out", type=Path, default=Path("data/processed/daily_sentiment.csv"), help="Günlük hisse sentiment CSV.")
    return parser


def main() -> None:
    configure_console()
    ensure_project_dirs()
    args = build_parser().parse_args()
    scored, daily = score_news(args.model, args.input, args.out, args.daily_out)
    print(f"{len(scored)} metin skorlandı: {args.out}")
    print(f"{len(daily)} günlük hisse skoru üretildi: {args.daily_out}")


if __name__ == "__main__":
    main()
