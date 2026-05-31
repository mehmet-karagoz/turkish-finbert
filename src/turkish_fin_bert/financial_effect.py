import argparse
from pathlib import Path

import pandas as pd

from .console import configure_console
from .paths import FIGURE_DIR, ensure_project_dirs
from .visualize import plot_bucket_returns, plot_price_sentiment, plot_return_correlation


def load_prices(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path)
    needed = {"date", "ticker", "close"}
    missing = needed - set(prices.columns)
    if missing:
        raise ValueError(f"Fiyat CSV içinde eksik kolonlar: {sorted(missing)}")
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.replace(".IS", "", regex=False).str.strip()
    return prices.dropna(subset=["date", "ticker", "close"]).sort_values(["ticker", "date"])


def add_forward_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    for days in [1, 5, 20]:
        prices[f"forward_return_{days}d"] = prices.groupby("ticker")["close"].shift(-days) / prices["close"] - 1
    return prices


def analyze_effect(sentiment_csv: Path, prices_csv: Path, out_csv: Path) -> pd.DataFrame:
    sentiment = pd.read_csv(sentiment_csv)
    sentiment["date"] = pd.to_datetime(sentiment["date"], errors="coerce")
    sentiment["ticker"] = sentiment["ticker"].astype(str).str.upper().str.replace(".IS", "", regex=False).str.strip()

    prices = add_forward_returns(load_prices(prices_csv))
    merged = sentiment.merge(prices, on=["date", "ticker"], how="inner")
    if merged.empty:
        raise RuntimeError("Sentiment ve fiyat verisi tarih/ticker bazında eşleşmedi.")

    try:
        merged["sentiment_bucket"] = pd.qcut(merged["sentiment_score"], q=5, labels=["çok düşük", "düşük", "orta", "yüksek", "çok yüksek"], duplicates="drop")
    except ValueError:
        merged["sentiment_bucket"] = "tek seviye"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)

    plot_bucket_returns(merged, FIGURE_DIR / "sentiment_bucket_returns.png")
    plot_return_correlation(merged, FIGURE_DIR / "sentiment_return_correlation.png")
    for ticker in merged["ticker"].value_counts().head(5).index:
        plot_price_sentiment(merged, ticker, FIGURE_DIR / f"price_sentiment_{ticker}.png")
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sentiment skorunun ileri getirilerle ilişkisini analiz eder.")
    parser.add_argument("--sentiment", type=Path, required=True, help="Günlük sentiment CSV.")
    parser.add_argument("--prices", type=Path, required=True, help="Fiyat CSV.")
    parser.add_argument("--out", type=Path, default=Path("reports/financial_effect.csv"), help="Birleşik analiz CSV.")
    return parser


def main() -> None:
    configure_console()
    ensure_project_dirs()
    args = build_parser().parse_args()
    df = analyze_effect(args.sentiment, args.prices, args.out)
    print(f"{len(df)} eşleşmiş gözlem analiz edildi: {args.out}")


if __name__ == "__main__":
    main()
