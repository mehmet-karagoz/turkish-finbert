import argparse
from pathlib import Path

import pandas as pd

from .console import configure_console
from .paths import ensure_project_dirs


def bist_symbol(ticker: str) -> str:
    ticker = ticker.upper().strip()
    return ticker if ticker.endswith(".IS") else f"{ticker}.IS"


def normalize_price_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    data = data.reset_index()
    normalized_columns: list[str] = []
    known = {"date", "open", "high", "low", "close", "adj close", "volume"}
    for col in data.columns:
        if isinstance(col, tuple):
            parts = [str(part).strip() for part in col if str(part).strip()]
            name = next((part for part in parts if part.lower() in known), parts[0] if parts else "")
        else:
            name = str(col).strip()
        normalized_columns.append(name.lower().replace(" ", "_"))

    data.columns = normalized_columns
    data["ticker"] = ticker.upper().replace(".IS", "")
    return data


def fetch_prices(tickers: list[str], start: str, end: str | None, out_csv: Path) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Fiyat verisi için yfinance kurulu olmalı: uv add yfinance") from exc

    rows = []
    for ticker in tickers:
        yf_symbol = bist_symbol(ticker)
        data = yf.download(yf_symbol, start=start, end=end, auto_adjust=True, progress=False)
        if data.empty:
            print(f"Uyarı: {yf_symbol} için veri bulunamadı.")
            continue
        data = normalize_price_frame(data, yf_symbol)
        keep = [col for col in ["date", "ticker", "open", "high", "low", "close", "volume"] if col in data.columns]
        rows.append(data[keep])

    if not rows:
        raise RuntimeError("Hiç fiyat verisi çekilemedi.")

    prices = pd.concat(rows, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out_csv, index=False)
    return prices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BIST hisseleri için yfinance fiyat verisi çeker.")
    parser.add_argument("--tickers", nargs="+", required=True, help="BIST sembolleri: THYAO ASELS GARAN")
    parser.add_argument("--start", default="2022-01-01", help="Başlangıç tarihi.")
    parser.add_argument("--end", default=None, help="Bitiş tarihi. Boşsa bugüne kadar çeker.")
    parser.add_argument("--out", type=Path, default=Path("data/raw/prices.csv"), help="Fiyat CSV çıktı yolu.")
    return parser


def main() -> None:
    configure_console()
    ensure_project_dirs()
    args = build_parser().parse_args()
    prices = fetch_prices(args.tickers, args.start, args.end, args.out)
    print(f"{len(prices)} fiyat satırı kaydedildi: {args.out}")


if __name__ == "__main__":
    main()
