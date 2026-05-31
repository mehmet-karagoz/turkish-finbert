import argparse
from pathlib import Path

import pandas as pd

from .console import configure_console
from .paths import ensure_project_dirs


def bist_symbol(ticker: str) -> str:
    ticker = ticker.upper().strip()
    return ticker if ticker.endswith(".IS") else f"{ticker}.IS"


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
        data = data.reset_index()
        data["ticker"] = yf_symbol.replace(".IS", "")
        data = data.rename(columns={col: col.lower() for col in data.columns})
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
