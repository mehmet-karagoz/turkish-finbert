import argparse
from pathlib import Path

import pandas as pd

from .console import configure_console
from .financial_effect import load_prices
from .paths import FIGURE_DIR, ensure_project_dirs
from .visualize import plot_drawdown, plot_equity_curve


def _rebalance_dates(dates: pd.Series, months: int) -> list[pd.Timestamp]:
    unique_dates = sorted(pd.to_datetime(dates.dropna().unique()))
    if not unique_dates:
        return []
    result = [unique_dates[0]]
    next_date = unique_dates[0] + pd.DateOffset(months=months)
    for date in unique_dates[1:]:
        if date >= next_date:
            result.append(date)
            next_date = date + pd.DateOffset(months=months)
    return result


def run_backtest(sentiment_csv: Path, prices_csv: Path, top_n: int, rebalance_months: int, out_csv: Path) -> pd.DataFrame:
    sentiment = pd.read_csv(sentiment_csv)
    sentiment["date"] = pd.to_datetime(sentiment["date"], errors="coerce")
    sentiment["ticker"] = sentiment["ticker"].astype(str).str.upper().str.replace(".IS", "", regex=False).str.strip()
    sentiment = sentiment.dropna(subset=["date", "ticker", "sentiment_score"]).sort_values(["ticker", "date"])

    prices = load_prices(prices_csv)
    prices["return"] = prices.groupby("ticker")["close"].pct_change().fillna(0.0)
    wide_returns = prices.pivot(index="date", columns="ticker", values="return").sort_index()

    rebalances = _rebalance_dates(wide_returns.index.to_series(), rebalance_months)
    if len(rebalances) < 2:
        raise RuntimeError("Backtest için en az iki rebalance tarihi gerekir.")

    daily_rows = []
    holdings: list[str] = []

    for date, returns in wide_returns.iterrows():
        available = [ticker for ticker in holdings if ticker in returns.index and pd.notna(returns[ticker])]
        strategy_return = float(returns[available].mean()) if available else 0.0
        benchmark_return = float(returns.dropna().mean()) if not returns.dropna().empty else 0.0
        daily_rows.append(
            {
                "date": date,
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "holdings": ",".join(available),
            }
        )

        # Aynı günün kapanış getirisine bakış sızıntısı olmasın diye yeni portföy sonraki güne uygulanır.
        if date in rebalances:
            history = sentiment[sentiment["date"] < date]
            latest = history.sort_values("date").groupby("ticker", as_index=False).tail(1)
            holdings = latest.sort_values("sentiment_score", ascending=False).head(top_n)["ticker"].tolist()

    result = pd.DataFrame(daily_rows)
    result["strategy_equity"] = (1 + result["strategy_return"]).cumprod()
    result["benchmark_equity"] = (1 + result["benchmark_return"]).cumprod()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    plot_equity_curve(result, FIGURE_DIR / "backtest_equity.png")
    plot_drawdown(result, FIGURE_DIR / "backtest_drawdown.png")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sentiment skoruyla basit eşit ağırlıklı portföy backtesti yapar.")
    parser.add_argument("--sentiment", type=Path, required=True, help="Günlük sentiment CSV.")
    parser.add_argument("--prices", type=Path, required=True, help="Fiyat CSV.")
    parser.add_argument("--top-n", type=int, default=5, help="Her rebalance döneminde seçilecek hisse sayısı.")
    parser.add_argument("--rebalance-months", type=int, default=3, help="Yeniden dengeleme periyodu.")
    parser.add_argument("--out", type=Path, default=Path("reports/backtest_equity.csv"), help="Backtest çıktı CSV.")
    return parser


def main() -> None:
    configure_console()
    ensure_project_dirs()
    args = build_parser().parse_args()
    result = run_backtest(args.sentiment, args.prices, args.top_n, args.rebalance_months, args.out)
    last = result.iloc[-1]
    print(f"Backtest kaydedildi: {args.out}")
    print(f"Strateji={last['strategy_equity']:.3f} Benchmark={last['benchmark_equity']:.3f}")


if __name__ == "__main__":
    main()
