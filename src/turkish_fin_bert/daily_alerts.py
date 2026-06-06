import argparse
import os
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from .console import configure_console
from .paths import ensure_project_dirs
from .signal_quality import build_event_signals


MARKET_KEYWORDS = [
    "bist 100",
    "bist100",
    "xu100",
    "borsa istanbul",
    "tum borsa",
    "tüm borsa",
    "endeks",
    "tcmb",
    "faiz karari",
    "faiz kararı",
    "politika faizi",
    "enflasyon",
    "doviz kuru",
    "döviz kuru",
    "brsa",
    "resmi gazete",
    "kredi derecelendirme",
    "jeopolitik",
]


def normalize_date(value: str | None, dates: pd.Series) -> pd.Timestamp:
    parsed_dates = pd.to_datetime(dates, errors="coerce").dropna()
    if parsed_dates.empty:
        raise ValueError("Gecerli tarih bulunamadi.")
    if value:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"Gecersiz tarih: {value}")
        return pd.Timestamp(parsed).normalize()
    return pd.Timestamp(parsed_dates.max()).normalize()


def prepare_scored_news(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"date", "ticker", "title", "text", "prediction", "sentiment_score", "prob_negative", "prob_positive"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Scored news CSV icinde eksik kolonlar: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["ticker"] = df["ticker"].fillna("").astype(str).str.upper().str.replace(".IS", "", regex=False).str.strip()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["url"] = df["url"].fillna("").astype(str).str.strip() if "url" in df else ""
    df["confidence"] = df[["prob_negative", "prob_positive"]].max(axis=1)
    df["impact_score"] = df["prob_positive"] - df["prob_negative"]
    return df.dropna(subset=["date"])


def prepare_daily_sentiment(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"date", "ticker", "sentiment_score", "prob_negative", "prob_positive", "news_count"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Daily sentiment CSV icinde eksik kolonlar: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["ticker"] = df["ticker"].fillna("").astype(str).str.upper().str.replace(".IS", "", regex=False).str.strip()
    return df.dropna(subset=["date", "ticker"]).loc[lambda data: data["ticker"].ne("")]


def is_market_wide_news(row: pd.Series) -> bool:
    if not str(row.get("ticker", "")).strip():
        return True
    haystack = f"{row.get('title', '')} {row.get('text', '')}".lower()
    return any(keyword in haystack for keyword in MARKET_KEYWORDS)


def _format_stock_rows(rows: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for _, row in rows.iterrows():
        lines.append(f"- {row['ticker']}: {row['sentiment_score']:+.3f} ({int(row['news_count'])} haber)")
    return lines or ["- Veri yok"]


def _format_news_rows(rows: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for _, row in rows.iterrows():
        ticker = row["ticker"] or "GENEL"
        title = row["title"] or row["text"][:100]
        lines.append(f"- {ticker}: {row['impact_score']:+.3f} | {title}")
    return lines or ["- Veri yok"]


def _format_event_rows(rows: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for _, row in rows.iterrows():
        title = row["title"] or "Baslik yok"
        lines.append(
            "- "
            f"{row['tickers']}: {row['avg_impact_score']:+.3f}, "
            f"onem {row['materiality_score']:.2f}, "
            f"{row['signal_strength']} | {row['event_type']} | {title}"
        )
    return lines or ["- Veri yok"]


def build_markdown_report(
    report_date: pd.Timestamp,
    event_signals: pd.DataFrame,
    top_stocks: pd.DataFrame,
    bottom_stocks: pd.DataFrame,
    positive_news: pd.DataFrame,
    negative_news: pd.DataFrame,
    market_news: pd.DataFrame,
    daily_rows: pd.DataFrame,
    min_abs_score: float = 0.20,
) -> str:
    avg_score = float(daily_rows["sentiment_score"].mean()) if not daily_rows.empty else 0.0
    total_news = int(daily_rows["news_count"].sum()) if not daily_rows.empty else 0
    direction = "pozitif" if avg_score > 0.05 else "negatif" if avg_score < -0.05 else "notr"
    date_text = report_date.strftime("%Y-%m-%d")
    signal_count = (
        len(event_signals)
        + len(top_stocks)
        + len(bottom_stocks)
        + len(positive_news)
        + len(negative_news)
        + len(market_news)
    )
    strong_event_count = (
        int(event_signals["signal_strength"].isin(["medium", "strong"]).sum()) if not event_signals.empty else 0
    )
    if not signal_count:
        summary = "Bugun esigi asan anlamli haber/hisse sinyali yok."
    elif strong_event_count:
        summary = "Bugun orta/guclu olay sinyali var."
    else:
        summary = "Bugun esigi asan ancak zayif sinyaller var."

    lines = [
        f"# BIST Gunluk Sentiment Raporu - {date_text}",
        "",
        f"Genel skor: {avg_score:+.3f} ({direction}), haber adedi: {total_news}",
        f"Sinyal esigi: |skor| >= {min_abs_score:.2f}",
        summary,
        "",
        "## Onemli Olay Ozeti",
        *_format_event_rows(event_signals),
        "",
        "## En Iyi Hisseler",
        *_format_stock_rows(top_stocks),
        "",
        "## En Kotu Hisseler",
        *_format_stock_rows(bottom_stocks),
        "",
        "## En Guclu Pozitif Haberler",
        *_format_news_rows(positive_news),
        "",
        "## En Guclu Negatif Haberler",
        *_format_news_rows(negative_news),
        "",
        "## Piyasa Geneli Aday Haberler",
        *_format_news_rows(market_news),
    ]
    return "\n".join(lines) + "\n"


def generate_daily_alerts(
    scored_news_csv: Path,
    daily_sentiment_csv: Path,
    out_dir: Path,
    date: str | None = None,
    top_n: int = 10,
    min_abs_score: float = 0.20,
) -> dict[str, pd.DataFrame | Path | str]:
    scored = prepare_scored_news(scored_news_csv)
    daily = prepare_daily_sentiment(daily_sentiment_csv)
    report_date = normalize_date(date, pd.concat([scored["date"], daily["date"]], ignore_index=True))

    news_day = scored[scored["date"].eq(report_date)].copy()
    daily_day = daily[daily["date"].eq(report_date)].copy()
    if news_day.empty and daily_day.empty:
        raise RuntimeError(f"{report_date.date()} icin haber/sentiment verisi yok.")

    top_stocks = (
        daily_day[daily_day["sentiment_score"].ge(min_abs_score)]
        .sort_values(["sentiment_score", "news_count"], ascending=[False, False])
        .head(top_n)
    )
    bottom_stocks = (
        daily_day[daily_day["sentiment_score"].le(-min_abs_score)]
        .sort_values(["sentiment_score", "news_count"], ascending=[True, False])
        .head(top_n)
    )
    positive_news = news_day[news_day["impact_score"].ge(min_abs_score)].sort_values("impact_score", ascending=False).head(top_n)
    negative_news = news_day[news_day["impact_score"].le(-min_abs_score)].sort_values("impact_score", ascending=True).head(top_n)
    market_news = news_day[news_day.apply(is_market_wide_news, axis=1)].copy()
    market_news = market_news[market_news["impact_score"].abs().ge(min_abs_score)]
    market_news = market_news.reindex(market_news["impact_score"].abs().sort_values(ascending=False).index).head(top_n)
    event_signals = build_event_signals(news_day, min_abs_score=min_abs_score, top_n=top_n)

    out_dir.mkdir(parents=True, exist_ok=True)
    date_slug = report_date.strftime("%Y-%m-%d")
    paths = {
        "event_signals": out_dir / f"{date_slug}_event_signals.csv",
        "top_stocks": out_dir / f"{date_slug}_top_stocks.csv",
        "bottom_stocks": out_dir / f"{date_slug}_bottom_stocks.csv",
        "positive_news": out_dir / f"{date_slug}_positive_news.csv",
        "negative_news": out_dir / f"{date_slug}_negative_news.csv",
        "market_news": out_dir / f"{date_slug}_market_news.csv",
        "markdown": out_dir / f"{date_slug}_daily_alerts.md",
    }
    event_signals.to_csv(paths["event_signals"], index=False)
    top_stocks.to_csv(paths["top_stocks"], index=False)
    bottom_stocks.to_csv(paths["bottom_stocks"], index=False)
    positive_news.to_csv(paths["positive_news"], index=False)
    negative_news.to_csv(paths["negative_news"], index=False)
    market_news.to_csv(paths["market_news"], index=False)

    markdown = build_markdown_report(
        report_date,
        event_signals,
        top_stocks,
        bottom_stocks,
        positive_news,
        negative_news,
        market_news,
        daily_day,
        min_abs_score=min_abs_score,
    )
    paths["markdown"].write_text(markdown, encoding="utf-8")

    return {
        "date": date_slug,
        "markdown_text": markdown,
        "markdown_path": paths["markdown"],
        "event_signals": event_signals,
        "top_stocks": top_stocks,
        "bottom_stocks": bottom_stocks,
        "positive_news": positive_news,
        "negative_news": negative_news,
        "market_news": market_news,
    }


def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": text[:3900]}).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        if response.status >= 400:
            raise RuntimeError(f"Telegram gonderimi basarisiz: HTTP {response.status}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gunluk BIST sentiment ranking ve haber alarm raporu uretir.")
    parser.add_argument("--scored-news", type=Path, required=True, help="score_news satir bazli cikti CSV.")
    parser.add_argument("--daily-sentiment", type=Path, required=True, help="score_news gunluk sentiment cikti CSV.")
    parser.add_argument("--date", help="Rapor tarihi. Bos verilirse verideki son tarih kullanilir.")
    parser.add_argument("--top-n", type=int, default=10, help="Listelerde gosterilecek satir sayisi.")
    parser.add_argument("--min-abs-score", type=float, default=0.20, help="Rapora girmek icin gereken minimum mutlak sentiment skoru.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/daily_alerts"), help="Rapor cikti klasoru.")
    parser.add_argument("--send-telegram", action="store_true", help="Raporu Telegram'a gonderir.")
    parser.add_argument("--telegram-token-env", default="TELEGRAM_BOT_TOKEN", help="Bot token env var adi.")
    parser.add_argument("--telegram-chat-id-env", default="TELEGRAM_CHAT_ID", help="Chat id env var adi.")
    return parser


def main() -> None:
    configure_console()
    ensure_project_dirs()
    args = build_parser().parse_args()
    result = generate_daily_alerts(
        args.scored_news,
        args.daily_sentiment,
        args.out_dir,
        date=args.date,
        top_n=args.top_n,
        min_abs_score=args.min_abs_score,
    )
    print(f"Gunluk rapor kaydedildi: {result['markdown_path']}")
    if args.send_telegram:
        token = os.getenv(args.telegram_token_env, "")
        chat_id = os.getenv(args.telegram_chat_id_env, "")
        if not token or not chat_id:
            raise RuntimeError(f"Telegram icin {args.telegram_token_env} ve {args.telegram_chat_id_env} env var gerekli.")
        send_telegram_message(token, chat_id, str(result["markdown_text"]))
        print("Telegram mesaji gonderildi.")


if __name__ == "__main__":
    main()
