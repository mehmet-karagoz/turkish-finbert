import argparse
from argparse import Namespace
from pathlib import Path

import pandas as pd

from .console import configure_console
from .daily_alerts import generate_daily_alerts
from .fetch_news import fetch_kap_api, normalize_tickers, read_ticker_file, rows_from_kap_disclosures, save_news
from .paths import ensure_project_dirs
from .prepare_dataset import prepare_dataset
from .score_news import score_news


RAW_HISTORY_KEYS = ["date", "ticker", "title", "url", "text"]
PREPARED_HISTORY_KEYS = ["date", "ticker", "title", "text"]
SCORED_HISTORY_KEYS = ["date", "ticker", "title", "url", "text"]
DAILY_HISTORY_KEYS = ["date", "ticker"]


def load_tickers(tickers_file: Path, extra_tickers: list[str] | None = None) -> list[str]:
    tickers = read_ticker_file(tickers_file)
    tickers.extend(extra_tickers or [])
    normalized = normalize_tickers(tickers)
    if not normalized:
        raise ValueError(f"Ticker listesi bos: {tickers_file}")
    return normalized


def append_history_csv(
    new_rows: pd.DataFrame,
    history_path: Path,
    dedupe_keys: list[str],
) -> pd.DataFrame:
    existing = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    frames = [frame for frame in [existing, new_rows] if not frame.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=new_rows.columns)
    for col in new_rows.columns:
        if col not in combined:
            combined[col] = pd.NA
    combined = combined[new_rows.columns]
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    keys = [key for key in dedupe_keys if key in combined.columns]
    if keys:
        combined = combined.drop_duplicates(subset=keys, keep="last")
    sort_keys = [key for key in ["date", "ticker", "published_at"] if key in combined.columns]
    if sort_keys:
        combined = combined.sort_values(sort_keys, na_position="last")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(history_path, index=False)
    return combined


def run_daily_pipeline(
    model_path: Path,
    tickers_file: Path,
    raw_out: Path,
    prepared_out: Path,
    scored_out: Path,
    daily_out: Path,
    alerts_dir: Path,
    baseline_daily_sentiment: Path | None = None,
    update_history: bool = False,
    historical_raw_out: Path | None = None,
    historical_prepared_out: Path | None = None,
    historical_scored_out: Path | None = None,
    historical_daily_out: Path | None = None,
    kap_days: int = 7,
    limit: int = 3000,
    report_date: str | None = None,
    top_n: int = 10,
    min_abs_score: float = 0.20,
    baseline_lookback_days: int = 60,
    baseline_min_history: int = 5,
    kap_from_date: str | None = None,
    kap_to_date: str | None = None,
    kap_disclosure_class: str = "ALL",
    append_raw: bool = False,
    extra_tickers: list[str] | None = None,
) -> dict[str, object]:
    tickers = load_tickers(tickers_file, extra_tickers)
    kap_args = Namespace(
        kap_days=kap_days,
        kap_from_date=kap_from_date,
        kap_to_date=kap_to_date,
        kap_disclosure_class=kap_disclosure_class,
    )

    disclosures = fetch_kap_api(kap_args)
    if limit:
        disclosures = disclosures[:limit]
    raw_rows = rows_from_kap_disclosures(disclosures, tickers)
    if raw_rows.empty:
        raise RuntimeError("KAP API'den izlenen ticker'larla eslesen haber gelmedi.")

    saved_raw = save_news(raw_rows, raw_out, append=append_raw)
    prepared = prepare_dataset(raw_out, prepared_out)
    scored, daily = score_news(model_path, prepared_out, scored_out, daily_out)
    historical_counts: dict[str, int] = {}
    effective_baseline = baseline_daily_sentiment
    if update_history:
        historical_raw_out = historical_raw_out or Path("data/raw/kap_api_historical_news.csv")
        historical_prepared_out = historical_prepared_out or Path("data/processed/kap_api_historical_prepared.csv")
        historical_scored_out = historical_scored_out or Path("data/processed/kap_api_historical_scored_news.csv")
        historical_daily_out = historical_daily_out or Path("data/processed/kap_api_historical_daily_sentiment.csv")
        historical_counts = {
            "historical_raw_rows": len(append_history_csv(saved_raw, historical_raw_out, RAW_HISTORY_KEYS)),
            "historical_prepared_rows": len(append_history_csv(prepared, historical_prepared_out, PREPARED_HISTORY_KEYS)),
            "historical_scored_rows": len(append_history_csv(scored, historical_scored_out, SCORED_HISTORY_KEYS)),
            "historical_daily_rows": len(append_history_csv(daily, historical_daily_out, DAILY_HISTORY_KEYS)),
        }
        if effective_baseline is None:
            effective_baseline = historical_daily_out

    alerts = generate_daily_alerts(
        scored_out,
        daily_out,
        alerts_dir,
        baseline_daily_sentiment_csv=effective_baseline,
        date=report_date,
        top_n=top_n,
        min_abs_score=min_abs_score,
        baseline_lookback_days=baseline_lookback_days,
        baseline_min_history=baseline_min_history,
    )

    return {
        "raw_path": raw_out,
        "prepared_path": prepared_out,
        "scored_path": scored_out,
        "daily_path": daily_out,
        "alerts_path": alerts["markdown_path"],
        "brief_path": alerts["brief_path"],
        "report_date": alerts["date"],
        "raw_rows": len(saved_raw),
        "prepared_rows": len(prepared),
        "scored_rows": len(scored),
        "daily_rows": len(daily),
        "baseline_daily_sentiment": effective_baseline,
        "historical_raw_path": historical_raw_out,
        "historical_prepared_path": historical_prepared_out,
        "historical_scored_path": historical_scored_out,
        "historical_daily_path": historical_daily_out,
        **historical_counts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KAP haberlerinden gunluk BIST sentiment raporunu tek komutla uretir.")
    parser.add_argument("--model", type=Path, default=Path("models/master_baseline_sentiment.joblib"), help="Kullanilacak sentiment modeli.")
    parser.add_argument("--tickers-file", type=Path, default=Path("data/raw/tickers_bist_kap.txt"), help="Izlenecek BIST ticker dosyasi.")
    parser.add_argument("--tickers", nargs="*", default=[], help="Dosyaya ek olarak izlenecek ticker'lar.")
    parser.add_argument("--kap-days", type=int, default=7, help="KAP API icin geriye donuk gun sayisi.")
    parser.add_argument("--kap-from-date", default=None, help="KAP baslangic tarihi. Ornek: 2026-06-01")
    parser.add_argument("--kap-to-date", default=None, help="KAP bitis tarihi. Ornek: 2026-06-06")
    parser.add_argument("--kap-disclosure-class", choices=["ALL", "ODA", "FR", "DG", "DUY"], default="ALL", help="KAP bildirim tipi filtresi.")
    parser.add_argument("--limit", type=int, default=3000, help="KAP'tan alinacak maksimum bildirim sayisi. 0 sinirsiz.")
    parser.add_argument("--date", default=None, help="Rapor tarihi. Bos verilirse verideki son tarih kullanilir.")
    parser.add_argument("--top-n", type=int, default=10, help="Ranking listelerinde gosterilecek satir sayisi.")
    parser.add_argument("--min-abs-score", type=float, default=0.20, help="Rapora girmek icin gereken minimum mutlak sentiment skoru.")
    parser.add_argument("--baseline-lookback-days", type=int, default=60, help="Gecmis karsilastirma icin geriye donuk gun sayisi.")
    parser.add_argument("--baseline-min-history", type=int, default=5, help="Anomali yorumu icin gereken minimum gecmis gun sayisi.")
    parser.add_argument("--append-raw", action="store_true", help="Ham KAP dosyasina ekle ve tekrar kayitlari temizle.")
    parser.add_argument("--update-history", action="store_true", help="Gunluk ciktilari historical CSV'lere ekle ve tekrar kayitlari temizle.")
    parser.add_argument("--raw-out", type=Path, default=Path("data/raw/kap_daily.csv"), help="Ham KAP CSV cikti yolu.")
    parser.add_argument("--prepared-out", type=Path, default=Path("data/processed/kap_daily_prepared.csv"), help="Temiz haber CSV cikti yolu.")
    parser.add_argument("--scored-out", type=Path, default=Path("data/processed/kap_daily_scored_news.csv"), help="Satir bazli skor CSV cikti yolu.")
    parser.add_argument("--daily-out", type=Path, default=Path("data/processed/kap_daily_sentiment.csv"), help="Gunluk hisse sentiment CSV cikti yolu.")
    parser.add_argument("--historical-raw-out", type=Path, default=Path("data/raw/kap_api_historical_news.csv"), help="Historical ham KAP CSV yolu.")
    parser.add_argument("--historical-prepared-out", type=Path, default=Path("data/processed/kap_api_historical_prepared.csv"), help="Historical temiz haber CSV yolu.")
    parser.add_argument("--historical-scored-out", type=Path, default=Path("data/processed/kap_api_historical_scored_news.csv"), help="Historical satir bazli skor CSV yolu.")
    parser.add_argument("--historical-daily-out", type=Path, default=Path("data/processed/kap_api_historical_daily_sentiment.csv"), help="Historical gunluk sentiment CSV yolu.")
    parser.add_argument("--baseline-daily-sentiment", type=Path, default=None, help="Gecmis karsilastirma icin ayrica kullanilacak daily sentiment CSV.")
    parser.add_argument("--alerts-dir", type=Path, default=Path("reports/daily_alerts"), help="Gunluk rapor cikti klasoru.")
    return parser


def main() -> None:
    configure_console()
    ensure_project_dirs()
    args = build_parser().parse_args()
    result = run_daily_pipeline(
        model_path=args.model,
        tickers_file=args.tickers_file,
        raw_out=args.raw_out,
        prepared_out=args.prepared_out,
        scored_out=args.scored_out,
        daily_out=args.daily_out,
        alerts_dir=args.alerts_dir,
        baseline_daily_sentiment=args.baseline_daily_sentiment,
        update_history=args.update_history,
        historical_raw_out=args.historical_raw_out,
        historical_prepared_out=args.historical_prepared_out,
        historical_scored_out=args.historical_scored_out,
        historical_daily_out=args.historical_daily_out,
        kap_days=args.kap_days,
        limit=args.limit,
        report_date=args.date,
        top_n=args.top_n,
        min_abs_score=args.min_abs_score,
        baseline_lookback_days=args.baseline_lookback_days,
        baseline_min_history=args.baseline_min_history,
        kap_from_date=args.kap_from_date,
        kap_to_date=args.kap_to_date,
        kap_disclosure_class=args.kap_disclosure_class,
        append_raw=args.append_raw,
        extra_tickers=args.tickers,
    )
    print(f"Ham haber: {result['raw_rows']} satir -> {result['raw_path']}")
    print(f"Temiz haber: {result['prepared_rows']} satir -> {result['prepared_path']}")
    print(f"Skorlanan haber: {result['scored_rows']} satir -> {result['scored_path']}")
    print(f"Gunluk sentiment: {result['daily_rows']} satir -> {result['daily_path']}")
    if args.update_history:
        print(f"Historical ham haber: {result['historical_raw_rows']} satir -> {result['historical_raw_path']}")
        print(f"Historical temiz haber: {result['historical_prepared_rows']} satir -> {result['historical_prepared_path']}")
        print(f"Historical skorlanan haber: {result['historical_scored_rows']} satir -> {result['historical_scored_path']}")
        print(f"Historical gunluk sentiment: {result['historical_daily_rows']} satir -> {result['historical_daily_path']}")
    if result["baseline_daily_sentiment"]:
        print(f"Baseline daily sentiment: {result['baseline_daily_sentiment']}")
    print(f"Rapor tarihi: {result['report_date']}")
    print(f"Kisa ozet: {result['brief_path']}")
    print(f"Gunluk rapor: {result['alerts_path']}")


if __name__ == "__main__":
    main()
