import argparse
from argparse import Namespace
from pathlib import Path

from .console import configure_console
from .daily_alerts import generate_daily_alerts
from .fetch_news import fetch_kap_api, normalize_tickers, read_ticker_file, rows_from_kap_disclosures, save_news
from .paths import ensure_project_dirs
from .prepare_dataset import prepare_dataset
from .score_news import score_news


def load_tickers(tickers_file: Path, extra_tickers: list[str] | None = None) -> list[str]:
    tickers = read_ticker_file(tickers_file)
    tickers.extend(extra_tickers or [])
    normalized = normalize_tickers(tickers)
    if not normalized:
        raise ValueError(f"Ticker listesi bos: {tickers_file}")
    return normalized


def run_daily_pipeline(
    model_path: Path,
    tickers_file: Path,
    raw_out: Path,
    prepared_out: Path,
    scored_out: Path,
    daily_out: Path,
    alerts_dir: Path,
    kap_days: int = 7,
    limit: int = 3000,
    report_date: str | None = None,
    top_n: int = 10,
    min_abs_score: float = 0.20,
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
    alerts = generate_daily_alerts(scored_out, daily_out, alerts_dir, date=report_date, top_n=top_n, min_abs_score=min_abs_score)

    return {
        "raw_path": raw_out,
        "prepared_path": prepared_out,
        "scored_path": scored_out,
        "daily_path": daily_out,
        "alerts_path": alerts["markdown_path"],
        "report_date": alerts["date"],
        "raw_rows": len(saved_raw),
        "prepared_rows": len(prepared),
        "scored_rows": len(scored),
        "daily_rows": len(daily),
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
    parser.add_argument("--append-raw", action="store_true", help="Ham KAP dosyasina ekle ve tekrar kayitlari temizle.")
    parser.add_argument("--raw-out", type=Path, default=Path("data/raw/kap_daily.csv"), help="Ham KAP CSV cikti yolu.")
    parser.add_argument("--prepared-out", type=Path, default=Path("data/processed/kap_daily_prepared.csv"), help="Temiz haber CSV cikti yolu.")
    parser.add_argument("--scored-out", type=Path, default=Path("data/processed/kap_daily_scored_news.csv"), help="Satir bazli skor CSV cikti yolu.")
    parser.add_argument("--daily-out", type=Path, default=Path("data/processed/kap_daily_sentiment.csv"), help="Gunluk hisse sentiment CSV cikti yolu.")
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
        kap_days=args.kap_days,
        limit=args.limit,
        report_date=args.date,
        top_n=args.top_n,
        min_abs_score=args.min_abs_score,
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
    print(f"Rapor tarihi: {result['report_date']}")
    print(f"Gunluk rapor: {result['alerts_path']}")


if __name__ == "__main__":
    main()
