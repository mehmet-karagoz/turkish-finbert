import argparse
import html
import os
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from .console import configure_console
from .paths import ensure_project_dirs
from .signal_baseline import build_signal_baseline, select_notable_anomalies
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

EVENT_TYPE_LABELS = {
    "capital_action": "sermaye/pay islemi",
    "contract_order": "sozlesme/is iliskisi",
    "debt_credit": "borc/kredi",
    "financial_result": "finansal sonuc",
    "legal_regulatory": "hukuki/duzenleyici olay",
    "management": "yonetim/organizasyon",
    "market": "piyasa geneli",
    "other": "olay tipi net degil",
    "routine": "rutin bildirim",
}

STRENGTH_LABELS = {
    "none": "sinyal yok",
    "weak": "zayif",
    "medium": "orta",
    "strong": "guclu",
}

SCOPE_LABELS = {
    "market_wide": "piyasa geneli",
    "multi_stock": "birden fazla hisse",
    "stock_specific": "hisse bazli",
}

ANOMALY_LABELS = {
    "insufficient_history": "gecmis veri az",
    "normal": "gecmise gore olagan",
    "elevated": "gecmise gore dikkat cekici",
    "unusual": "gecmise gore sira disi",
}

ANOMALY_RANK = {
    "insufficient_history": 1,
    "normal": 0,
    "elevated": 2,
    "unusual": 3,
}

ANOMALY_PRIORITY = {
    "insufficient_history": 0.25,
    "normal": 0.05,
    "elevated": 0.70,
    "unusual": 1.00,
}

STRENGTH_PRIORITY = {
    "none": 0.00,
    "weak": 0.25,
    "medium": 0.65,
    "strong": 1.00,
}

ACTION_PRIORITY = {
    "takip et": 1.00,
    "piyasa geneli dikkat": 0.90,
    "detay kontrol et": 0.75,
    "gecmis veri yetersiz": 0.35,
    "zayif sinyal": 0.20,
    "rutin / ignore": 0.00,
}

EVENT_TYPE_REASONS = {
    "capital_action": "sermaye/pay islemi fiyatlama etkisi yaratabilir",
    "contract_order": "gelir beklentisini etkileyebilecek is iliskisi",
    "debt_credit": "finansman kosullarini etkileyebilecek borc/kredi haberi",
    "financial_result": "dogrudan finansal performans bilgisi iceriyor",
    "legal_regulatory": "hukuki veya duzenleyici belirsizlik yaratabilir",
    "management": "yonetim/organizasyon haberi; finansal etki dolayli olabilir",
    "market": "tek hisse degil, piyasa geneli etkileyebilecek baslik",
    "other": "olay tipi net degil; haber detayi kontrol edilmeli",
    "routine": "rutin bildirim; tek basina guclu sinyal sayilmamali",
}

EVENT_TYPE_SHORT_REASONS = {
    "capital_action": "sermaye/pay islemi",
    "contract_order": "is iliskisi",
    "debt_credit": "borc/kredi",
    "financial_result": "finansal sonuc",
    "legal_regulatory": "hukuki/duzenleyici",
    "management": "yonetim haberi",
    "market": "piyasa geneli baslik",
    "other": "olay tipi net degil",
    "routine": "rutin bildirim",
}

ACTION_ORDER = [
    "takip et",
    "piyasa geneli dikkat",
    "detay kontrol et",
    "gecmis veri yetersiz",
    "zayif sinyal",
    "rutin / ignore",
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


def build_baseline_daily_frame(
    current_daily: pd.DataFrame,
    report_date: pd.Timestamp,
    baseline_daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if baseline_daily is None or baseline_daily.empty:
        return current_daily.copy()
    history = baseline_daily[baseline_daily["date"].lt(report_date)].copy()
    combined = pd.concat([history, current_daily], ignore_index=True)
    return combined.drop_duplicates(subset=["date", "ticker"], keep="last")


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
            f"oncelik {int(row.get('priority_score', 0))}/100, "
            f"onem {row['materiality_score']:.2f}, "
            f"{row['signal_strength']} | {row['event_type']} | {title}"
        )
    return lines or ["- Veri yok"]


def _signal_summary(event_signals: pd.DataFrame, signal_count: int) -> str:
    strong_event_count = (
        int(event_signals["signal_strength"].isin(["medium", "strong"]).sum()) if not event_signals.empty else 0
    )
    if not signal_count:
        return "Bugun esigi asan anlamli haber/hisse sinyali yok."
    if strong_event_count:
        return "Bugun orta/guclu olay sinyali var."
    return "Bugun esigi asan ancak zayif sinyaller var."


def _max_priority(event_signals: pd.DataFrame) -> int:
    if event_signals.empty or "priority_score" not in event_signals:
        return 0
    return int(event_signals["priority_score"].max())


def _priority_level(event_signals: pd.DataFrame) -> str:
    max_priority = _max_priority(event_signals)
    if max_priority >= 70:
        return f"yuksek ({max_priority}/100)"
    if max_priority >= 50:
        return f"orta ({max_priority}/100)"
    if max_priority >= 35:
        return f"dusuk ({max_priority}/100)"
    if max_priority > 0:
        return f"cok dusuk ({max_priority}/100)"
    return "yok"


def _brief_decision(event_signals: pd.DataFrame, signal_count: int) -> str:
    if not signal_count:
        return "Bugun aksiyon gerektiren olay yok."
    if event_signals.empty:
        return "Hisse bazli skor var ama olay kalitesi dusuk; aksiyon yok."
    max_priority = _max_priority(event_signals)
    if event_signals["signal_strength"].eq("strong").any() or max_priority >= 70:
        return "Bugun yuksek oncelikli takip gerektiren olay var."
    if event_signals["signal_strength"].eq("medium").any() or max_priority >= 50:
        return "Bugun orta oncelikli takip gerektiren olay var."
    if max_priority >= 35:
        return "Bugun dusuk oncelikli manuel kontrol sinyali var."
    return "Bugun oncelikli olay yok; sinyaller zayif."


def _brief_note(event_signals: pd.DataFrame, market_news: pd.DataFrame) -> str:
    if event_signals.empty and market_news.empty:
        return "Esigi asan haber akisi tespit edilmedi."
    if event_signals.empty:
        return "Haber skoru esigi assa da olay bazli guc dusuk."

    medium_strong_count = int(event_signals["signal_strength"].isin(["medium", "strong"]).sum())
    weak_count = int(event_signals["signal_strength"].eq("weak").sum())
    if medium_strong_count:
        return f"{medium_strong_count} orta/guclu olay sinyali var; haber detaylari kontrol edilmeli."
    return f"{weak_count} zayif hisse bazli sinyal var; tek basina islem karari icin yeterli guc yok."


def _flow_summary(
    daily_rows: pd.DataFrame,
    event_signals: pd.DataFrame,
    top_stocks: pd.DataFrame,
    bottom_stocks: pd.DataFrame,
    market_news: pd.DataFrame,
) -> str:
    total_news = int(daily_rows["news_count"].sum()) if not daily_rows.empty else 0
    ticker_count = int(daily_rows["ticker"].nunique()) if not daily_rows.empty else 0
    threshold_ticker_count = len(top_stocks) + len(bottom_stocks)
    return (
        f"{total_news} haber, {ticker_count} hisse, "
        f"{threshold_ticker_count} esik ustu hisse, "
        f"{len(event_signals)} olay, {len(market_news)} piyasa geneli aday"
    )


def _daily_result(event_signals: pd.DataFrame, signal_count: int) -> str:
    if not signal_count:
        return "Onemli olay yok; takip listesi bos."
    if event_signals.empty:
        return "Skor var ama olay bazli anlam dusuk; rutin takip yeterli."
    max_priority = _max_priority(event_signals)
    if max_priority >= 70:
        return "Yuksek oncelikli olay var; detay kontrol edilmeli."
    if max_priority >= 50:
        return "Orta oncelikli olay var; haber detayi kontrol edilmeli."
    if max_priority >= 35:
        return "Zayif sinyaller var; manuel kontrol disinda aksiyon gerektirmiyor."
    return "Oncelikli olay yok; sinyaller zayif."


def _signal_count(
    event_signals: pd.DataFrame,
    top_stocks: pd.DataFrame,
    bottom_stocks: pd.DataFrame,
    positive_news: pd.DataFrame,
    negative_news: pd.DataFrame,
    market_news: pd.DataFrame,
) -> int:
    return (
        len(event_signals)
        + len(top_stocks)
        + len(bottom_stocks)
        + len(positive_news)
        + len(negative_news)
        + len(market_news)
    )


def _event_type_label(event_type: object) -> str:
    return EVENT_TYPE_LABELS.get(str(event_type), str(event_type) or "olay tipi net degil")


def _strength_label(strength: object) -> str:
    return STRENGTH_LABELS.get(str(strength), str(strength) or "sinyal yok")


def _scope_label(scope: object) -> str:
    return SCOPE_LABELS.get(str(scope), str(scope) or "kapsam net degil")


def _direction_label(score: float) -> str:
    if score > 0:
        return "pozitif"
    if score < 0:
        return "negatif"
    return "notr"


def _anomaly_level_label(level: object) -> str:
    return ANOMALY_LABELS.get(str(level), str(level) or "gecmis karsilastirma yok")


def _event_type_reason(event_type: object) -> str:
    return EVENT_TYPE_REASONS.get(str(event_type), "haber basligi manuel kontrol gerektiriyor")


def _event_type_short_reason(event_type: object) -> str:
    return EVENT_TYPE_SHORT_REASONS.get(str(event_type), "manuel kontrol")


def _ticker_anomaly_rows(tickers: object, signal_baseline: pd.DataFrame | None) -> pd.DataFrame:
    if signal_baseline is None or signal_baseline.empty:
        return pd.DataFrame()
    ticker_list = [ticker.strip() for ticker in str(tickers).split(",") if ticker.strip()]
    if not ticker_list:
        return pd.DataFrame()
    rows = signal_baseline[signal_baseline["ticker"].isin(ticker_list)].copy()
    if rows.empty:
        return rows
    rows["anomaly_rank"] = rows["anomaly_level"].map(ANOMALY_RANK).fillna(0)
    return rows.sort_values(["anomaly_rank", "sentiment_score"], ascending=[False, False])


def _best_anomaly_level(tickers: object, signal_baseline: pd.DataFrame | None) -> str:
    rows = _ticker_anomaly_rows(tickers, signal_baseline)
    if rows.empty:
        return ""
    return str(rows.iloc[0]["anomaly_level"])


def _ticker_anomaly_text(tickers: object, signal_baseline: pd.DataFrame | None) -> str:
    rows = _ticker_anomaly_rows(tickers, signal_baseline)
    if rows.empty:
        return ""
    parts = []
    for _, row in rows.head(3).iterrows():
        parts.append(f"{row['ticker']} {_anomaly_level_label(row['anomaly_level'])}")
    return " Gecmis karsilastirma: " + "; ".join(parts) + "."


def _ticker_anomaly_compact_text(tickers: object, signal_baseline: pd.DataFrame | None) -> str:
    rows = _ticker_anomaly_rows(tickers, signal_baseline)
    if rows.empty:
        return ""
    parts = []
    for _, row in rows.head(2).iterrows():
        parts.append(f"{row['ticker']} {_anomaly_level_label(row['anomaly_level'])}")
    return ", ".join(parts)


def _baseline_summary(signal_baseline: pd.DataFrame | None, min_abs_score: float) -> str:
    if signal_baseline is None or signal_baseline.empty:
        return "Yeterli gecmis veri yok."
    signal_rows = signal_baseline[signal_baseline["sentiment_score"].abs().ge(min_abs_score)]
    if signal_rows.empty:
        return "Skor esigini asan hisse yok."
    unusual_count = int(signal_rows["anomaly_level"].eq("unusual").sum())
    elevated_count = int(signal_rows["anomaly_level"].eq("elevated").sum())
    insufficient_count = int(signal_rows["anomaly_level"].eq("insufficient_history").sum())
    if unusual_count or elevated_count:
        return f"{unusual_count} sira disi, {elevated_count} dikkat cekici hisse sinyali var."
    if insufficient_count == len(signal_rows):
        return "Esigi asan hisselerde yeterli gecmis veri yok."
    return "Bugunku esik ustu sinyaller gecmise gore olagan."


def _format_anomaly_rows(rows: pd.DataFrame) -> list[str]:
    if rows.empty:
        return ["- Veri yok"]
    lines: list[str] = []
    for _, row in rows.iterrows():
        z_text = "z yok" if pd.isna(row["anomaly_z"]) else f"z={float(row['anomaly_z']):+.2f}"
        lines.append(
            "- "
            f"{row['ticker']}: {float(row['sentiment_score']):+.3f}, "
            f"{_anomaly_level_label(row['anomaly_level'])}, "
            f"{z_text}, gecmis gun={int(row['history_days'])}"
        )
    return lines


def _event_explanation(row: pd.Series) -> str:
    event_type = str(row["event_type"])
    strength = str(row["signal_strength"])
    scope = _scope_label(row["market_scope"]).capitalize()
    if strength == "weak":
        return f"{scope}; skor esigi asti ama onem skoru sinirli."
    if event_type == "other":
        return f"{scope}; olay tipi net olmadigi icin haber detayi kontrol edilmeli."
    return f"{scope}; {_event_type_label(event_type)} basligi nedeniyle takip edilmeli."


def _event_action(row: pd.Series, signal_baseline: pd.DataFrame | None = None) -> str:
    event_type = str(row["event_type"])
    strength = str(row["signal_strength"])
    scope = str(row["market_scope"])
    anomaly_level = _best_anomaly_level(row["tickers"], signal_baseline)

    if event_type == "routine":
        return "rutin / ignore"
    if scope == "market_wide" and strength in {"medium", "strong"}:
        return "piyasa geneli dikkat"
    if strength == "strong":
        return "takip et"
    if strength == "medium" or anomaly_level in {"unusual", "elevated"}:
        return "detay kontrol et"
    if anomaly_level == "insufficient_history":
        return "gecmis veri yetersiz"
    return "zayif sinyal"


def _priority_reason(row: pd.Series, signal_baseline: pd.DataFrame | None = None) -> str:
    action = _event_action(row, signal_baseline)
    anomaly_level = _best_anomaly_level(row["tickers"], signal_baseline)
    parts = [
        f"aksiyon={action}",
        f"guc={_strength_label(row['signal_strength'])}",
        f"onem={float(row['materiality_score']):.2f}",
    ]
    if anomaly_level:
        parts.append(f"gecmis={_anomaly_level_label(anomaly_level)}")
    return "; ".join(parts)


def _priority_score(row: pd.Series, signal_baseline: pd.DataFrame | None = None) -> int:
    materiality = min(1.0, max(0.0, float(row["materiality_score"])))
    impact = min(1.0, abs(float(row["avg_impact_score"])))
    strength = STRENGTH_PRIORITY.get(str(row["signal_strength"]), 0.0)
    anomaly = ANOMALY_PRIORITY.get(_best_anomaly_level(row["tickers"], signal_baseline), 0.0)
    action = ACTION_PRIORITY.get(_event_action(row, signal_baseline), 0.0)
    score = materiality * 40 + impact * 25 + strength * 15 + anomaly * 12 + action * 8
    return int(round(min(100.0, max(0.0, score))))


def add_event_priority(event_signals: pd.DataFrame, signal_baseline: pd.DataFrame | None = None) -> pd.DataFrame:
    events = event_signals.copy()
    if events.empty:
        events["priority_score"] = pd.Series(dtype="int64")
        events["priority_reason"] = pd.Series(dtype="object")
        return events
    events["priority_score"] = events.apply(_priority_score, axis=1, signal_baseline=signal_baseline)
    events["priority_reason"] = events.apply(_priority_reason, axis=1, signal_baseline=signal_baseline)
    return events.sort_values(
        ["priority_score", "materiality_score", "avg_impact_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _action_counts(event_signals: pd.DataFrame, signal_baseline: pd.DataFrame | None = None) -> dict[str, int]:
    counts = {action: 0 for action in ACTION_ORDER}
    if event_signals.empty:
        return counts
    for _, row in event_signals.iterrows():
        action = _event_action(row, signal_baseline)
        counts[action] = counts.get(action, 0) + 1
    return counts


def _action_summary(event_signals: pd.DataFrame, signal_baseline: pd.DataFrame | None = None) -> str:
    counts = _action_counts(event_signals, signal_baseline)
    nonzero = [f"{count} {action}" for action, count in counts.items() if count]
    return ", ".join(nonzero) if nonzero else "aksiyon gerektiren olay yok"


def _event_reason(row: pd.Series, signal_baseline: pd.DataFrame | None = None) -> str:
    event_type = str(row["event_type"])
    strength = str(row["signal_strength"])
    materiality = float(row["materiality_score"])
    reason_parts = [_event_type_reason(event_type)]

    if strength == "weak":
        reason_parts.append(f"olay gucu zayif, onem skoru {materiality:.2f}")
    elif strength == "medium":
        reason_parts.append(f"olay gucu orta, onem skoru {materiality:.2f}")
    elif strength == "strong":
        reason_parts.append(f"olay gucu yuksek, onem skoru {materiality:.2f}")

    anomaly_level = _best_anomaly_level(row["tickers"], signal_baseline)
    if anomaly_level:
        reason_parts.append(_anomaly_level_label(anomaly_level))
    return "; ".join(reason_parts) + "."


def _brief_event_reason(row: pd.Series, signal_baseline: pd.DataFrame | None = None) -> str:
    parts = [_event_type_short_reason(row["event_type"])]
    anomaly_text = _ticker_anomaly_compact_text(row["tickers"], signal_baseline)
    if anomaly_text:
        parts.append(anomaly_text)
    elif str(row["signal_strength"]) == "weak":
        parts.append("zayif sinyal")
    return "; ".join(parts)


def _format_action_rows(
    event_signals: pd.DataFrame,
    signal_baseline: pd.DataFrame | None = None,
    max_rows: int = 10,
) -> list[str]:
    if event_signals.empty:
        return ["- Veri yok"]
    lines: list[str] = []
    for _, row in event_signals.head(max_rows).iterrows():
        score = float(row["avg_impact_score"])
        lines.append(
            f"- {row['tickers']}: {_event_action(row, signal_baseline)} | "
            f"oncelik {int(row.get('priority_score', 0))}/100 | "
            f"{_direction_label(score)} {score:+.3f} | {_event_reason(row, signal_baseline)}"
        )
    return lines


def _format_watchlist(
    event_signals: pd.DataFrame,
    signal_baseline: pd.DataFrame | None = None,
    max_rows: int = 3,
) -> list[str]:
    if event_signals.empty:
        return ["- Yok"]
    rows: list[str] = []
    for _, row in event_signals.head(max_rows).iterrows():
        score = float(row["avg_impact_score"])
        rows.append(
            f"- {row['tickers']}: "
            f"Oncelik: {int(row.get('priority_score', 0))}/100 | "
            f"Aksiyon: {_event_action(row, signal_baseline)} | "
            f"{_direction_label(score)} {_strength_label(row['signal_strength'])} {score:+.2f} | "
            f"{_event_type_label(row['event_type'])} | "
            f"Neden: {_brief_event_reason(row, signal_baseline)}."
        )
    return rows


def _format_ticker_list(rows: pd.DataFrame, max_rows: int = 5) -> str:
    if rows.empty:
        return "Yok"
    return ", ".join(f"{row['ticker']} {row['sentiment_score']:+.2f}" for _, row in rows.head(max_rows).iterrows())


def _html(text: object) -> str:
    return html.escape(str(text), quote=False)


def _format_telegram_watchlist(
    event_signals: pd.DataFrame,
    signal_baseline: pd.DataFrame | None = None,
    max_rows: int = 3,
) -> list[str]:
    if event_signals.empty:
        return ["Yok"]
    lines: list[str] = []
    for _, row in event_signals.head(max_rows).iterrows():
        score = float(row["avg_impact_score"])
        lines.append(
            f"<b>{_html(row['tickers'])}</b> "
            f"<code>{score:+.2f}</code> | "
            f"{_html(_event_action(row, signal_baseline))} | "
            f"oncelik <code>{int(row.get('priority_score', 0))}/100</code>\n"
            f"{_html(_brief_event_reason(row, signal_baseline))}"
        )
    return lines


def build_brief_report(
    report_date: pd.Timestamp,
    event_signals: pd.DataFrame,
    top_stocks: pd.DataFrame,
    bottom_stocks: pd.DataFrame,
    positive_news: pd.DataFrame,
    negative_news: pd.DataFrame,
    market_news: pd.DataFrame,
    daily_rows: pd.DataFrame,
    signal_baseline: pd.DataFrame | None = None,
    min_abs_score: float = 0.20,
) -> str:
    avg_score = float(daily_rows["sentiment_score"].mean()) if not daily_rows.empty else 0.0
    total_news = int(daily_rows["news_count"].sum()) if not daily_rows.empty else 0
    direction = "pozitif" if avg_score > 0.05 else "negatif" if avg_score < -0.05 else "notr"
    signal_count = _signal_count(event_signals, top_stocks, bottom_stocks, positive_news, negative_news, market_news)
    summary = _signal_summary(event_signals, signal_count)
    flow_summary = _flow_summary(daily_rows, event_signals, top_stocks, bottom_stocks, market_news)
    medium_strong_count = (
        int(event_signals["signal_strength"].isin(["medium", "strong"]).sum()) if not event_signals.empty else 0
    )
    weak_count = int(event_signals["signal_strength"].eq("weak").sum()) if not event_signals.empty else 0
    market_line = "Var" if not market_news.empty else "Yok"

    lines = [
        f"# BIST Kisa Ozet - {report_date.strftime('%Y-%m-%d')}",
        f"Durum: {summary}",
        f"Karar: {_brief_decision(event_signals, signal_count)}",
        f"Sonuc: {_daily_result(event_signals, signal_count)}",
        f"Not: {_brief_note(event_signals, market_news)}",
        f"Genel: {avg_score:+.3f} ({direction}), {total_news} haber",
        f"Akis: {flow_summary}",
        f"Oncelik seviyesi: {_priority_level(event_signals)}",
        f"Esik: |skor| >= {min_abs_score:.2f}",
        f"Gecmis karsilastirma: {_baseline_summary(signal_baseline, min_abs_score)}",
        f"Aksiyonlar: {_action_summary(event_signals, signal_baseline)}",
        f"Piyasa geneli: {market_line}",
        f"Olaylar: {medium_strong_count} orta/guclu, {weak_count} zayif",
        "Izlenecekler:",
        *_format_watchlist(event_signals, signal_baseline),
        f"Pozitif hisseler: {_format_ticker_list(top_stocks)}",
        f"Negatif hisseler: {_format_ticker_list(bottom_stocks)}",
    ]
    return "\n".join(lines) + "\n"


def build_telegram_report(
    report_date: pd.Timestamp,
    event_signals: pd.DataFrame,
    top_stocks: pd.DataFrame,
    bottom_stocks: pd.DataFrame,
    positive_news: pd.DataFrame,
    negative_news: pd.DataFrame,
    market_news: pd.DataFrame,
    daily_rows: pd.DataFrame,
    signal_baseline: pd.DataFrame | None = None,
    min_abs_score: float = 0.20,
) -> str:
    avg_score = float(daily_rows["sentiment_score"].mean()) if not daily_rows.empty else 0.0
    total_news = int(daily_rows["news_count"].sum()) if not daily_rows.empty else 0
    direction = "pozitif" if avg_score > 0.05 else "negatif" if avg_score < -0.05 else "notr"
    signal_count = _signal_count(event_signals, top_stocks, bottom_stocks, positive_news, negative_news, market_news)
    flow_summary = _flow_summary(daily_rows, event_signals, top_stocks, bottom_stocks, market_news)
    medium_strong_count = (
        int(event_signals["signal_strength"].isin(["medium", "strong"]).sum()) if not event_signals.empty else 0
    )
    weak_count = int(event_signals["signal_strength"].eq("weak").sum()) if not event_signals.empty else 0
    market_line = "Var" if not market_news.empty else "Yok"
    watchlist = "\n\n".join(_format_telegram_watchlist(event_signals, signal_baseline))

    lines = [
        f"<b>BIST Gunluk Ozet</b> <code>{report_date.strftime('%Y-%m-%d')}</code>",
        "",
        f"<b>Karar</b>\n{_html(_brief_decision(event_signals, signal_count))}",
        "",
        f"<b>Sonuc</b>\n{_html(_daily_result(event_signals, signal_count))}",
        "",
        "<b>Durum</b>",
        f"<code>Genel    {avg_score:+.3f} ({direction})</code>",
        f"<code>Haber    {total_news}</code>",
        f"<code>Oncelik  {_priority_level(event_signals)}</code>",
        f"<code>Olay     {medium_strong_count} orta/guclu, {weak_count} zayif</code>",
        "",
        f"<b>Akis</b>\n{_html(flow_summary)}",
        f"<b>Gecmis</b>\n{_html(_baseline_summary(signal_baseline, min_abs_score))}",
        f"<b>Aksiyonlar</b>\n{_html(_action_summary(event_signals, signal_baseline))}",
        f"<b>Piyasa geneli</b>\n{_html(market_line)}",
        "",
        f"<b>Izlenecekler</b>\n{watchlist}",
        "",
        f"<b>Pozitif</b>: {_html(_format_ticker_list(top_stocks))}",
        f"<b>Negatif</b>: {_html(_format_ticker_list(bottom_stocks))}",
        f"<b>Esik</b>: <code>|skor| &gt;= {min_abs_score:.2f}</code>",
    ]
    return "\n".join(lines)[:3900] + "\n"


def build_markdown_report(
    report_date: pd.Timestamp,
    event_signals: pd.DataFrame,
    top_stocks: pd.DataFrame,
    bottom_stocks: pd.DataFrame,
    positive_news: pd.DataFrame,
    negative_news: pd.DataFrame,
    market_news: pd.DataFrame,
    daily_rows: pd.DataFrame,
    signal_baseline: pd.DataFrame | None = None,
    min_abs_score: float = 0.20,
) -> str:
    avg_score = float(daily_rows["sentiment_score"].mean()) if not daily_rows.empty else 0.0
    total_news = int(daily_rows["news_count"].sum()) if not daily_rows.empty else 0
    direction = "pozitif" if avg_score > 0.05 else "negatif" if avg_score < -0.05 else "notr"
    date_text = report_date.strftime("%Y-%m-%d")
    signal_count = _signal_count(event_signals, top_stocks, bottom_stocks, positive_news, negative_news, market_news)
    summary = _signal_summary(event_signals, signal_count)
    flow_summary = _flow_summary(daily_rows, event_signals, top_stocks, bottom_stocks, market_news)

    lines = [
        f"# BIST Gunluk Sentiment Raporu - {date_text}",
        "",
        f"Genel skor: {avg_score:+.3f} ({direction}), haber adedi: {total_news}",
        f"Sinyal esigi: |skor| >= {min_abs_score:.2f}",
        summary,
        "",
        "## Gunluk Karar",
        f"Karar: {_brief_decision(event_signals, signal_count)}",
        f"Sonuc: {_daily_result(event_signals, signal_count)}",
        f"Akis: {flow_summary}",
        f"Oncelik seviyesi: {_priority_level(event_signals)}",
        "",
        "## Onemli Olay Ozeti",
        *_format_event_rows(event_signals),
        "",
        "## Aksiyon Ozeti",
        f"Dagilim: {_action_summary(event_signals, signal_baseline)}",
        *_format_action_rows(event_signals, signal_baseline),
        "",
        "## Gecmis Karsilastirma",
        f"Ozet: {_baseline_summary(signal_baseline, min_abs_score)}",
        *_format_anomaly_rows(select_notable_anomalies(signal_baseline, min_abs_score=min_abs_score, top_n=10) if signal_baseline is not None else pd.DataFrame()),
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
    baseline_daily_sentiment_csv: Path | None = None,
    date: str | None = None,
    top_n: int = 10,
    min_abs_score: float = 0.20,
    baseline_lookback_days: int = 60,
    baseline_min_history: int = 5,
) -> dict[str, pd.DataFrame | Path | str]:
    scored = prepare_scored_news(scored_news_csv)
    daily = prepare_daily_sentiment(daily_sentiment_csv)
    report_date = normalize_date(date, pd.concat([scored["date"], daily["date"]], ignore_index=True))
    baseline_daily = prepare_daily_sentiment(baseline_daily_sentiment_csv) if baseline_daily_sentiment_csv else None
    baseline_input = build_baseline_daily_frame(daily, report_date, baseline_daily)

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
    signal_baseline = build_signal_baseline(
        baseline_input,
        report_date,
        lookback_days=baseline_lookback_days,
        min_history=baseline_min_history,
        min_abs_score=min_abs_score,
    )
    event_signals = add_event_priority(event_signals, signal_baseline)

    out_dir.mkdir(parents=True, exist_ok=True)
    date_slug = report_date.strftime("%Y-%m-%d")
    paths = {
        "event_signals": out_dir / f"{date_slug}_event_signals.csv",
        "top_stocks": out_dir / f"{date_slug}_top_stocks.csv",
        "bottom_stocks": out_dir / f"{date_slug}_bottom_stocks.csv",
        "positive_news": out_dir / f"{date_slug}_positive_news.csv",
        "negative_news": out_dir / f"{date_slug}_negative_news.csv",
        "market_news": out_dir / f"{date_slug}_market_news.csv",
        "signal_baseline": out_dir / f"{date_slug}_signal_baseline.csv",
        "brief": out_dir / f"{date_slug}_brief.md",
        "telegram": out_dir / f"{date_slug}_telegram.html",
        "markdown": out_dir / f"{date_slug}_daily_alerts.md",
    }
    event_signals.to_csv(paths["event_signals"], index=False)
    top_stocks.to_csv(paths["top_stocks"], index=False)
    bottom_stocks.to_csv(paths["bottom_stocks"], index=False)
    positive_news.to_csv(paths["positive_news"], index=False)
    negative_news.to_csv(paths["negative_news"], index=False)
    market_news.to_csv(paths["market_news"], index=False)
    signal_baseline.to_csv(paths["signal_baseline"], index=False)

    markdown = build_markdown_report(
        report_date,
        event_signals,
        top_stocks,
        bottom_stocks,
        positive_news,
        negative_news,
        market_news,
        daily_day,
        signal_baseline=signal_baseline,
        min_abs_score=min_abs_score,
    )
    brief = build_brief_report(
        report_date,
        event_signals,
        top_stocks,
        bottom_stocks,
        positive_news,
        negative_news,
        market_news,
        daily_day,
        signal_baseline=signal_baseline,
        min_abs_score=min_abs_score,
    )
    telegram = build_telegram_report(
        report_date,
        event_signals,
        top_stocks,
        bottom_stocks,
        positive_news,
        negative_news,
        market_news,
        daily_day,
        signal_baseline=signal_baseline,
        min_abs_score=min_abs_score,
    )
    paths["brief"].write_text(brief, encoding="utf-8")
    paths["telegram"].write_text(telegram, encoding="utf-8")
    paths["markdown"].write_text(markdown, encoding="utf-8")

    return {
        "date": date_slug,
        "brief_text": brief,
        "brief_path": paths["brief"],
        "telegram_text": telegram,
        "telegram_path": paths["telegram"],
        "markdown_text": markdown,
        "markdown_path": paths["markdown"],
        "event_signals": event_signals,
        "top_stocks": top_stocks,
        "bottom_stocks": bottom_stocks,
        "positive_news": positive_news,
        "negative_news": negative_news,
        "market_news": market_news,
        "signal_baseline": signal_baseline,
    }


def send_telegram_message(token: str, chat_id: str, text: str, parse_mode: str | None = None) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload_data = {"chat_id": chat_id, "text": text[:3900]}
    if parse_mode:
        payload_data["parse_mode"] = parse_mode
    payload = urllib.parse.urlencode(payload_data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        if response.status >= 400:
            raise RuntimeError(f"Telegram gonderimi basarisiz: HTTP {response.status}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gunluk BIST sentiment ranking ve haber alarm raporu uretir.")
    parser.add_argument("--scored-news", type=Path, required=True, help="score_news satir bazli cikti CSV.")
    parser.add_argument("--daily-sentiment", type=Path, required=True, help="score_news gunluk sentiment cikti CSV.")
    parser.add_argument("--baseline-daily-sentiment", type=Path, help="Gecmis karsilastirma icin ayrica kullanilacak daily sentiment CSV.")
    parser.add_argument("--date", help="Rapor tarihi. Bos verilirse verideki son tarih kullanilir.")
    parser.add_argument("--top-n", type=int, default=10, help="Listelerde gosterilecek satir sayisi.")
    parser.add_argument("--min-abs-score", type=float, default=0.20, help="Rapora girmek icin gereken minimum mutlak sentiment skoru.")
    parser.add_argument("--baseline-lookback-days", type=int, default=60, help="Gecmis karsilastirma icin geriye donuk gun sayisi.")
    parser.add_argument("--baseline-min-history", type=int, default=5, help="Anomali yorumu icin gereken minimum gecmis gun sayisi.")
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
        baseline_daily_sentiment_csv=args.baseline_daily_sentiment,
        date=args.date,
        top_n=args.top_n,
        min_abs_score=args.min_abs_score,
        baseline_lookback_days=args.baseline_lookback_days,
        baseline_min_history=args.baseline_min_history,
    )
    print(f"Gunluk rapor kaydedildi: {result['markdown_path']}")
    if args.send_telegram:
        token = os.getenv(args.telegram_token_env, "")
        chat_id = os.getenv(args.telegram_chat_id_env, "")
        if not token or not chat_id:
            raise RuntimeError(f"Telegram icin {args.telegram_token_env} ve {args.telegram_chat_id_env} env var gerekli.")
        send_telegram_message(token, chat_id, str(result["telegram_text"]), parse_mode="HTML")
        print("Telegram mesaji gonderildi.")


if __name__ == "__main__":
    main()
