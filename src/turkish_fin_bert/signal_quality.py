import hashlib
import math
import re
import unicodedata

import pandas as pd


EVENT_TYPES = [
    (
        "debt_credit",
        1.00,
        [
            "temerrut",
            "iflas",
            "konkordato",
            "borc",
            "kredi",
            "tahvil",
            "finansman",
        ],
    ),
    (
        "legal_regulatory",
        0.95,
        [
            "dava",
            "ceza",
            "sorusturma",
            "tedbir",
            "spk",
            "brsa",
            "rekabet kurumu",
        ],
    ),
    (
        "capital_action",
        0.90,
        [
            "sermaye",
            "temettu",
            "bedelli",
            "bedelsiz",
            "geri alim",
            "pay alimi",
            "pay satimi",
        ],
    ),
    (
        "financial_result",
        0.85,
        [
            "finansal tablo",
            "bilanco",
            "kar veya zarar",
            "faaliyet raporu",
            "gelir tablosu",
        ],
    ),
    (
        "contract_order",
        0.80,
        [
            "sozlesme",
            "ihale",
            "is iliskisi",
            "siparis",
        ],
    ),
    (
        "management",
        0.55,
        [
            "yonetim",
            "atama",
            "istifa",
            "genel mudur",
            "yonetim kurulu",
        ],
    ),
    (
        "routine",
        0.25,
        [
            "sirket genel bilgi formu",
            "kurumsal yonetim bilgi formu",
            "hak kullanimi",
            "fon kullanimi",
        ],
    ),
]

MARKET_SCOPE_KEYWORDS = [
    "bist",
    "xu100",
    "borsa istanbul",
    "endeks",
    "tcmb",
    "faiz",
    "enflasyon",
    "doviz",
    "resmi gazete",
]


def fold_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return folded.lower()


def compact_text(value: object) -> str:
    text = fold_text(value)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def classify_event(text: str) -> tuple[str, float]:
    haystack = compact_text(text)
    for event_type, weight, keywords in EVENT_TYPES:
        if any(keyword in haystack for keyword in keywords):
            return event_type, weight
    return "other", 0.45


def event_group_key(row: pd.Series) -> str:
    url = str(row.get("url", "")).strip()
    if url:
        return f"url:{url}"
    ticker = str(row.get("ticker", "")).strip().upper()
    title = compact_text(row.get("title", ""))[:120]
    return f"title:{ticker}:{title}"


def make_event_id(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def infer_market_scope(rows: pd.DataFrame, tickers: list[str]) -> str:
    if not tickers:
        return "market_wide"
    haystack = compact_text(" ".join(rows["title"].fillna("").astype(str).tolist()))
    haystack += " " + compact_text(" ".join(rows["text"].fillna("").astype(str).tolist()))
    if any(keyword in haystack for keyword in MARKET_SCOPE_KEYWORDS):
        return "market_wide"
    if len(tickers) > 1:
        return "multi_stock"
    return "stock_specific"


def signal_direction(score: float, min_abs_score: float) -> str:
    if score >= min_abs_score:
        return "positive"
    if score <= -min_abs_score:
        return "negative"
    return "neutral"


def signal_strength(materiality: float, score: float, min_abs_score: float) -> str:
    if abs(score) < min_abs_score:
        return "none"
    if materiality >= 0.75:
        return "strong"
    if materiality >= 0.50:
        return "medium"
    return "weak"


def materiality_score(
    score: float,
    confidence: float,
    event_weight: float,
    news_count: int,
    market_scope: str,
) -> float:
    repeat_boost = min(0.15, math.log1p(max(news_count - 1, 0)) * 0.08)
    scope_boost = 0.08 if market_scope == "market_wide" else 0.04 if market_scope == "multi_stock" else 0.0
    score_part = min(1.0, abs(score)) * 0.55
    confidence_part = min(1.0, max(0.0, confidence)) * 0.20
    event_part = event_weight * 0.20
    value = score_part + confidence_part + event_part + repeat_boost + scope_boost
    return round(min(1.0, max(0.0, value)), 4)


def build_event_signals(news_day: pd.DataFrame, min_abs_score: float = 0.20, top_n: int = 10) -> pd.DataFrame:
    if news_day.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "tickers",
                "primary_ticker",
                "title",
                "event_type",
                "market_scope",
                "news_count",
                "avg_impact_score",
                "confidence",
                "materiality_score",
                "signal_direction",
                "signal_strength",
                "reason",
                "url",
            ]
        )

    data = news_day.copy()
    if "impact_score" not in data:
        data["impact_score"] = data["prob_positive"] - data["prob_negative"]
    if "confidence" not in data:
        data["confidence"] = data[["prob_negative", "prob_positive"]].max(axis=1)
    data["event_key"] = data.apply(event_group_key, axis=1)

    rows: list[dict[str, object]] = []
    for key, group in data.groupby("event_key", sort=False):
        tickers = sorted({str(value).strip().upper() for value in group["ticker"].fillna("") if str(value).strip()})
        title = str(group["title"].fillna("").astype(str).iloc[0]).strip()
        text_blob = f"{title} {' '.join(group['text'].fillna('').astype(str).tolist())}"
        event_type, event_weight = classify_event(text_blob)
        avg_score = float(group["impact_score"].mean())
        confidence = float(group["confidence"].mean())
        scope = infer_market_scope(group, tickers)
        news_count = int(len(group))
        materiality = materiality_score(avg_score, confidence, event_weight, news_count, scope)
        direction = signal_direction(avg_score, min_abs_score)
        strength = signal_strength(materiality, avg_score, min_abs_score)
        primary_ticker = tickers[0] if tickers else "GENEL"
        reason = f"{direction}; {strength}; {event_type}; {scope}; {news_count} haber"
        rows.append(
            {
                "event_id": make_event_id(key),
                "tickers": ",".join(tickers) if tickers else "GENEL",
                "primary_ticker": primary_ticker,
                "title": title,
                "event_type": event_type,
                "market_scope": scope,
                "news_count": news_count,
                "avg_impact_score": round(avg_score, 4),
                "confidence": round(confidence, 4),
                "materiality_score": materiality,
                "signal_direction": direction,
                "signal_strength": strength,
                "reason": reason,
                "url": str(group["url"].fillna("").astype(str).iloc[0]).strip() if "url" in group else "",
            }
        )

    events = pd.DataFrame(rows)
    events = events[events["signal_strength"].ne("none")]
    if events.empty:
        return events
    return events.sort_values(
        ["materiality_score", "avg_impact_score"],
        ascending=[False, False],
    ).head(top_n)
