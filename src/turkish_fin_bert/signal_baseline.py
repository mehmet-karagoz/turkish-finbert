from __future__ import annotations

import math

import pandas as pd


BASELINE_COLUMNS = [
    "date",
    "ticker",
    "sentiment_score",
    "news_count",
    "history_days",
    "baseline_mean",
    "baseline_std",
    "baseline_abs_p90",
    "delta_from_mean",
    "anomaly_z",
    "anomaly_level",
]


def _prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    needed = {"date", "ticker", "sentiment_score", "news_count"}
    missing = needed - set(daily.columns)
    if missing:
        raise ValueError(f"Daily sentiment icinde eksik kolonlar: {sorted(missing)}")

    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["ticker"] = df["ticker"].fillna("").astype(str).str.upper().str.replace(".IS", "", regex=False).str.strip()
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    df["news_count"] = pd.to_numeric(df["news_count"], errors="coerce").fillna(0).astype(int)
    return df.dropna(subset=["date", "ticker", "sentiment_score"]).loc[lambda data: data["ticker"].ne("")]


def _classify_row(
    row: pd.Series,
    min_abs_score: float,
    min_history: int,
    elevated_z: float,
    unusual_z: float,
    min_std: float,
) -> str:
    history_days = int(row["history_days"])
    if history_days < min_history:
        return "insufficient_history"

    score = float(row["sentiment_score"])
    if abs(score) < min_abs_score:
        return "normal"

    baseline_std = float(row["baseline_std"])
    delta = float(row["delta_from_mean"])
    anomaly_z = row["anomaly_z"]
    if baseline_std < min_std or pd.isna(anomaly_z):
        if abs(delta) >= 0.40:
            return "unusual"
        if abs(delta) >= min_abs_score:
            return "elevated"
        return "normal"

    z_abs = abs(float(anomaly_z))
    if z_abs >= unusual_z:
        return "unusual"
    if z_abs >= elevated_z:
        return "elevated"
    return "normal"


def build_signal_baseline(
    daily: pd.DataFrame,
    report_date: pd.Timestamp | str,
    lookback_days: int = 60,
    min_history: int = 5,
    min_abs_score: float = 0.20,
    elevated_z: float = 1.5,
    unusual_z: float = 2.0,
    min_std: float = 0.03,
) -> pd.DataFrame:
    """Compare today's ticker scores against prior daily sentiment only."""
    df = _prepare_daily(daily)
    if df.empty:
        return pd.DataFrame(columns=BASELINE_COLUMNS)

    report_ts = pd.Timestamp(report_date).normalize()
    current = df[df["date"].eq(report_ts)].copy()
    if current.empty:
        return pd.DataFrame(columns=BASELINE_COLUMNS)

    history = df[df["date"].lt(report_ts)].copy()
    if lookback_days > 0:
        start_ts = report_ts - pd.Timedelta(days=lookback_days)
        history = history[history["date"].ge(start_ts)]

    if history.empty:
        current["history_days"] = 0
        current["baseline_mean"] = math.nan
        current["baseline_std"] = math.nan
        current["baseline_abs_p90"] = math.nan
    else:
        stats = (
            history.groupby("ticker")["sentiment_score"]
            .agg(
                history_days="count",
                baseline_mean="mean",
                baseline_std=lambda values: float(values.std(ddof=0)),
                baseline_abs_p90=lambda values: float(values.abs().quantile(0.90)),
            )
            .reset_index()
        )
        current = current.merge(stats, on="ticker", how="left")
        current["history_days"] = current["history_days"].fillna(0).astype(int)

    current["delta_from_mean"] = current["sentiment_score"] - current["baseline_mean"]
    current["anomaly_z"] = current["delta_from_mean"] / current["baseline_std"]
    current.loc[current["baseline_std"].lt(min_std) | current["baseline_std"].isna(), "anomaly_z"] = math.nan
    current["anomaly_level"] = current.apply(
        _classify_row,
        axis=1,
        min_abs_score=min_abs_score,
        min_history=min_history,
        elevated_z=elevated_z,
        unusual_z=unusual_z,
        min_std=min_std,
    )
    current["abs_sentiment_score"] = current["sentiment_score"].abs()
    current["level_rank"] = current["anomaly_level"].map(
        {"unusual": 3, "elevated": 2, "normal": 1, "insufficient_history": 0}
    )
    current = current.sort_values(["level_rank", "abs_sentiment_score", "news_count"], ascending=[False, False, False])
    return current[BASELINE_COLUMNS].reset_index(drop=True)


def select_notable_anomalies(baseline: pd.DataFrame, min_abs_score: float = 0.20, top_n: int = 10) -> pd.DataFrame:
    if baseline.empty:
        return baseline.copy()
    rows = baseline[
        baseline["anomaly_level"].isin(["elevated", "unusual"])
        & baseline["sentiment_score"].abs().ge(min_abs_score)
    ].copy()
    if rows.empty:
        return rows
    rows["abs_sentiment_score"] = rows["sentiment_score"].abs()
    rows["level_rank"] = rows["anomaly_level"].map({"unusual": 2, "elevated": 1}).fillna(0)
    return rows.sort_values(["level_rank", "abs_sentiment_score", "news_count"], ascending=[False, False, False]).head(top_n)
