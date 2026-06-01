from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns


matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _save_current(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_label_distribution(df: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(7, 4))
    order = ["negative", "neutral", "positive"]
    sns.countplot(data=df, x="label", order=[label for label in order if label in set(df["label"])])
    plt.title("Etiket dağılımı")
    plt.xlabel("Etiket")
    plt.ylabel("Metin sayısı")
    _save_current(out)


def plot_text_lengths(df: pd.DataFrame, out: Path) -> None:
    lengths = df["text"].fillna("").str.len()
    plt.figure(figsize=(8, 4))
    sns.histplot(lengths, bins=30)
    plt.title("Metin uzunluğu dağılımı")
    plt.xlabel("Karakter sayısı")
    plt.ylabel("Metin sayısı")
    _save_current(out)


def plot_ticker_label_distribution(df: pd.DataFrame, out: Path) -> None:
    if "ticker" not in df or "label" not in df:
        return
    top = df["ticker"].value_counts().head(15).index
    view = df[df["ticker"].isin(top)]
    if view.empty:
        return
    table = pd.crosstab(view["ticker"], view["label"])
    table.plot(kind="bar", stacked=True, figsize=(10, 5))
    plt.title("Hisse bazlı etiket dağılımı")
    plt.xlabel("Hisse")
    plt.ylabel("Metin sayısı")
    _save_current(out)


def plot_source_counts(df: pd.DataFrame, out: Path) -> None:
    if "source" not in df:
        return
    counts = df["source"].fillna("unknown").value_counts().head(15)
    if counts.empty:
        return
    plt.figure(figsize=(8, 4))
    counts.plot(kind="bar")
    plt.title("Kaynak dağılımı")
    plt.xlabel("Kaynak")
    plt.ylabel("Metin sayısı")
    _save_current(out)


def plot_ticker_counts(df: pd.DataFrame, out: Path) -> None:
    if "ticker" not in df:
        return
    counts = df["ticker"].fillna("").astype(str).str.strip()
    counts = counts[counts.ne("")]
    if counts.empty:
        return
    plt.figure(figsize=(9, 4))
    counts.value_counts().head(20).plot(kind="bar")
    plt.title("Hisse bazlı metin sayısı")
    plt.xlabel("Hisse")
    plt.ylabel("Metin sayısı")
    _save_current(out)


def plot_daily_counts(df: pd.DataFrame, out: Path) -> None:
    if "date" not in df:
        return
    daily = df.dropna(subset=["date"]).groupby("date").size()
    if daily.empty:
        return
    plt.figure(figsize=(10, 4))
    daily.plot()
    plt.title("Günlük metin sayısı")
    plt.xlabel("Tarih")
    plt.ylabel("Metin sayısı")
    _save_current(out)


def plot_confusion_matrix(cm, labels: list[str], out: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion matrix")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    _save_current(out)


def plot_class_scores(report: dict, out: Path) -> None:
    rows = []
    for label in ["negative", "neutral", "positive"]:
        if label in report:
            rows.append(
                {
                    "label": label,
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "f1-score": report[label]["f1-score"],
                }
            )
    if not rows:
        return
    scores = pd.DataFrame(rows).melt(id_vars="label", var_name="metric", value_name="score")
    plt.figure(figsize=(8, 4))
    sns.barplot(data=scores, x="label", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.title("Sınıf bazlı model skorları")
    plt.xlabel("Etiket")
    plt.ylabel("Skor")
    _save_current(out)


def plot_price_sentiment(df: pd.DataFrame, ticker: str, out: Path) -> None:
    view = df[df["ticker"] == ticker].sort_values("date")
    if view.empty:
        return
    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax1.plot(view["date"], view["close"], color="tab:blue", label="Kapanış")
    ax1.set_ylabel("Kapanış", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(view["date"], view["sentiment_score"], color="tab:orange", label="Sentiment")
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.set_ylabel("Sentiment", color="tab:orange")
    plt.title(f"{ticker} fiyat ve sentiment")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160)
    plt.close()


def plot_bucket_returns(df: pd.DataFrame, out: Path) -> None:
    cols = [col for col in ["forward_return_1d", "forward_return_5d", "forward_return_20d"] if col in df]
    if not cols or "sentiment_bucket" not in df:
        return
    summary = df.groupby("sentiment_bucket", observed=True)[cols].mean().reset_index()
    melted = summary.melt(id_vars="sentiment_bucket", var_name="horizon", value_name="avg_return")
    plt.figure(figsize=(9, 4))
    sns.barplot(data=melted, x="sentiment_bucket", y="avg_return", hue="horizon")
    plt.title("Sentiment seviyesine göre ileri getiri")
    plt.xlabel("Sentiment bucket")
    plt.ylabel("Ortalama ileri getiri")
    _save_current(out)


def plot_return_correlation(df: pd.DataFrame, out: Path) -> None:
    cols = [col for col in ["sentiment_score", "sentiment_3d", "sentiment_7d", "sentiment_14d", "forward_return_1d", "forward_return_5d", "forward_return_20d"] if col in df]
    if len(cols) < 2:
        return
    corr = df[cols].corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Sentiment ve ileri getiri korelasyonu")
    _save_current(out)


def plot_equity_curve(df: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["strategy_equity"], label="Sentiment stratejisi")
    if "benchmark_equity" in df:
        plt.plot(df["date"], df["benchmark_equity"], label="Eşit ağırlıklı benchmark")
    plt.title("Backtest kümülatif getiri")
    plt.xlabel("Tarih")
    plt.ylabel("Sermaye eğrisi")
    plt.legend()
    _save_current(out)


def plot_drawdown(df: pd.DataFrame, out: Path) -> None:
    equity = df["strategy_equity"]
    drawdown = equity / equity.cummax() - 1
    plt.figure(figsize=(10, 4))
    plt.fill_between(df["date"], drawdown, 0, color="tab:red", alpha=0.35)
    plt.title("Strateji drawdown")
    plt.xlabel("Tarih")
    plt.ylabel("Drawdown")
    _save_current(out)
