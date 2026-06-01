import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .console import configure_console
from .paths import FIGURE_DIR, ensure_project_dirs
from .visualize import plot_class_scores, plot_confusion_matrix


LABEL_ORDER = ["negative", "neutral", "positive"]


def temporal_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date").reset_index(drop=True)
    split_at = max(1, int(len(df) * (1 - test_size)))
    split_at = min(split_at, len(df) - 1)
    train_df = df.iloc[:split_at].copy()
    test_df = df.iloc[split_at:].copy()

    if train_df["label"].nunique() < 2:
        raise ValueError("Eğitim bölümünde en az iki label olmalı. Daha fazla etiketli veri ekleyin.")
    return train_df, test_df


def stratified_split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df["label"].value_counts().min() < 2:
        raise ValueError("Stratified split için her label sınıfında en az iki satır olmalı.")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    return train_df.copy(), test_df.copy()


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20_000)),
            ("clf", LogisticRegression(max_iter=1_000, class_weight="balanced")),
        ]
    )


def train_baseline(
    input_csv: Path,
    model_out: Path,
    report_dir: Path,
    test_size: float = 0.2,
    split_strategy: str = "temporal",
    seed: int = 42,
    fit_full: bool = True,
) -> dict:
    df = pd.read_csv(input_csv)
    needed = {"date", "text", "label"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Eksik kolonlar: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "text", "label"]).copy()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(LABEL_ORDER)].copy()
    if len(df) < 6:
        raise ValueError("Baseline eğitim için en az 6 etiketli satır önerilir.")

    if split_strategy == "stratified":
        train_df, test_df = stratified_split(df, test_size=test_size, seed=seed)
    else:
        train_df, test_df = temporal_split(df, test_size=test_size)

    model = build_model()
    model.fit(train_df["text"], train_df["label"])
    pred = model.predict(test_df["text"])

    report = classification_report(test_df["label"], pred, labels=LABEL_ORDER, output_dict=True, zero_division=0)
    metrics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "fit_rows": int(len(df) if fit_full else len(train_df)),
        "split_strategy": split_strategy,
        "accuracy": float(accuracy_score(test_df["label"], pred)),
        "macro_f1": float(f1_score(test_df["label"], pred, labels=LABEL_ORDER, average="macro", zero_division=0)),
        "classification_report": report,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    final_model = build_model()
    if fit_full:
        final_model.fit(df["text"], df["label"])
    else:
        final_model = model
    joblib.dump(final_model, model_out)

    (report_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    cm = confusion_matrix(test_df["label"], pred, labels=LABEL_ORDER)
    plot_confusion_matrix(cm, LABEL_ORDER, FIGURE_DIR / "confusion_matrix.png")
    plot_class_scores(report, FIGURE_DIR / "class_scores.png")

    predictions = test_df[["date", "ticker", "title", "text", "label"]].copy() if "ticker" in test_df and "title" in test_df else test_df[["date", "text", "label"]].copy()
    predictions["prediction"] = pred
    predictions.to_csv(report_dir / "baseline_test_predictions.csv", index=False)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TF-IDF + LogisticRegression baseline sentiment modeli eğitir.")
    parser.add_argument("--input", type=Path, required=True, help="Temizlenmiş ve etiketli CSV.")
    parser.add_argument("--model-out", type=Path, default=Path("models/baseline_sentiment.joblib"), help="Model çıktı yolu.")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"), help="Rapor klasörü.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Zaman sıralı test oranı.")
    parser.add_argument("--split-strategy", choices=["temporal", "stratified"], default="temporal", help="Validasyon ayrımı.")
    parser.add_argument("--seed", type=int, default=42, help="Stratified split rastgelelik tohumu.")
    parser.add_argument("--no-fit-full", action="store_true", help="Kaydedilen modeli tüm veriyle yeniden eğitme.")
    return parser


def main() -> None:
    configure_console()
    ensure_project_dirs()
    args = build_parser().parse_args()
    metrics = train_baseline(
        args.input,
        args.model_out,
        args.report_dir,
        args.test_size,
        split_strategy=args.split_strategy,
        seed=args.seed,
        fit_full=not args.no_fit_full,
    )
    print(f"Model kaydedildi: {args.model_out}")
    print(f"Accuracy={metrics['accuracy']:.3f} Macro-F1={metrics['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
