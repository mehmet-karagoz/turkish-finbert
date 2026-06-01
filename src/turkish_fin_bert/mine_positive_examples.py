import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from .auto_label import normalize_for_rules
from .console import configure_console
from .create_labeling_batch import BASE_COLUMNS, row_key
from .mine_negative_examples import _dedupe_key
from .paths import ensure_project_dirs
from .text import clean_text


NEGATIVE_CONTEXT_PATTERNS = [
    "kar payi dagitilmamasi",
    "kar dagitilmamasi",
    "kar payi dagitilmayacak",
    "kar dagitilmayacak",
    "dagitilmamasi",
    "iptal",
    "geri cekilmesi",
    "olumsuz karsilanmasi",
    "dava",
    "ceza",
    "konkordato",
    "temerrut",
    "haciz",
]

POSITIVE_PATTERNS = [
    (
        "share_buyback",
        [
            "paylarin geri alinmasina iliskin bildirim",
            "paylarin geri alinmasi",
            "pay geri alim islemleri",
            "pay geri alim",
        ],
    ),
    (
        "new_business_contract",
        [
            "yeni is iliskisi",
            "sozlesme imzalanmasi",
            "sozlesme imzalan",
            "siparis al",
            "is alindi",
        ],
    ),
    (
        "tender_award",
        [
            "ihalenin uhdemizde kalmasi",
            "ihale kazan",
            "ihale sonucu",
            "ihale sureci sonucu",
        ],
    ),
    (
        "dividend_distribution",
        [
            "temettu odemesi",
            "temettu dagitim",
            "kar payi dagitim karari",
            "kar payi dagitimi",
            "kar payi odemesi",
            "nakit kar payi",
            "kar payi dagitilmasi",
        ],
    ),
    (
        "credit_rating_upgrade",
        [
            "notu yukselt",
            "not artir",
            "notu artir",
            "gorunum pozitif",
        ],
    ),
    (
        "capital_bonus_or_ceiling",
        [
            "bedelsiz sermaye artirimi",
            "kayitli sermaye tavani artir",
        ],
    ),
]


def high_confidence_positive_reason(title: object, text: object) -> str:
    normalized = normalize_for_rules(f"{clean_text(title)} {clean_text(text)}")
    if any(pattern in normalized for pattern in NEGATIVE_CONTEXT_PATTERNS):
        return ""
    for reason, patterns in POSITIVE_PATTERNS:
        if any(pattern in normalized for pattern in patterns):
            return reason
    return ""


def _balanced_sample(df: pd.DataFrame, max_rows: int, per_reason: int) -> pd.DataFrame:
    if per_reason > 0:
        parts = []
        for _, group in df.groupby("notes", sort=True):
            group = group.sort_values(["date", "ticker", "title"]).reset_index(drop=True)
            if len(group) > per_reason:
                positions = np.linspace(0, len(group) - 1, per_reason, dtype=int)
                group = group.iloc[positions].copy()
            parts.append(group)
        df = pd.concat(parts, ignore_index=True) if parts else df.head(0).copy()
    df = df.sort_values(["date", "ticker", "title"]).reset_index(drop=True)
    if max_rows > 0 and len(df) > max_rows:
        df = df.head(max_rows).copy()
    return df


def mine_positive_examples(
    input_csvs: list[Path],
    output_csv: Path,
    exclude_labeled: list[Path] | None = None,
    max_rows: int = 80,
    per_reason: int = 20,
) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in input_csvs]
    if not frames:
        raise ValueError("En az bir input CSV verilmeli.")

    df = pd.concat(frames, ignore_index=True)
    required = {"date", "ticker", "source", "title", "text", "url"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Eksik kolonlar: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["ticker"] = df["ticker"].fillna("").astype(str).str.upper().str.replace(".IS", "", regex=False).str.strip()
    df["source"] = df["source"].fillna("unknown").astype(str).map(clean_text)
    df["title"] = df["title"].fillna("").astype(str).map(clean_text)
    df["text"] = df["text"].fillna("").astype(str).map(clean_text)
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df = df[df["ticker"].ne("") & df["text"].ne("")].copy()
    df["notes"] = [high_confidence_positive_reason(row.title, row.text) for row in df.itertuples()]
    df = df[df["notes"].ne("")].copy()
    df = df.drop_duplicates(subset=["ticker", "title", "text"])

    for path in exclude_labeled or []:
        if not path.exists():
            continue
        labeled = pd.read_csv(path)
        if "label" in labeled.columns:
            labeled = labeled[labeled["label"].fillna("").astype(str).str.strip().ne("")]
        keys = set(row_key(labeled).astype(str)) | set(_dedupe_key(labeled).astype(str))
        df = df[~row_key(df).astype(str).isin(keys) & ~_dedupe_key(df).astype(str).isin(keys)].copy()

    df = _balanced_sample(df, max_rows=max_rows, per_reason=per_reason)
    df.insert(0, "label_id", [f"POS{i:06d}" for i in range(1, len(df) + 1)])
    df.insert(1, "label", "positive")
    df["notes"] = "weak_label_positive:" + df["notes"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out = df[BASE_COLUMNS].copy()
    out.to_csv(output_csv, index=False)
    print(f"{len(out)} pozitif aday etiketlendi: {output_csv}")
    if len(out):
        print(out["notes"].value_counts().to_string())
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KAP dosyalarından yüksek güvenli pozitif etiketli örnek çıkarır.")
    parser.add_argument("--input", action="append", type=Path, required=True, help="Taranacak haber CSV. Birden fazla kez verilebilir.")
    parser.add_argument("--output", type=Path, default=Path("data/labels/positive_labeling_batch.csv"), help="Pozitif etiketli CSV çıktısı.")
    parser.add_argument("--exclude-labeled", action="append", type=Path, default=[], help="Daha önce etiketlenmiş master/batch CSV.")
    parser.add_argument("--max-rows", type=int, default=80, help="Maksimum çıktı satırı. 0 sınırsız.")
    parser.add_argument("--per-reason", type=int, default=20, help="Her pozitif kuraldan maksimum satır. 0 kapalı.")
    return parser


def main() -> None:
    configure_console()
    ensure_project_dirs()
    args = build_parser().parse_args()
    mine_positive_examples(args.input, args.output, args.exclude_labeled, args.max_rows, args.per_reason)


if __name__ == "__main__":
    main()
