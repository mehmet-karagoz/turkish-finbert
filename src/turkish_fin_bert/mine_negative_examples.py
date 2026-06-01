import argparse
from pathlib import Path

import pandas as pd

from .auto_label import normalize_for_rules
from .console import configure_console
from .create_labeling_batch import BASE_COLUMNS, row_key
from .paths import ensure_project_dirs
from .text import clean_text


POSITIVE_LEGAL_OUTCOME_PATTERNS = [
    "sirketimiz lehine",
    "lehine karar",
    "iptal talebine red",
    "kesin red karari",
    "ceza ihbarnamesinin iptaline",
    "para cezasi hakkinda yurutmenin durdurulmasi",
    "idari para cezasi yurutmenin durdurulmasi",
    "konkordato surecinin sona ermesi",
    "konkordato surecinden feragat",
]

NEGATIVE_PATTERNS = [
    (
        "dividend_not_distributed",
        [
            "kar payi dagitilmamasi",
            "kar dagitilmamasi",
            "kar payi dagitilmayacak",
            "kar dagitilmayacak",
            "karinin dagitilmamasi",
        ],
    ),
    (
        "lawsuit_against_company",
        [
            "ortaklik aleyhine dava",
            "sirket aleyhine dava",
            "sirketimiz aleyhine dava",
            "bankamiza acilan dava",
            "aleyhine acilan dava",
            "tazminat davasi",
            "marka ihlal davasi",
            "itiraz davasi",
            "taraf oldugu dava",
            "iptal davasi",
        ],
    ),
    (
        "penalty_or_concordat",
        [
            "idari para cezasi",
            "para cezasi",
            "konkordato",
            "temerrut",
            "haciz",
        ],
    ),
    (
        "cancelled_or_rejected_action",
        [
            "pay geri alim iptali",
            "genel kurul iptali",
            "hak kullanim surec iptal",
            "basvurusunun geri cekilmesi",
            "basvurusunun kurul tarafindan olumsuz karsilanmasi",
            "olumsuz karsilanmasi",
            "fon kurulus iptal",
            "sermaye azaltimi",
        ],
    ),
]


def high_confidence_negative_reason(title: object, text: object) -> str:
    normalized = normalize_for_rules(f"{clean_text(title)} {clean_text(text)}")
    if any(pattern in normalized for pattern in POSITIVE_LEGAL_OUTCOME_PATTERNS):
        return ""
    for reason, patterns in NEGATIVE_PATTERNS:
        if any(pattern in normalized for pattern in patterns):
            return reason
    return ""


def _dedupe_key(df: pd.DataFrame) -> pd.Series:
    return (
        df.get("ticker", pd.Series("", index=df.index)).fillna("").astype(str).str.upper().str.strip()
        + "|"
        + df.get("title", pd.Series("", index=df.index)).fillna("").astype(str).map(normalize_for_rules)
        + "|"
        + df.get("text", pd.Series("", index=df.index)).fillna("").astype(str).map(normalize_for_rules)
    )


def mine_negative_examples(
    input_csvs: list[Path],
    output_csv: Path,
    exclude_labeled: list[Path] | None = None,
    max_rows: int = 100,
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
    df["notes"] = [high_confidence_negative_reason(row.title, row.text) for row in df.itertuples()]
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

    df = df.sort_values(["date", "ticker", "title"]).reset_index(drop=True)
    if max_rows > 0:
        df = df.head(max_rows).copy()

    df.insert(0, "label_id", [f"NEG{i:06d}" for i in range(1, len(df) + 1)])
    df.insert(1, "label", "negative")
    df["notes"] = "weak_label_negative:" + df["notes"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out = df[BASE_COLUMNS].copy()
    out.to_csv(output_csv, index=False)
    print(f"{len(out)} negatif aday etiketlendi: {output_csv}")
    if len(out):
        print(out["notes"].value_counts().to_string())
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KAP dosyalarından yüksek güvenli negatif etiketli örnek çıkarır.")
    parser.add_argument("--input", action="append", type=Path, required=True, help="Taranacak haber CSV. Birden fazla kez verilebilir.")
    parser.add_argument("--output", type=Path, default=Path("data/labels/negative_labeling_batch.csv"), help="Negatif etiketli CSV çıktısı.")
    parser.add_argument("--exclude-labeled", action="append", type=Path, default=[], help="Daha önce etiketlenmiş master/batch CSV.")
    parser.add_argument("--max-rows", type=int, default=100, help="Maksimum çıktı satırı. 0 sınırsız.")
    return parser


def main() -> None:
    configure_console()
    ensure_project_dirs()
    args = build_parser().parse_args()
    mine_negative_examples(args.input, args.output, args.exclude_labeled, args.max_rows)


if __name__ == "__main__":
    main()
