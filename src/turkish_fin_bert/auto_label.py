import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

from .console import configure_console
from .create_labeling_batch import LABELS
from .paths import ensure_project_dirs
from .text import clean_text, combine_title_text


TURKISH_TRANSLATION = str.maketrans(
    {
        "ç": "c",
        "Ç": "C",
        "ğ": "g",
        "Ğ": "G",
        "ı": "i",
        "I": "I",
        "İ": "I",
        "ö": "o",
        "Ö": "O",
        "ş": "s",
        "Ş": "S",
        "ü": "u",
        "Ü": "U",
    }
)
SPACE_RE = re.compile(r"\s+")


NEGATIVE_RULES = [
    (
        "dividend_not_distributed",
        [
            "kar payi dagitilmamasi",
            "kar dagitilmamasi",
            "kar payi dagitilmayacak",
            "kar dagitilmayacak",
            "dagitilmamasi",
            "dagitilmamasina",
        ],
    ),
    (
        "legal_or_penalty",
        [
            "idari para cezasi",
            "para cezasi",
            "ceza",
            "sorusturma",
            "dava",
            "tedbir",
            "iflas",
            "konkordato",
            "haciz",
            "temerrut",
        ],
    ),
    (
        "cancellation_or_suspension",
        [
            "iptal",
            "faaliyet durdur",
            "faaliyetin durdur",
            "borsa kotundan cik",
            "islem sirasi kapat",
            "olumsuz",
        ],
    ),
]

POSITIVE_RULES = [
    (
        "share_buyback",
        [
            "paylarin geri alinmasi",
            "pay geri alim",
            "geri alim islemleri",
            "paylarin geri alimi",
        ],
    ),
    (
        "new_business_or_tender",
        [
            "yeni is iliskisi",
            "sozlesme imzalan",
            "ihale sureci",
            "ihale sonucu",
            "ihale kazan",
            "siparis",
            "is alindi",
        ],
    ),
    (
        "dividend_distribution",
        [
            "kar payi dagitim",
            "kar payi bildirimi",
            "temettu odemesi",
            "temettu dagitim",
            "kar dagitimina iliskin",
            "kar dagitimi",
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

NEUTRAL_RULES = [
    (
        "routine_governance",
        [
            "genel kurul",
            "bagimsiz denetim kurulusunun belirlenmesi",
            "yonetim kurulu komiteleri",
            "esas sozlesme",
            "sirket genel bilgi formu",
            "kurumsal yonetim bilgi formu",
            "sorumluluk beyani",
            "faaliyet raporu",
            "finansal rapor",
            "haftalik rapor",
            "degerleme raporu",
            "unvan degisikligi",
            "kar dagitim politikasi",
        ],
    ),
    (
        "capital_markets_routine",
        [
            "pay disinda sermaye piyasasi araci",
            "ihrac tavani",
            "tertip ihrac belgesi",
            "ihrac belgesi",
            "izahname",
            "yatirim kurulusu varant",
            "piyasa yapiciligi kapsaminda",
            "likidite saglayicilik kapsaminda",
            "pay satis bilgi formu",
            "sermaye artirimindan elde edilecek",
        ],
    ),
    (
        "generic_update",
        [
            "ozel durum aciklamasi genel",
            "iliskili taraf islemleri",
            "surdurulebilirlik raporu",
            "birlesme islemlerine iliskin bildirim",
            "pay alim teklifi yoluyla",
        ],
    ),
]


def normalize_for_rules(value: object) -> str:
    text = clean_text(value).translate(TURKISH_TRANSLATION).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return SPACE_RE.sub(" ", text).strip()


def _find_rule(text: str, rules: list[tuple[str, list[str]]]) -> str | None:
    for name, patterns in rules:
        if any(pattern in text for pattern in patterns):
            return name
    return None


def infer_label(title: object, text: object) -> tuple[str, str]:
    rule_text = normalize_for_rules(combine_title_text(title, text))

    negative_rule = _find_rule(rule_text, NEGATIVE_RULES)
    if negative_rule:
        return "negative", negative_rule

    positive_rule = _find_rule(rule_text, POSITIVE_RULES)
    if positive_rule:
        return "positive", positive_rule

    neutral_rule = _find_rule(rule_text, NEUTRAL_RULES)
    if neutral_rule:
        return "neutral", neutral_rule

    return "neutral", "default_neutral"


def auto_label_batch(input_csv: Path, output_csv: Path | None = None, overwrite: bool = False) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required = {"title", "text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Eksik kolonlar: {sorted(missing)}")

    df["label"] = df["label"].fillna("").astype(str)
    labels = df["label"].str.lower().str.strip()
    existing_bad = sorted(set(labels[labels.ne("")]) - LABELS)
    if existing_bad:
        raise ValueError(f"Geçersiz mevcut label değerleri: {existing_bad}")

    if "notes" not in df.columns:
        df["notes"] = ""
    df["notes"] = df["notes"].fillna("").astype(str)

    updated = 0
    for idx, row in df.iterrows():
        current_label = labels.loc[idx]
        if current_label and not overwrite:
            continue

        label, rule = infer_label(row.get("title", ""), row.get("text", ""))
        df.at[idx, "label"] = label
        existing_note = clean_text(row.get("notes", ""))
        auto_note = f"weak_label:{rule}"
        df.at[idx, "notes"] = f"{existing_note}; {auto_note}" if existing_note else auto_note
        updated += 1

    target = output_csv or input_csv
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    print(f"{updated} satır etiketlendi: {target}")
    print(df["label"].value_counts().to_string())
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KAP haber batch dosyasına kural tabanlı ön etiket basar.")
    parser.add_argument("--input", type=Path, required=True, help="Etiketlenecek CSV yolu.")
    parser.add_argument("--output", type=Path, default=None, help="Çıktı CSV yolu. Verilmezse input üstüne yazar.")
    parser.add_argument("--overwrite", action="store_true", help="Dolu label değerlerini de yeniden üretir.")
    return parser


def main() -> None:
    configure_console()
    ensure_project_dirs()
    args = build_parser().parse_args()
    auto_label_batch(args.input, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
