import re
import unicodedata


HTML_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
MOJIBAKE_MARKERS = ("Ã", "Ä", "Å", "Â", "�")


def mojibake_score(value: str) -> int:
    return sum(value.count(marker) for marker in MOJIBAKE_MARKERS)


def fix_turkish_mojibake(value: str) -> str:
    """UTF-8 metnin Windows-1254/Latin-1 gibi okunmasindan dogan bozulmalari onarir."""
    if not any(marker in value for marker in MOJIBAKE_MARKERS):
        return value

    candidates = [value]
    for encoding in ("cp1254", "cp1252", "latin1"):
        try:
            candidates.append(value.encode(encoding).decode("utf-8"))
        except UnicodeError:
            continue
    return min(candidates, key=mojibake_score)


def clean_text(value: object) -> str:
    """Finans metnini model eğitimine uygun, sade tek satır metne çevirir."""
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except TypeError:
        pass
    text = str(value)
    if text.strip().lower() in {"nan", "none", "nat"}:
        return ""
    text = fix_turkish_mojibake(text)
    text = unicodedata.normalize("NFKC", text)
    text = HTML_RE.sub(" ", text)
    text = text.replace("\u00a0", " ")
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def combine_title_text(title: object, text: object) -> str:
    title_clean = clean_text(title)
    text_clean = clean_text(text)
    if title_clean and text_clean and title_clean not in text_clean:
        return f"{title_clean}. {text_clean}"
    return text_clean or title_clean
