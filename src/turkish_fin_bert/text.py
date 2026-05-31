import re
import unicodedata


HTML_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


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
