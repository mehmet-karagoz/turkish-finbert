import argparse
import csv
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd

from .console import configure_console
from .paths import ensure_project_dirs
from .text import clean_text


NEWS_COLUMNS = ["date", "ticker", "source", "title", "text", "url", "language", "published_at"]
DEFAULT_USER_AGENT = "turkish-fin-bert/0.1 research crawler"
TURKISH_TRANSLATION = str.maketrans(
    {
        "ç": "c",
        "Ç": "C",
        "ğ": "g",
        "Ğ": "G",
        "ı": "i",
        "İ": "I",
        "ö": "o",
        "Ö": "O",
        "ş": "s",
        "Ş": "S",
        "ü": "u",
        "Ü": "U",
        "â": "a",
        "Â": "A",
        "î": "i",
        "Î": "I",
        "û": "u",
        "Û": "U",
    }
)


@dataclass
class NewsItem:
    title: str
    text: str
    url: str
    published_at: str
    source: str


class ArticleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._tag_stack: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        self._tag_stack.append(tag)
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        current = self._tag_stack[-1] if self._tag_stack else ""
        value = clean_text(unescape(data))
        if not value:
            return
        if current == "title":
            self.title_parts.append(value)
        elif current in {"h1", "h2", "h3", "p", "li"}:
            self.text_parts.append(value)

    @property
    def title(self) -> str:
        return clean_text(" ".join(self.title_parts))

    @property
    def text(self) -> str:
        return clean_text(" ".join(self.text_parts))


def read_url(url: str, timeout: int = 20) -> str:
    request = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
        encoding = response.headers.get_content_charset() or "utf-8"
    return raw.decode(encoding, errors="replace")


def read_text_source(path_or_url: str) -> str:
    parsed = urlparse(path_or_url)
    if parsed.scheme in {"http", "https"}:
        return read_url(path_or_url)
    return Path(path_or_url).read_text(encoding="utf-8")


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def first_child_text(element: ET.Element, names: set[str]) -> str:
    for child in list(element):
        if local_name(child.tag) in names and child.text:
            return clean_text(child.text)
    return ""


def first_link(element: ET.Element) -> str:
    for child in list(element):
        if local_name(child.tag) != "link":
            continue
        href = child.attrib.get("href")
        if href:
            return href.strip()
        if child.text:
            return child.text.strip()
    return ""


def parse_datetime(value: str) -> tuple[str, str]:
    value = clean_text(value)
    if not value:
        return "", ""
    parsed: datetime | None = None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        parsed = None
    if parsed is None:
        dayfirst = bool(re.match(r"^\d{1,2}[./-]\d{1,2}[./-]\d{4}", value))
        parsed_ts = pd.to_datetime(value, errors="coerce", utc=True, dayfirst=dayfirst)
        if pd.notna(parsed_ts):
            parsed = parsed_ts.to_pydatetime()
    if parsed is None:
        return "", value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    return parsed.date().isoformat(), parsed.isoformat()


def parse_feed_items(feed_xml: str, source_name: str = "") -> list[NewsItem]:
    root = ET.fromstring(feed_xml)
    root_name = local_name(root.tag)
    items: list[ET.Element]

    if root_name == "rss":
        channel = next((child for child in list(root) if local_name(child.tag) == "channel"), root)
        items = [child for child in list(channel) if local_name(child.tag) == "item"]
    else:
        items = [child for child in list(root) if local_name(child.tag) == "entry"]

    parsed_items: list[NewsItem] = []
    for item in items:
        title = first_child_text(item, {"title"})
        link = first_link(item)
        summary = first_child_text(item, {"description", "summary", "content", "encoded"})
        published_raw = first_child_text(item, {"pubdate", "published", "updated", "date"})
        _, published_at = parse_datetime(published_raw)
        source = source_name or urlparse(link).netloc or "rss"
        parsed_items.append(NewsItem(title=title, text=strip_html(summary), url=link, published_at=published_at, source=source))
    return parsed_items


def strip_html(value: str) -> str:
    parser = ArticleTextParser()
    parser.feed(value or "")
    return parser.text or clean_text(re.sub(r"<[^>]+>", " ", value or ""))


def fetch_article(url: str, source: str = "") -> NewsItem:
    html = read_url(url)
    parser = ArticleTextParser()
    parser.feed(html)
    source_name = source or urlparse(url).netloc or "web"
    return NewsItem(title=parser.title, text=parser.text, url=url, published_at="", source=source_name)


def load_aliases(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    aliases: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ticker = clean_text(row.get("ticker", "")).upper().replace(".IS", "")
            alias = clean_text(row.get("alias", ""))
            if ticker and alias:
                aliases.setdefault(ticker, []).append(alias)
    return aliases


def detect_tickers(text: str, tickers: list[str], aliases: dict[str, list[str]] | None = None) -> list[str]:
    text_upper = text.upper()
    text_normalized = normalize_for_match(text)
    found: list[str] = []
    aliases = aliases or {}
    for ticker in tickers:
        normalized = ticker.upper().replace(".IS", "")
        ticker_pattern = rf"(?<![A-Z0-9]){re.escape(normalized)}(?:\.IS)?(?![A-Z0-9])"
        alias_hit = any(alias_matches(alias, text_upper, text_normalized) for alias in aliases.get(normalized, []))
        if re.search(ticker_pattern, text_upper) or alias_hit:
            found.append(normalized)
    return sorted(set(found))


def normalize_for_match(value: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", value.translate(TURKISH_TRANSLATION))
    ascii_text = "".join(char for char in ascii_text if not unicodedata.combining(char))
    ascii_text = re.sub(r"[^A-Za-z0-9]+", " ", ascii_text)
    return clean_text(ascii_text).upper()


def alias_matches(alias: str, text_upper: str, text_normalized: str) -> bool:
    alias_clean = clean_text(alias)
    if not alias_clean:
        return False
    alias_upper = alias_clean.upper()
    alias_normalized = normalize_for_match(alias_clean)

    if len(alias_normalized) <= 4:
        pattern = rf"(?<![A-Z0-9]){re.escape(alias_normalized)}(?![A-Z0-9])"
        return bool(re.search(pattern, text_normalized))
    return alias_upper in text_upper or alias_normalized in text_normalized


def rows_from_items(
    items: list[NewsItem],
    tickers: list[str],
    aliases: dict[str, list[str]],
    include_untagged: bool,
    fetch_article_text: bool,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for item in items:
        enriched = item
        if fetch_article_text and item.url:
            try:
                article = fetch_article(item.url, source=item.source)
                if len(article.text) > len(item.text):
                    enriched = NewsItem(
                        title=article.title or item.title,
                        text=article.text,
                        url=item.url,
                        published_at=item.published_at,
                        source=item.source,
                    )
            except Exception as exc:  # noqa: BLE001 - haber toplamada tek URL hatası tüm işi durdurmasın.
                print(f"Uyarı: haber metni alınamadı: {item.url} ({exc})")

        combined = f"{enriched.title} {enriched.text}"
        matched_tickers = detect_tickers(combined, tickers, aliases) if tickers else []
        if not matched_tickers and include_untagged:
            matched_tickers = [""]

        date, published_at = parse_datetime(enriched.published_at)
        if not date:
            date = datetime.now(timezone.utc).date().isoformat()
        for ticker in matched_tickers:
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "source": enriched.source,
                    "title": clean_text(enriched.title),
                    "text": clean_text(enriched.text),
                    "url": enriched.url,
                    "language": "tr",
                    "published_at": published_at or enriched.published_at,
                }
            )
    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def parse_kap_date(value: str | None) -> tuple[str, str]:
    return parse_datetime(clean_text(value or ""))


def split_stock_codes(value: str | None) -> list[str]:
    value = clean_text(value or "")
    if not value:
        return []
    return normalize_tickers(re.split(r"[,;\s]+", value))


def rows_from_kap_disclosures(disclosures: list[dict], tickers: list[str]) -> pd.DataFrame:
    allowed = set(tickers)
    rows: list[dict[str, str]] = []
    for item in disclosures:
        stock_codes = split_stock_codes(item.get("stockCodes") or item.get("stockCode"))
        if allowed:
            stock_codes = [ticker for ticker in stock_codes if ticker in allowed]
        if not stock_codes:
            continue

        date, published_at = parse_kap_date(item.get("publishDate"))
        disclosure_index = item.get("disclosureIndex")
        subject = clean_text(item.get("subject", ""))
        company = clean_text(item.get("kapTitle", ""))
        summary = clean_text(item.get("summary", ""))
        related = clean_text(item.get("relatedStocks", ""))
        title = clean_text(f"{company} - {subject}") if company else subject
        text = clean_text(" ".join(part for part in [subject, summary, related] if part))
        url = f"https://www.kap.org.tr/tr/Bildirim/{disclosure_index}" if disclosure_index else "https://www.kap.org.tr/tr/bildirim-sorgu"

        for ticker in stock_codes:
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "source": "KAP",
                    "title": title,
                    "text": text or title,
                    "url": url,
                    "language": "tr",
                    "published_at": published_at,
                }
            )
    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def read_url_list(path: Path) -> list[str]:
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            urls.append(value)
    return urls


def read_ticker_file(path: Path) -> list[str]:
    tickers: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        tickers.extend(part.strip() for part in re.split(r"[,;\s]+", value) if part.strip())
    return tickers


def normalize_tickers(tickers: list[str]) -> list[str]:
    return sorted({ticker.upper().replace(".IS", "").strip() for ticker in tickers if ticker.strip()})


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=NEWS_COLUMNS)
    return pd.read_csv(path)


def save_news(df: pd.DataFrame, out_csv: Path, append: bool) -> pd.DataFrame:
    existing = load_existing(out_csv) if append else pd.DataFrame(columns=NEWS_COLUMNS)
    combined = pd.concat([existing, df], ignore_index=True)
    for col in NEWS_COLUMNS:
        if col not in combined:
            combined[col] = ""
    combined = combined[NEWS_COLUMNS].fillna("")
    combined = combined.drop_duplicates(subset=["date", "ticker", "title", "url", "text"])
    combined = combined.sort_values(["date", "ticker", "source"], na_position="last")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    return combined


def fetch_rss_sources(args: argparse.Namespace) -> list[NewsItem]:
    items: list[NewsItem] = []
    rss_sources: list[str] = []
    rss_sources.extend(args.rss_url or [])
    for path in args.rss_url_file or []:
        rss_sources.extend(read_url_list(path))

    for source in rss_sources:
        feed_xml = read_text_source(source)
        items.extend(parse_feed_items(feed_xml, source_name=urlparse(source).netloc or "rss"))
    for source in args.rss_file or []:
        feed_xml = read_text_source(source)
        items.extend(parse_feed_items(feed_xml, source_name=Path(source).stem))
    return items


def fetch_url_sources(args: argparse.Namespace, source_name: str) -> list[NewsItem]:
    urls: list[str] = []
    urls.extend(args.url or [])
    for path in args.url_file or []:
        urls.extend(read_url_list(path))
    return [fetch_article(url, source=source_name) for url in urls]


def _kap_date(value: str | None, fallback: datetime) -> str:
    if not value:
        return fallback.date().isoformat()
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return fallback.date().isoformat()
    return parsed.date().isoformat()


def fetch_kap_api(args: argparse.Namespace) -> list[dict]:
    now = datetime.now(timezone.utc)
    from_date = _kap_date(args.kap_from_date, now - timedelta(days=args.kap_days))
    to_date = _kap_date(args.kap_to_date, now)
    body = {
        "fromDate": from_date,
        "toDate": to_date,
        "memberType": "IGS",
        "mkkMemberOidList": [],
        "inactiveMkkMemberOidList": [],
        "disclosureClass": args.kap_disclosure_class if args.kap_disclosure_class != "ALL" else "",
        "subjectList": [],
        "isLate": "",
        "mainSector": "",
        "sector": "",
        "subSector": "",
        "marketOid": "",
        "index": "",
        "bdkReview": "",
        "bdkMemberOidList": [],
        "year": "",
        "term": "",
        "ruleType": "",
        "period": "",
        "fromSrc": False,
        "srcCategory": "",
        "disclosureIndexList": [],
    }
    request = Request(
        "https://www.kap.org.tr/tr/api/disclosure/members/byCriteria",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Content-Type": "application/json",
            "Accept-Language": "tr",
        },
        method="POST",
    )
    with urlopen(request, timeout=40) as response:
        raw = response.read()
    data = json.loads(raw.decode("utf-8", errors="replace"))
    return data if isinstance(data, list) else []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gerçek haber/KAP metinlerini standart haber CSV şemasına toplar.")
    parser.add_argument("--source", choices=["rss", "urls", "kap-links", "kap-api"], required=True, help="Toplanacak kaynak tipi.")
    parser.add_argument("--tickers", nargs="*", default=[], help="Aranacak BIST sembolleri: THYAO ASELS GARAN")
    parser.add_argument("--tickers-file", action="append", type=Path, help="Satır satır veya virgülle ayrılmış BIST sembolleri dosyası.")
    parser.add_argument("--aliases", type=Path, default=None, help="ticker,alias kolonlu şirket adı eşleştirme CSV'si.")
    parser.add_argument("--out", type=Path, default=Path("data/raw/news.csv"), help="Haber çıktı CSV yolu.")
    parser.add_argument("--append", action="store_true", help="Var olan CSV üzerine ekle ve tekrarları sil.")
    parser.add_argument("--include-untagged", action="store_true", help="Ticker bulunmayan haberleri de boş ticker ile kaydet.")
    parser.add_argument("--fetch-article-text", action="store_true", help="RSS özetinin yanında haber sayfasından tam metin çekmeyi dener.")
    parser.add_argument("--limit", type=int, default=0, help="Kaydedilecek maksimum haber sayısı. 0 sınırsız.")

    parser.add_argument("--rss-url", action="append", help="RSS/Atom URL. Birden fazla kez verilebilir.")
    parser.add_argument("--rss-url-file", action="append", type=Path, help="Satır satır RSS/Atom URL içeren dosya.")
    parser.add_argument("--rss-file", action="append", help="Yerel RSS/Atom XML dosyası. Test ve arşiv için kullanışlı.")
    parser.add_argument("--url", action="append", help="Doğrudan haber veya KAP bildirim URL'si.")
    parser.add_argument("--url-file", action="append", type=Path, help="Satır satır URL içeren dosya.")
    parser.add_argument("--kap-days", type=int, default=30, help="KAP API için geriye dönük gün sayısı.")
    parser.add_argument("--kap-from-date", default=None, help="KAP başlangıç tarihi. Örnek: 2026-05-01")
    parser.add_argument("--kap-to-date", default=None, help="KAP bitiş tarihi. Örnek: 2026-06-01")
    parser.add_argument("--kap-disclosure-class", choices=["ALL", "ODA", "FR", "DG", "DUY"], default="ALL", help="KAP bildirim tipi filtresi.")
    return parser


def main() -> None:
    configure_console()
    ensure_project_dirs()
    args = build_parser().parse_args()
    aliases = load_aliases(args.aliases)
    tickers = list(args.tickers or [])
    for path in args.tickers_file or []:
        tickers.extend(read_ticker_file(path))
    tickers = normalize_tickers(tickers)

    if args.source == "kap-api":
        disclosures = fetch_kap_api(args)
        if args.limit:
            disclosures = disclosures[: args.limit]
        df = rows_from_kap_disclosures(disclosures, tickers)
        saved = save_news(df, args.out, append=args.append)
        print(f"{len(df)} yeni KAP satırı hazırlandı. Toplam {len(saved)} satır kaydedildi: {args.out}")
        return

    if args.source == "rss":
        items = fetch_rss_sources(args)
    elif args.source == "urls":
        items = fetch_url_sources(args, source_name="web")
    else:
        items = fetch_url_sources(args, source_name="KAP")

    if args.limit:
        items = items[: args.limit]
    df = rows_from_items(items, tickers, aliases, args.include_untagged, args.fetch_article_text)
    saved = save_news(df, args.out, append=args.append)
    print(f"{len(df)} yeni satır hazırlandı. Toplam {len(saved)} satır kaydedildi: {args.out}")


if __name__ == "__main__":
    main()
