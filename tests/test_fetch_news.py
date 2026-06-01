from pathlib import Path

from turkish_fin_bert.fetch_news import (
    detect_tickers,
    normalize_for_match,
    normalize_tickers,
    parse_datetime,
    parse_feed_items,
    read_ticker_file,
    rows_from_items,
    rows_from_kap_disclosures,
    split_stock_codes,
)


def test_detect_tickers_matches_symbol_and_alias():
    aliases = {"THYAO": ["Türk Hava Yolları"], "ASELS": ["Aselsan"]}
    text = "Türk Hava Yolları güçlü trafik açıkladı, ASELS yeni sözleşme aldı."
    assert detect_tickers(text, ["THYAO", "ASELS", "GARAN"], aliases) == ["ASELS", "THYAO"]


def test_detect_tickers_is_turkish_character_tolerant():
    aliases = {"KCHOL": ["Koç Holding"], "AHGAZ": ["Ahlatcı Doğal Gaz"]}
    text = "Koc Holding kredi notu aldı. Ahlatci Dogal Gaz pay geri alımı yaptı."
    assert detect_tickers(text, ["KCHOL", "AHGAZ"], aliases) == ["AHGAZ", "KCHOL"]


def test_normalize_for_match_handles_turkish_letters():
    assert normalize_for_match("Şişecam, Koç ve Tüpraş") == "SISECAM KOC VE TUPRAS"


def test_parse_feed_items_from_sample_feed():
    feed = Path("data/raw/sample_feed.xml").read_text(encoding="utf-8")
    items = parse_feed_items(feed, source_name="sample")
    assert len(items) == 3
    assert items[0].title.startswith("THYAO")


def test_rows_from_items_creates_one_row_per_matched_ticker():
    feed = Path("data/raw/sample_feed.xml").read_text(encoding="utf-8")
    items = parse_feed_items(feed, source_name="sample")
    df = rows_from_items(items, ["THYAO", "ASELS"], aliases={}, include_untagged=False, fetch_article_text=False)
    assert df["ticker"].tolist() == ["THYAO", "ASELS"]
    assert set(["date", "ticker", "source", "title", "text", "url"]).issubset(df.columns)


def test_read_ticker_file_supports_lines_and_commas(tmp_path):
    path = tmp_path / "tickers.txt"
    path.write_text("# yorum\nTHYAO, ASELS\nGARAN.IS\n", encoding="utf-8")
    assert normalize_tickers(read_ticker_file(path)) == ["ASELS", "GARAN", "THYAO"]


def test_parse_datetime_uses_dayfirst_for_turkish_kap_dates():
    date, published_at = parse_datetime("01.06.2026 19:36:07")

    assert date == "2026-06-01"
    assert published_at.startswith("2026-06-01")


def test_rows_from_kap_disclosures_creates_rows_for_stock_codes():
    disclosures = [
        {
            "publishDate": "01.06.2026 19:36:07",
            "kapTitle": "ÇİMSA ÇİMENTO SANAYİ VE TİCARET A.Ş.",
            "subject": "Şirket Genel Bilgi Formu",
            "summary": "Yönetim kurulu bilgileri güncellendi.",
            "stockCodes": "CIMSA",
            "disclosureIndex": 1611693,
        }
    ]

    df = rows_from_kap_disclosures(disclosures, ["CIMSA", "THYAO"])

    assert df.loc[0, "date"] == "2026-06-01"
    assert df.loc[0, "ticker"] == "CIMSA"
    assert "Şirket Genel Bilgi Formu" in df.loc[0, "title"]
    assert df.loc[0, "url"].endswith("/1611693")


def test_split_stock_codes_normalizes_code_lists():
    assert split_stock_codes("THYAO, ASELS GARAN.IS") == ["ASELS", "GARAN", "THYAO"]
