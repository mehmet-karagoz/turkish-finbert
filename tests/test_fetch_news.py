from pathlib import Path

from turkish_fin_bert.fetch_news import detect_tickers, normalize_tickers, parse_feed_items, read_ticker_file, rows_from_items


def test_detect_tickers_matches_symbol_and_alias():
    aliases = {"THYAO": ["Türk Hava Yolları"], "ASELS": ["Aselsan"]}
    text = "Türk Hava Yolları güçlü trafik açıkladı, ASELS yeni sözleşme aldı."
    assert detect_tickers(text, ["THYAO", "ASELS", "GARAN"], aliases) == ["ASELS", "THYAO"]


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
