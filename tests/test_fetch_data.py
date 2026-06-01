import pandas as pd
import pytest

from turkish_fin_bert.fetch_data import normalize_price_frame, resolve_tickers


def test_normalize_price_frame_flattens_yfinance_multiindex_columns():
    columns = pd.MultiIndex.from_tuples(
        [
            ("Open", "THYAO.IS"),
            ("High", "THYAO.IS"),
            ("Low", "THYAO.IS"),
            ("Close", "THYAO.IS"),
            ("Volume", "THYAO.IS"),
        ]
    )
    df = pd.DataFrame([[1, 2, 0.5, 1.5, 1000]], columns=columns)
    df.index = pd.to_datetime(["2026-05-26"])
    df.index.name = "Date"

    out = normalize_price_frame(df, "THYAO.IS")

    assert ["date", "open", "high", "low", "close", "volume", "ticker"] == list(out.columns)
    assert out.loc[0, "ticker"] == "THYAO"


def test_read_ticker_file_supports_comments_commas_and_suffixes(tmp_path):
    path = tmp_path / "tickers.txt"
    path.write_text("# yorum\nTHYAO, ASELS\nGARAN.IS\n", encoding="utf-8")

    assert resolve_tickers(None, [path]) == ["ASELS", "GARAN", "THYAO"]


def test_resolve_tickers_merges_cli_and_file_values(tmp_path):
    path = tmp_path / "tickers.txt"
    path.write_text("ASELS; GARAN\n", encoding="utf-8")

    assert resolve_tickers(["THYAO", "GARAN.IS"], [path]) == ["ASELS", "GARAN", "THYAO"]


def test_resolve_tickers_requires_at_least_one_value():
    with pytest.raises(ValueError, match="En az bir ticker"):
        resolve_tickers(None, None)
