import pandas as pd

from turkish_fin_bert.fetch_data import normalize_price_frame


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
