import pandas as pd

from turkish_fin_bert.financial_effect import add_forward_returns


def test_add_forward_returns_uses_future_close_per_ticker():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "ticker": ["AAA", "AAA", "AAA"],
            "close": [100.0, 110.0, 121.0],
        }
    )
    result = add_forward_returns(prices)
    assert round(result.loc[0, "forward_return_1d"], 4) == 0.1
    assert pd.isna(result.loc[0, "forward_return_5d"])
