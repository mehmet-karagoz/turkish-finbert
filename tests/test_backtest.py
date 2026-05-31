import pandas as pd

from turkish_fin_bert.backtest import _rebalance_dates


def test_rebalance_dates_supports_daily_mode():
    dates = pd.Series(pd.to_datetime(["2026-05-01", "2026-05-02", "2026-05-03"]))

    out = _rebalance_dates(dates, months=0)

    assert len(out) == 3
