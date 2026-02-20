import pandas as pd

from stock_per_pbr_trend.analysis import calculate_price_confidence_intervals


def test_calculate_price_confidence_intervals_smoke() -> None:
    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    df = pd.DataFrame(
        {
            "Close": range(400),
        },
        index=idx,
    )

    out = calculate_price_confidence_intervals(df)
    assert "regression" in out.columns
    assert out.index.freqstr in ("W-SUN", None)
