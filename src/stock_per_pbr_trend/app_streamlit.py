from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from stock_per_pbr_trend.analysis import (
    BreakoutConfig,
    add_per_pbr_columns,
    analyze_volume_breakout,
    calculate_price_confidence_intervals,
    calculate_ratio_confidence_intervals,
)
from stock_per_pbr_trend.charting import create_ci_chart, create_price_chart, create_ratio_chart


@dataclass(frozen=True)
class UiConfig:
    default_symbols: str = (
        "0050.TW\nSPY\nQQQ\nVT\nVWRA.L\nEWH\nMCHI\nINDA\nEWJ\n^STI\nJNK\nBSV\nBIV\nTLT\nVDE\nfxy"
    )


def _prepare_price_plot_df(df: pd.DataFrame, mas: list[int]) -> pd.DataFrame:
    out = df.copy()

    # Note: The original implementation used a fixed 182-day window for "3.5Y".
    # We keep that behavior for visual continuity.
    out["MA_3.5Y"] = out["Close"].rolling(window=182, min_periods=182).mean()

    for ma in mas:
        out[f"MA{ma}"] = out["Close"].rolling(window=ma, min_periods=ma).mean()

    plot_columns = ["Open", "High", "Low", "Close", "Volume", "MA_3.5Y", "MA_Volume_3M"]
    plot_columns.extend([f"MA{ma}" for ma in mas])
    return out[plot_columns]


def run() -> None:
    st.set_page_config(layout="wide", page_title="Stock PER/PBR Trend")
    st.title("📈 Stock Analysis Tool")

    ui = UiConfig()

    with st.sidebar:
        st.header("Settings")

        stock_input = st.text_area(
            "Stock symbols (one per line)",
            ui.default_symbols,
            help="Examples: 2330.TW, AAPL, MSFT. One symbol per line.",
        )

        st.subheader("BreakoutConfig")
        period = st.selectbox(
            "period",
            options=["6mo", "1y", "2y", "5y", "10y", "max"],
            index=3,
            help="Yahoo Finance history() period.",
        )
        avg_vol_period_days = st.number_input(
            "avg_vol_period_days",
            min_value=10,
            max_value=365,
            value=90,
            step=5,
        )
        std_dev_period_days = st.number_input(
            "std_dev_period_days",
            min_value=60,
            max_value=4000,
            value=int(252 * 3.5),
            step=10,
            help="Used for MA_3.5Y + regression window requirement.",
        )
        high_volume_threshold = st.number_input(
            "high_volume_threshold",
            min_value=1.0,
            max_value=10.0,
            value=1.4,
            step=0.1,
            format="%.1f",
        )
        lookback_days = st.number_input(
            "lookback_days",
            min_value=5,
            max_value=365,
            value=30,
            step=1,
        )
        min_points_floor = st.number_input(
            "min_points_floor",
            min_value=0,
            max_value=10000,
            value=500,
            step=50,
        )

        st.subheader("Charts")
        mas = st.multiselect(
            "Moving averages (days)",
            options=[5, 10, 20, 50, 100, 150, 200],
            default=[5, 10, 20, 50, 150, 200],
        )
        price_tail_days = st.number_input(
            "Price chart lookback (trading days)",
            min_value=252,
            max_value=252 * 20,
            value=252 * 5,
            step=252,
        )
        ci_window_weeks = st.number_input(
            "CI regression window (weeks)",
            min_value=26,
            max_value=520,
            value=182,
            step=13,
        )
        ci_tail_weeks = st.number_input(
            "CI chart lookback (weeks)",
            min_value=52,
            max_value=52 * 20,
            value=252 * 5 // 5,
            step=52,
        )

        st.subheader("Run")
        run_btn = st.button("Analyze Stocks", type="primary")

    cfg = BreakoutConfig(
        period=period,
        avg_vol_period_days=int(avg_vol_period_days),
        std_dev_period_days=int(std_dev_period_days),
        high_volume_threshold=float(high_volume_threshold),
        lookback_days=int(lookback_days),
        min_points_floor=int(min_points_floor),
    )

    if run_btn:
        stocks = [s.strip() for s in stock_input.split("\n") if s.strip()]

        with st.spinner("Analyzing stocks..."):
            for ticker in stocks:
                st.subheader(f"Analysis for {ticker}")

                try:
                    result = analyze_volume_breakout(ticker, config=cfg)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Error analyzing {ticker}: {exc}")
                    continue

                if result is None:
                    st.warning(f"No significant volume breakout found for {ticker}")
                    continue

                df = result.data.copy()
                df_weekly = calculate_price_confidence_intervals(df, window_size_weeks=int(ci_window_weeks))

                mas_sorted = sorted({int(x) for x in mas})
                df_plot = _prepare_price_plot_df(df, mas_sorted).tail(int(price_tail_days))

                ci_quantiles = [0.975, 0.875, 0.125, 0.025, 0.002, 0.999]
                ci_columns = ["Close", "regression"] + [f"CI_{q}" for q in ci_quantiles]
                df_weekly_plot = df_weekly[ci_columns].tail(int(ci_tail_weeks))

                fig_main = create_price_chart(
                    ticker=result.ticker,
                    df_plot=df_plot,
                    mas=mas_sorted,
                    recent_volumes=result.recent_volumes,
                )
                fig_ci = create_ci_chart(result.ticker, df_weekly_plot)

                st.plotly_chart(fig_main, use_container_width=True)
                st.plotly_chart(fig_ci, use_container_width=True)

                # PER/PBR section
                df_with_ratios = add_per_pbr_columns(result.ticker, df)
                if ("PER" in df_with_ratios.columns) and ("PBR" in df_with_ratios.columns):
                    weekly = df_with_ratios.resample("W").last().copy()
                    weekly["PER"] = weekly["PER"].ffill()
                    weekly["PBR"] = weekly["PBR"].ffill()

                    if weekly["PER"].notna().any() and weekly["PBR"].notna().any():
                        weekly = calculate_ratio_confidence_intervals(weekly, window_size_weeks=int(ci_window_weeks))
                        for ratio in ["PER", "PBR"]:
                            fig = create_ratio_chart(result.ticker, ratio, weekly)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No valid PER/PBR data available for {ticker}")
                else:
                    st.warning(
                        "PER/PBR are derived from Yahoo Finance snapshot fundamentals (trailingEps, bookValue) "
                        "and may be unavailable for some tickers."
                    )
