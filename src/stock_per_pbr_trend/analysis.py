from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


@dataclass(frozen=True)
class BreakoutConfig:
    period: str = "5y"
    avg_vol_period_days: int = 90
    std_dev_period_days: int = int(252 * 3.5)
    high_volume_threshold: float = 1.4
    lookback_days: int = 30
    min_points_floor: int = 500


@dataclass(frozen=True)
class BreakoutResult:
    ticker: str
    recent_high: float
    recent_low: float
    avg_volume_3m: float
    recent_volumes: list[float]
    data: pd.DataFrame


def fetch_price_history(ticker: str, period: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    if hist is None or hist.empty:
        return pd.DataFrame()
    return hist


def analyze_volume_breakout(
    ticker: str,
    config: BreakoutConfig = BreakoutConfig(),
) -> BreakoutResult | None:
    """Detect recent high-volume days relative to 3-month average volume.

    Returns a `BreakoutResult` when at least one "high volume" day exists in the recent window.
    """

    hist = fetch_price_history(ticker, period=config.period)
    if hist.empty:
        return None

    min_points = max(config.std_dev_period_days + config.avg_vol_period_days + config.lookback_days, config.min_points_floor)
    if len(hist) < min_points:
        return None

    hist = hist.copy()
    hist["MA_3.5Y"] = hist["Close"].rolling(
        window=int(config.std_dev_period_days), min_periods=int(config.std_dev_period_days)
    ).mean()
    hist["MA_Volume_3M"] = hist["Volume"].rolling(
        window=config.avg_vol_period_days, min_periods=config.avg_vol_period_days
    ).mean()

    recent = hist.iloc[-config.lookback_days :]
    recent_high = float(recent["High"].max())
    recent_low = float(recent["Low"].min())

    avg_3m_vol = float(hist["MA_Volume_3M"].iloc[-config.lookback_days:].mean())
    high_volume_days = recent[recent["Volume"] > avg_3m_vol * config.high_volume_threshold]
    if high_volume_days.empty:
        return None

    return BreakoutResult(
        ticker=ticker,
        recent_high=recent_high,
        recent_low=recent_low,
        avg_volume_3m=float(hist["MA_Volume_3M"].iloc[-1]),
        recent_volumes=[float(v) for v in high_volume_days["Volume"].tolist()],
        data=hist,
    )


def calculate_price_confidence_intervals(data: pd.DataFrame, window_size_weeks: int = 182) -> pd.DataFrame:
    """Weekly regression + confidence bands on price."""

    weekly = data.resample("W").last().copy()
    weekly["Close"] = weekly["Close"].ffill()

    def linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        if len(x) < 2:
            return float("nan"), float("nan")
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(m), float(c)

    weekly["regression"] = np.nan
    weekly["difference"] = np.nan

    for i in range(len(weekly)):
        if i < window_size_weeks - 1:
            y = weekly["Close"].iloc[: i + 1].to_numpy(dtype=float)
            x = np.arange(len(y), dtype=float)
            effective_window = len(y)
        else:
            y = weekly["Close"].iloc[i - window_size_weeks + 1 : i + 1].to_numpy(dtype=float)
            x = np.arange(window_size_weeks, dtype=float)
            effective_window = window_size_weeks

        if len(x) >= 2:
            slope, intercept = linear_regression(x, y)
            regression = slope * effective_window + intercept
            weekly.iloc[i, weekly.columns.get_loc("regression")] = regression
            weekly.iloc[i, weekly.columns.get_loc("difference")] = weekly["Close"].iloc[i] - regression

    weekly["rolling_std"] = weekly["difference"].rolling(window=window_size_weeks, min_periods=1).std()

    for q in [0.975, 0.875, 0.125, 0.025, 0.002, 0.999]:
        weekly[f"CI_{q}"] = norm.ppf(q, loc=weekly["regression"], scale=weekly["rolling_std"])

    return weekly


def add_per_pbr_columns(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """Compute PER/PBR from current EPS/book value (Yahoo Finance info snapshot).

    Notes:
    - This is *not* historical fundamentals; it applies current `trailingEps` and `bookValue`
      across the historical price series.
    """

    info = yf.Ticker(ticker).info
    trailing_eps = info.get("trailingEps")
    book_value = info.get("bookValue")
    if not trailing_eps or not book_value:
        return df

    out = df.copy()
    out["PER"] = out["Close"] / float(trailing_eps)
    out["PBR"] = out["Close"] / float(book_value)
    return out


def calculate_ratio_confidence_intervals(
    data: pd.DataFrame,
    window_size_weeks: int = 182,
    ratios: tuple[Literal["PER"], Literal["PBR"]] | tuple[str, ...] = ("PER", "PBR"),
) -> pd.DataFrame:
    weekly = data.copy()

    for ratio in ratios:
        weekly[f"{ratio}_regression"] = np.nan
        weekly[f"{ratio}_difference"] = np.nan

        for i in range(len(weekly)):
            if i < window_size_weeks - 1:
                y = weekly[ratio].iloc[: i + 1].to_numpy(dtype=float)
                x = np.arange(len(y), dtype=float) + 1.0
                effective_window = len(y)
            else:
                y = weekly[ratio].iloc[i - window_size_weeks + 1 : i + 1].to_numpy(dtype=float)
                x = np.arange(window_size_weeks, dtype=float) + 1.0
                effective_window = window_size_weeks

            if len(x) >= 2:
                A = np.vstack([x, np.ones(len(x))]).T
                slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
                regression = float(slope) * effective_window + float(intercept)
                weekly.iloc[i, weekly.columns.get_loc(f"{ratio}_regression")] = regression
                weekly.iloc[i, weekly.columns.get_loc(f"{ratio}_difference")] = weekly[ratio].iloc[i] - regression

        weekly[f"{ratio}_rolling_std"] = weekly[f"{ratio}_difference"].rolling(
            window=window_size_weeks, min_periods=1
        ).std()
        for q in [0.975, 0.875, 0.125, 0.025, 0.002, 0.999]:
            weekly[f"{ratio}_CI_{q}"] = norm.ppf(
                q, loc=weekly[f"{ratio}_regression"], scale=weekly[f"{ratio}_rolling_std"]
            )

    return weekly
