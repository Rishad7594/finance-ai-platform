# src/forecasting/forecaster.py
"""
Intraday stock price forecaster using multi-signal ensemble:
  - EWMA momentum (trend continuation)
  - Mean-reversion anchor (prevents runaway predictions)
  - Volatility-scaled random walk (realistic noise)

No heavy ML libraries — uses only numpy/pandas math.
"""

import os
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import requests

logger = logging.getLogger("forecaster")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)

PORT = os.getenv("PORT", "8000")
API_BASE = f"http://127.0.0.1:{PORT}"

# Map yfinance interval -> pandas freq used for resampling
INTERVAL_TO_PANDAS = {
    "1m": "1min",
    "2m": "2min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "60m": "60min",
    "90m": "90min",
    "1h": "60min",
    "4h": "4h",
    "1d": "1d",
}


def _get_sentiment_boost(ticker: str) -> float:
    """Fetch optional small sentiment boost from local /news-today summary (±3%)."""
    try:
        resp = requests.get(f"{API_BASE}/news-today", timeout=6)
        if resp.status_code != 200:
            return 0.0
        data = resp.json()
        summary = data.get("summary", {})
        stats = summary.get(ticker.upper(), {})
        pos = int(stats.get("positive", 0))
        neg = int(stats.get("negative", 0))
        if pos + neg == 0:
            return 0.0
        boost = (pos - neg) / (pos + neg) * 0.02
        return float(max(min(boost, 0.03), -0.03))
    except Exception:
        return 0.0


def _download_with_fallback(ticker: str, period: str, yf_intervals: list):
    """
    Try yf.download with a set of intervals. Return (df, used_interval) on success.
    """
    for interval in yf_intervals:
        try:
            logger.info("Trying yfinance: ticker=%s period=%s interval=%s", ticker, period, interval)
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if df is None or df.empty:
                logger.warning("yfinance returned empty for %s %s %s", ticker, period, interval)
                continue
            # Flatten multi-level columns (yfinance >= 0.2.18 returns MultiIndex)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Close" not in df.columns or df["Close"].dropna().empty:
                logger.warning("No Close values for %s with interval %s", ticker, interval)
                continue
            # Ensure Close is a 1D Series of floats
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            return df, interval
        except Exception as e:
            logger.warning("yfinance download failed for %s interval=%s: %s", ticker, interval, e)
            continue
    return None, None


def _ensemble_forecast(ts: pd.Series, steps: int) -> np.ndarray:
    """
    Multi-signal ensemble forecast:
      60% EWMA momentum  — captures recent trend direction
      20% mean-reversion  — anchors toward rolling average
      20% volatility walk  — adds realistic jitter

    Returns numpy array of `steps` predicted prices.
    """
    values = ts.values.astype(float)
    n = len(values)

    # --- Signal 1: EWMA Momentum (60%) ---
    # Compute EMA with span = min(20, half of data)
    ema_span = min(20, max(5, n // 2))
    ema = pd.Series(values).ewm(span=ema_span, adjust=False).mean().values

    # Slope: average of last 5 EMA differences (direction + magnitude)
    slope_window = min(5, len(ema) - 1)
    ema_diffs = np.diff(ema[-slope_window - 1:])
    avg_slope = float(np.mean(ema_diffs))

    # Decay the slope slightly over time (momentum fades)
    decay = 0.97
    momentum_forecast = np.zeros(steps)
    last_price = float(values[-1])
    for i in range(steps):
        last_price += avg_slope * (decay ** i)
        momentum_forecast[i] = last_price

    # --- Signal 2: Mean-Reversion Anchor (20%) ---
    # Pull toward rolling mean of last 60 candles
    rolling_window = min(60, n)
    rolling_mean = float(np.mean(values[-rolling_window:]))
    reversion_strength = 0.03  # 3% pull per step toward mean

    reversion_forecast = np.zeros(steps)
    rev_price = float(values[-1])
    for i in range(steps):
        rev_price += (rolling_mean - rev_price) * reversion_strength
        reversion_forecast[i] = rev_price

    # --- Signal 3: Volatility-Scaled Random Walk (20%) ---
    # Use actual candle-to-candle returns volatility
    returns = np.diff(values[-rolling_window:]) / values[-rolling_window:-1]
    returns = returns[np.isfinite(returns)]
    vol = float(np.std(returns)) if len(returns) > 1 else 0.001

    walk_forecast = np.zeros(steps)
    walk_price = float(values[-1])
    np.random.seed(int(datetime.now().timestamp()) % 2**31)
    for i in range(steps):
        # Random return drawn from actual volatility distribution
        random_return = np.random.normal(0, vol)
        walk_price *= (1 + random_return)
        walk_forecast[i] = walk_price

    # --- Blend: 60% momentum + 20% reversion + 20% walk ---
    ensemble = (0.60 * momentum_forecast +
                0.20 * reversion_forecast +
                0.20 * walk_forecast)

    return ensemble


def forecast_intraday(ticker: str, steps: int = 20, lookback_days: int = 30, preferred_interval: str = "30m"):
    """
    Forecast next `steps` periods (default 20 x 30min = 10 hours).
    Returns (forecast_list, history_list) where each element is {"time": "YYYY-MM-DD HH:MM", "price": float}.
    """
    if steps <= 0:
        raise ValueError("steps must be > 0")

    # Build candidate intervals (preferred first)
    candidates = [preferred_interval, "15m", "60m", "90m", "1h"]
    candidates = [c for c in candidates if c in INTERVAL_TO_PANDAS]

    df, used_interval = _download_with_fallback(ticker, f"{lookback_days}d", candidates)
    if df is None:
        raise RuntimeError(f"No intraday data found for {ticker} with intervals {candidates}")

    # Use Close series and resample to regular frequency
    pd_freq = INTERVAL_TO_PANDAS.get(used_interval, "30min")
    logger.info("Using interval=%s -> pandas freq=%s", used_interval, pd_freq)

    # Ensure datetime index is sorted, no duplicates
    df = df.sort_index()
    try:
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert(None)
    except Exception:
        pass

    # Resample to fixed frequency and forward-fill gaps
    try:
        ts = df["Close"].resample(pd_freq).last().ffill()
    except Exception:
        logger.warning("Resampling failed, falling back to raw Close series")
        ts = df["Close"].dropna()

    if ts.empty:
        raise RuntimeError(f"No close-price data available for {ticker} after resampling")

    # Use up to 200 data points for training (more context = better trend)
    ts_train = ts[-200:] if len(ts) > 200 else ts

    # --- Run multi-signal ensemble forecast ---
    forecast_values = _ensemble_forecast(ts_train, steps)

    # Optional sentiment adjustment
    try:
        boost = _get_sentiment_boost(ticker)
        if abs(boost) > 0:
            logger.info("Applying sentiment boost %.4f to forecast for %s", boost, ticker)
            forecast_values = forecast_values * (1.0 + boost)
    except Exception:
        pass

    # Prepare history: last 100 points with actual timestamps
    last_n = min(100, len(ts))
    hist_index = ts.index[-last_n:]
    history = [
        {"time": pd.Timestamp(t).strftime("%Y-%m-%d %H:%M"), "price": float(round(float(p), 2))}
        for t, p in zip(hist_index, ts.values[-last_n:])
    ]

    # Build forecast timestamps starting after last history point
    last_time = hist_index[-1]
    forecast_times = []
    if pd_freq.endswith("min"):
        minutes = int(pd_freq.replace("min", ""))
        for i in range(1, len(forecast_values) + 1):
            forecast_times.append(last_time + timedelta(minutes=minutes * i))
    elif pd_freq.endswith("h"):
        hours = int(pd_freq.replace("h", ""))
        for i in range(1, len(forecast_values) + 1):
            forecast_times.append(last_time + timedelta(hours=hours * i))
    elif pd_freq == "1d":
        for i in range(1, len(forecast_values) + 1):
            forecast_times.append(last_time + timedelta(days=i))
    else:
        for i in range(1, len(forecast_values) + 1):
            forecast_times.append(last_time + timedelta(minutes=30 * i))

    forecast = [
        {"time": pd.Timestamp(t).strftime("%Y-%m-%d %H:%M"), "price": float(round(float(p), 2))}
        for t, p in zip(forecast_times, forecast_values)
    ]

    return forecast, history


def forecast_multi(tickers, steps: int = 20, lookback_days: int = 30, preferred_interval: str = "30m"):
    out = {}
    for tk in tickers:
        try:
            f, h = forecast_intraday(tk, steps=steps, lookback_days=lookback_days, preferred_interval=preferred_interval)
            out[tk.upper()] = {"forecast": f, "history": h}
        except Exception as e:
            logger.error("Forecast failed for %s: %s", tk, e)
            out[tk.upper()] = {"error": str(e)}
    return out


if __name__ == "__main__":
    # quick local smoke test
    try:
        preds, hist = forecast_intraday("AAPL", steps=20, lookback_days=30)
        print("History sample (last 3):", hist[-3:])
        print("Forecast sample (first 5):", preds[:5])
    except Exception as ex:
        print("Local forecast test failed:", ex)
