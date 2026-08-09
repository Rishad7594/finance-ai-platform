# src/forecasting/forecaster.py
"""
Robust intraday forecaster (resampling + safe ARIMA fallbacks).

Key fixes:
- use yfinance intervals like "15m" (Yahoo format)
- resample feed to regular pandas freq (e.g., "15min") so statsmodels won't complain
- if pmdarima isn't usable due to binary incompatibility, fallback to statsmodels.ARIMA using `.values`
- return history timestamps from the feed (so charts show today's date)
- clearer logging & error messages
"""

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
            if "Close" not in df.columns or df["Close"].dropna().empty:
                logger.warning("No Close values for %s with interval %s", ticker, interval)
                continue
            return df, interval
        except Exception as e:
            logger.warning("yfinance download failed for %s interval=%s: %s", ticker, interval, e)
            continue
    return None, None


def forecast_intraday(ticker: str, steps: int = 40, lookback_days: int = 5, preferred_interval: str = "15m"):
    """
    Forecast next `steps` periods (default 40 -> 10 hours at 15min).
    Returns (forecast_list, history_list) where each list element is {"time": "YYYY-MM-DD HH:MM", "price": float}.
    """
    if steps <= 0:
        raise ValueError("steps must be > 0")

    # build candidate intervals (preferred first)
    candidates = [preferred_interval, "30m", "60m", "90m", "1h"]
    candidates = [c for c in candidates if c in INTERVAL_TO_PANDAS]

    df, used_interval = _download_with_fallback(ticker, f"{lookback_days}d", candidates)
    if df is None:
        raise RuntimeError(f"No intraday data found for {ticker} with intervals {candidates}")

    # use Close series and resample to regular frequency (pandas)
    pd_freq = INTERVAL_TO_PANDAS.get(used_interval, "15min")
    logger.info("Using interval=%s -> pandas freq=%s", used_interval, pd_freq)

    # ensure datetime index has no duplicate / is sorted
    df = df.sort_index()
    # if tz-aware, convert to naive to simplify formatting (keeps wall-clock time)
    try:
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert(None)
    except Exception:
        # some indexes don't support tz_convert; ignore
        pass

    # resample to fixed frequency and forward-fill any gaps
    try:
        if pd_freq == "1d":
            ts = df["Close"].resample(pd_freq).last().ffill()
        else:
            ts = df["Close"].resample(pd_freq).last().ffill()
    except Exception:
        # fallback: use raw Close values without resampling
        logger.warning("Resampling failed, falling back to raw Close series")
        ts = df["Close"].dropna()

    if ts.empty:
        raise RuntimeError(f"No close-price data available for {ticker} after resampling")

    # limit training size for speed
    ts_train = ts[-2000:] if len(ts) > 2000 else ts

    # Forecast values holder
    forecast_values = None

    # Try pmdarima.auto_arima first (import inside to avoid import-time failures)
    try:
        from pmdarima import auto_arima
        logger.info("Fitting pmdarima.auto_arima for %s", ticker)
        model = auto_arima(
            ts_train,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_order=8,
        )
        forecast_values = np.asarray(model.predict(n_periods=steps))
    except Exception as e:
        logger.warning("pmdarima unavailable or failed (%s); using statsmodels.ARIMA fallback", e)
        try:
            from statsmodels.tsa.arima.model import ARIMA
            # pass raw numeric values (avoid date-index issues)
            model = ARIMA(ts_train.values, order=(2, 1, 2))
            fitted = model.fit()
            forecast_values = np.asarray(fitted.forecast(steps=steps))
        except Exception as e2:
            logger.exception("ARIMA fallback failed: %s", e2)
            raise RuntimeError("Forecasting failed: no working ARIMA implementation") from e2

    # Add small realistic noise proportional to recent volatility
    try:
        vol = float(np.nanstd(ts_train.values))
        noise = np.random.normal(0, vol * 0.002, size=len(forecast_values))
        forecast_values = forecast_values + noise
    except Exception:
        pass

    # Optional sentiment adjustment
    try:
        boost = _get_sentiment_boost(ticker)
        if abs(boost) > 0:
            logger.info("Applying sentiment boost %.4f to forecast for %s", boost, ticker)
            forecast_values = forecast_values * (1.0 + boost)
    except Exception:
        pass

    # Prepare history times using actual index values (so chart shows today's timestamps)
    last_n = min(100, len(ts))
    hist_index = ts.index[-last_n:]
    history = [
        {"time": pd.Timestamp(t).strftime("%Y-%m-%d %H:%M"), "price": float(round(float(p), 2))}
        for t, p in zip(hist_index, ts.values[-last_n:])
    ]

    # Build forecast timestamps starting after last history timestamp
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
        # default 15 minutes
        for i in range(1, len(forecast_values) + 1):
            forecast_times.append(last_time + timedelta(minutes=15 * i))

    forecast = [
        {"time": pd.Timestamp(t).strftime("%Y-%m-%d %H:%M"), "price": float(round(float(p), 2))}
        for t, p in zip(forecast_times, forecast_values)
    ]

    return forecast, history


def forecast_multi(tickers, steps: int = 40, lookback_days: int = 5, preferred_interval: str = "15m"):
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
        preds, hist = forecast_intraday("AAPL", steps=16, lookback_days=3)
        print("History sample:", hist[-3:])
        print("Forecast sample:", preds[:5])
    except Exception as ex:
        print("Local forecast test failed:", ex)
