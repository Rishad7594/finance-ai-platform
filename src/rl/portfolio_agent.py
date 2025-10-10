# src/rl/portfolio_agent.py

from __future__ import annotations
import math
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

# yfinance + scipy are light dependencies, but you may need to install them:
#   pip install yfinance scipy
import yfinance as yf
from scipy.optimize import minimize


def _fetch_prices(tickers: List[str], lookback: str = "1y") -> pd.DataFrame:
    """
    Download historical adjusted close prices for the given tickers.
    Returns a DataFrame with columns = tickers and rows = dates.
    """
    # yfinance handles multiple tickers in one go
    data = yf.download(tickers, period=lookback, auto_adjust=False, progress=False)
    if "Adj Close" in data:
        prices = data["Adj Close"].copy()
    else:
        # Fallback if provider shape changes
        prices = data["Close"].copy()

    # Ensure 2-D even for single ticker
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    # Standardize column order
    prices = prices.dropna(how="all").ffill().dropna()
    prices = prices.loc[:, [t for t in tickers if t in prices.columns]]
    return prices


def _mean_var_inputs(prices: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Convert price series to arithmetic simple returns.
    """
    rets = prices.pct_change().dropna(how="all")
    # Replace any residual inf/nan
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    mu = rets.mean()            # expected return per period
    Sigma = rets.cov()          # covariance
    return mu, Sigma


def _risk_aversion_from_target(target: float) -> float:
    """
    Map risk_target in [0,1] to a risk-aversion lambda (λ).
    - target=0.0 => very conservative => large λ
    - target=1.0 => very aggressive   => small λ
    We clamp to guardrails to keep the optimizer well-behaved.
    """
    t = float(np.clip(target, 0.0, 1.0))
    # Exponential mapping gives nicer spread than linear
    lam = 10 ** (2 - 1.8 * t)   #  ~[100 -> 1.58]
    return float(np.clip(lam, 1e-2, 1e3))


def _optimize_weights(mu: pd.Series, Sigma: pd.DataFrame, lam: float) -> np.ndarray:
    """
    Mean-variance optimization:
        maximize    mu^T w - lam * w^T Σ w
        subject to  sum(w) = 1,  w >= 0
    We solve by minimizing the negative of the objective.
    """
    n = len(mu)
    mu_v = mu.values
    Sigma_v = Sigma.values

    def objective(w: np.ndarray) -> float:
        # minimize => -(mu^T w - lam * w^T Σ w)
        return float(-mu_v @ w + lam * (w @ Sigma_v @ w))

    # Constraints: sum(w) = 1, bounds: 0 <= w <= 1
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n

    # Start from equal weights
    w0 = np.ones(n) / n

    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=[cons])
    if not res.success:
        # Fallback: equal weights
        return w0
    # Normalize in case of tiny numerical drift
    w = np.clip(res.x, 0, 1)
    w = w / w.sum()
    return w


def rebalance_opt(
    tickers: List[str],
    risk_target: float = 0.5,
    lookback: str = "1y",
) -> Dict[str, Any]:
    """
    Compute portfolio weights using mean-variance optimization.
    - tickers: list of symbols (e.g., ["AAPL", "MSFT", "TSLA"])
    - risk_target in [0,1]: 0 = conservative, 1 = aggressive
    - lookback: e.g., "6mo", "1y", "2y"
    Returns a JSON-serializable dict.
    """
    # Clean + dedupe tickers
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    tickers = list(dict.fromkeys(tickers))  # preserve order while deduping

    if len(tickers) == 0:
        return {"error": "No valid tickers provided."}

    prices = _fetch_prices(tickers, lookback=lookback)
    available = list(prices.columns)

    if len(available) == 0:
        return {"error": "No price data downloaded. Check tickers or internet connectivity."}

    if set(available) != set(tickers):
        missing = [t for t in tickers if t not in available]
        note = f"Missing data for: {missing}" if missing else None
    else:
        note = None

    mu, Sigma = _mean_var_inputs(prices)
    # Edge case: if covariance is singular (e.g., too few rows)
    if not np.isfinite(Sigma.values).all() or Sigma.shape[0] < 2:
        weights = np.ones(len(available)) / len(available)
        return {
            "tickers": available,
            "weights": weights.tolist(),
            "note": "Insufficient data for optimization; returned equal weights.",
        }

    lam = _risk_aversion_from_target(risk_target)
    weights = _optimize_weights(mu.loc[available], Sigma.loc[available, available], lam)

    # Diagnostics
    port_return = float(mu.loc[available].values @ weights)
    port_var = float(weights @ Sigma.loc[available, available].values @ weights)
    port_vol = float(math.sqrt(max(port_var, 0.0)))

    result = {
        "tickers": available,
        "weights": [float(w) for w in weights],
        "risk_target": float(np.clip(risk_target, 0, 1)),
        "lookback": lookback,
        "expected_return_per_period": port_return,
        "expected_vol_per_period": port_vol,
    }
    if note:
        result["note"] = note
    return result


# Backwards-compatible stub if someone still imports the old name
def rebalance_stub(tickers: List[str], risk_target: float = 0.5) -> Dict[str, Any]:
    n = len(tickers) if tickers else 0
    if n == 0:
        return {"tickers": [], "weights": []}
    w = [1.0 / n] * n
    return {"tickers": tickers, "weights": w, "note": "stub equal-weight; plug RL here"}
