from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import ta
except Exception:  # pragma: no cover - optional dep in CI
    ta = None


def _ensure_ta():
    if ta is None:
        raise RuntimeError("ta library not available")


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 60:
        raise ValueError("Not enough rows to build features")

    _ensure_ta()

    out = pd.DataFrame(index=df.index.copy())
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"].astype(float)

    # Returns
    out["ret_1"] = close.pct_change()
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["logret_1"] = np.log1p(out["ret_1"]) 

    # Volatility
    out["vol_5"] = out["logret_1"].rolling(5).std()
    out["vol_20"] = out["logret_1"].rolling(20).std()

    # Moving averages
    out["sma_5"] = close.rolling(5).mean() / close - 1
    out["sma_20"] = close.rolling(20).mean() / close - 1
    out["sma_50"] = close.rolling(50).mean() / close - 1

    # RSI, MACD
    out["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi() / 100.0
    macd = ta.trend.MACD(close)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()

    # Volume features
    out["vol_z_20"] = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-9)

    # High/Low range
    out["hl_range"] = (high - low) / (close + 1e-9)

    # Target: next-day return
    target = close.pct_change().shift(-1)
    out["target"] = target

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out

