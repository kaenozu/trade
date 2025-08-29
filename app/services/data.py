from __future__ import annotations

import datetime as dt
from typing import Optional
import os
import time

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency in tests
    yf = None


CACHE_DIR = os.path.join(os.getcwd(), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(ticker: str, period_days: int) -> str:
    safe = ticker.replace("/", "_").replace("\\", "_")
    return os.path.join(CACHE_DIR, f"yf_{safe}_{period_days}d.csv")


def fetch_ohlcv(ticker: str, period_days: int = 400, end: Optional[dt.date] = None, ttl_seconds: int = 8*3600) -> pd.DataFrame:
    if not ticker or len(ticker) < 2:
        raise ValueError("Invalid ticker")
    if period_days < 60:
        raise ValueError("period_days must be >= 60")

    if yf is None:
        raise ValueError("yfinance not available in this environment")

    period = f"{int(period_days)}d"
    # Cache read
    cache_file = _cache_path(ticker, period_days)
    now = time.time()
    if os.path.exists(cache_file) and (now - os.path.getmtime(cache_file) <= ttl_seconds):
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    if df is None or df.empty:
        try:
            df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
        except Exception as e:
            raise ValueError(f"Failed to fetch data: {e}")
        # Write cache best-effort
        try:
            if df is not None and not df.empty:
                df.to_csv(cache_file)
        except Exception:
            pass

    if df is None or df.empty:
        raise ValueError("No data returned for ticker")

    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            raise ValueError("Unexpected data format from provider")
    df = df[needed].dropna()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df
