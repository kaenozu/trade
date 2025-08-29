from __future__ import annotations

import datetime as dt
import os
import re
import time
from urllib.parse import quote

import pandas as pd

try:
    import requests
except Exception:  # pragma: no cover - requests is expected to be available
    requests = None  # type: ignore

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency in tests
    yf = None


CACHE_DIR = os.path.join(os.getcwd(), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(ticker: str, period_days: int) -> str:
    safe = ticker.replace("/", "_").replace("\\", "_")
    p = os.path.join(CACHE_DIR, f"yf_{safe}_{period_days}d.csv")
    real = os.path.realpath(p)
    root = os.path.realpath(CACHE_DIR)
    if not real.startswith(root + os.sep):
        raise ValueError("Unsafe cache path")
    return real


_TICKER_RE = re.compile(r"^[A-Za-z0-9._-]{1,15}$")


def _validate_ticker(ticker: str) -> None:
    if not ticker or not _TICKER_RE.match(ticker):
        raise ValueError("Invalid ticker")


def _make_session() -> requests.Session:
    if requests is None:
        raise RuntimeError("requests is not available")
    s = requests.Session()
    ua = os.getenv(
        "YF_UA",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    )
    s.headers.update(
        {
            "User-Agent": ua,
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
        }
    )
    # requests picks up HTTP(S)_PROXY automatically from env if present
    return s


def fetch_last_close_direct(ticker: str, timeout: float = 5.0) -> tuple[float, str]:
    """Fetch last close via Yahoo chart API directly.

    Returns (price, asof_date_str)
    """
    _validate_ticker(ticker)
    s = _make_session()
    # Use short range for speed; 5d daily ensures at least one bar
    safe_ticker = quote(ticker, safe="A-Za-z0-9._-")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{safe_ticker}?interval=1d&range=5d"
    r = s.get(url, allow_redirects=False, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    result = (data or {}).get("chart", {}).get("result")
    if not result:
        raise ValueError("No result from Yahoo chart API")
    res0 = result[0]
    indicators = res0.get("indicators") or {}
    closes = None
    # Prefer adjusted close if present
    adj = indicators.get("adjclose")
    if isinstance(adj, list) and adj:
        closes = adj[0].get("adjclose")
    if not closes:
        q = indicators.get("quote")
        if isinstance(q, list) and q:
            closes = q[0].get("close")
    ts = res0.get("timestamp") or []
    if not closes or not ts:
        raise ValueError("Malformed chart payload")
    # pick latest non-null
    price = None
    asof = None
    for i in range(len(closes) - 1, -1, -1):
        c = closes[i]
        if c is not None:
            price = float(c)
            t = int(ts[i])
            asof = dt.datetime.utcfromtimestamp(t).date().isoformat()
            break
    if price is None or asof is None:
        raise ValueError("No valid close found")
    return price, asof


def fetch_ohlcv(
    ticker: str,
    period_days: int = 400,
    end: dt.date | None = None,
    ttl_seconds: int = 8 * 3600,
) -> pd.DataFrame:
    _validate_ticker(ticker)
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
        sess = None
        try:
            sess = _make_session()
        except Exception:
            sess = None
        try:
            df = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                session=sess,
            )
        except Exception as e:
            raise ValueError() from e
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

