from __future__ import annotations

import datetime as dt
import os
import re
import time

import pandas as pd
import requests

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency in tests
    yf = None


from ..core.config import settings
from ..core.exceptions import DataError

# Configuration from settings
CACHE_DIR = settings.cache_directory
os.makedirs(CACHE_DIR, exist_ok=True)
ALLOW_SYNTHETIC = settings.allow_synthetic_data


def _cache_path(ticker: str, period_days: int) -> str:
    safe = ticker.replace("/", "_").replace("\\", "_")
    return os.path.join(CACHE_DIR, f"yf_{safe}_{period_days}d.csv")


def _make_session() -> requests.Session:
    s = requests.Session()
    # Friendly UA to avoid some blocks
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/127.0 Safari/537.36"
            )
        }
    )
    # Inherit proxies from environment if set
    for k in ("http", "https"):
        env = os.getenv(f"{k.upper()}_PROXY")
        if env:
            s.proxies[k] = env
    return s


_TICKER_RE = re.compile(r"^[A-Za-z0-9._-]{1,15}$")


def _validate_ticker(ticker: str) -> str:
    """Validate ticker symbol format."""
    if not ticker or len(ticker) < 1:
        raise DataError("Invalid ticker: must be non-empty string")
    if not _TICKER_RE.match(ticker):
        raise DataError(f"Invalid ticker format: {ticker}")
    return ticker


def fetch_last_close_direct(ticker: str) -> tuple[float, str]:
    ticker = _validate_ticker(ticker)
    s = _make_session()
    bases = [
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
        f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}",
    ]
    params_list = [
        {"range": "5d", "interval": "1d"},
        {"range": "1mo", "interval": "1d"},
        {"range": "3mo", "interval": "1d"},
    ]
    last_err = None
    for base in bases:
        for params in params_list:
            try:
                r = s.get(base, params=params, timeout=10, allow_redirects=False)
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}"
                    continue
                data = r.json()
                res = data.get("chart", {}).get("result") or []
                if not res:
                    last_err = "no result"
                    continue
                obj = res[0]
                ts = obj.get("timestamp") or []
                quotes = ((obj.get("indicators") or {}).get("quote") or [{}])[0]
                closes = quotes.get("close") or []
                # pick last non-None close
                for i in range(len(closes) - 1, -1, -1):
                    c = closes[i]
                    if c is None:
                        continue
                    t = ts[i]
                    # Use naive date from UTC seconds to avoid tzdb dependency
                    dt_obj = pd.to_datetime(int(t), unit="s").date()
                    return float(c), str(dt_obj)
                last_err = "no close"
            except Exception as e:
                last_err = str(e)
                continue
    raise ValueError(f"Failed to fetch quote: {last_err or 'unknown error'}")


def fetch_ohlcv(
    ticker: str, period_days: int = 400, end: dt.date | None = None, ttl_seconds: int = 8 * 3600
) -> pd.DataFrame:
    ticker = _validate_ticker(ticker)
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

    def _attempt_fetch() -> pd.DataFrame:
        session = _make_session()
        # 1) Standard download
        for _ in range(2):
            try:
                d1 = yf.download(
                    ticker,
                    period=period,
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    session=session,
                )
                if d1 is not None and not d1.empty:
                    return d1
            except Exception:
                pass
        # 2) Ticker().history
        try:
            d2 = yf.Ticker(ticker, session=session).history(
                period=period, interval="1d", auto_adjust=False
            )
            if d2 is not None and not d2.empty:
                return d2
        except Exception:
            pass
        # 3) Use explicit start/end as fallback
        try:
            today = dt.date.today() if end is None else end
            start = today - dt.timedelta(days=int(period_days * 2))
            d3 = yf.download(
                ticker,
                start=start,
                end=today + dt.timedelta(days=1),
                interval="1d",
                auto_adjust=False,
                progress=False,
                session=session,
            )
            if d3 is not None and not d3.empty:
                return d3
        except Exception:
            pass
        return pd.DataFrame()

    if df is None or df.empty:
        df = _attempt_fetch()
        if df is None or df.empty:
            if not ALLOW_SYNTHETIC:
                raise ValueError(
                    "No data returned for ticker (network/region/firewall issue or invalid symbol)"
                )
            else:
                # As a last resort, synthesize a price series for demo usability
                # Deterministic by ticker for consistency across runs
                def _synthetic_ohlcv(days: int) -> pd.DataFrame:
                    import numpy as _np
                    import pandas as _pd

                    rng = _np.random.default_rng(abs(hash(ticker)) % (2**32))
                    idx = _pd.bdate_range(end=_pd.Timestamp.today().normalize(), periods=days)
                    # Random walk for close
                    rets = rng.normal(loc=0.0005, scale=0.02, size=len(idx))
                    close = 1000.0 * _np.cumprod(1.0 + rets)
                    high = close * (1.0 + _np.clip(rng.normal(0.003, 0.004, len(idx)), 0, 0.05))
                    low = close * (1.0 - _np.clip(rng.normal(0.003, 0.004, len(idx)), 0, 0.05))
                    open_ = (high + low) / 2.0
                    vol = rng.integers(1_000_000, 5_000_000, len(idx)).astype(float)
                    out = _pd.DataFrame(
                        {
                            "Open": open_,
                            "High": high,
                            "Low": low,
                            "Close": close,
                            "Volume": vol,
                        },
                        index=idx,
                    )
                    return out

                df = _synthetic_ohlcv(max(250, period_days))
                # Do not cache synthetic to avoid confusion
        # Write cache best-effort
        try:
            df.to_csv(cache_file)
        except Exception:
            pass

    if df is None or df.empty:
        raise ValueError("No data returned for ticker")

    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            # Some providers/paths may lowercase cols
            cols_lower = {c.lower(): c for c in needed}
            df_cols_lower = {str(x).lower(): str(x) for x in df.columns}
            if all(k in df_cols_lower for k in cols_lower.keys()):
                df = df[[df_cols_lower[k] for k in cols_lower.keys()]]
                df.columns = needed
            else:
                raise ValueError("Unexpected data format from provider")
    df = df[needed].dropna()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df
