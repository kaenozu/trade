from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Geometric random walk
    rets = rng.normal(0.0005, 0.02, size=n)
    price = 100.0 * np.exp(np.cumsum(rets))
    close = pd.Series(price)
    high = close * (1 + rng.uniform(0.0, 0.01, size=n))
    low = close * (1 - rng.uniform(0.0, 0.01, size=n))
    open_ = close.shift(1).fillna(close.iloc[0])
    vol = rng.integers(1_000_000, 5_000_000, size=n)
    idx = pd.bdate_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "Open": open_.values,
        "High": high.values,
        "Low": low.values,
        "Close": close.values,
        "Volume": vol.astype(float),
    }, index=idx)
    return df

