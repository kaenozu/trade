from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd


def _best_interval_max_product(returns: pd.Series):
    # Maximize product(1+r) -> maximize sum(log(1+r))
    logg = np.log1p(returns.clip(lower=-0.99))
    prefix = np.r_[0.0, logg.cumsum().values]
    best_gain = -1e18
    best_i = 0
    best_j = 0
    min_prefix = 0.0
    min_idx = 0
    for j in range(1, len(prefix)):
        val = prefix[j]
        gain = val - min_prefix
        if gain > best_gain:
            best_gain = gain
            best_i = min_idx
            best_j = j - 1
        if val < min_prefix:
            min_prefix = val
            min_idx = j
    return best_i, best_j, float(best_gain)


def generate_trade_plan(pred_df: pd.DataFrame) -> Dict:
    if pred_df.empty:
        return {
            "buy_date": None,
            "sell_date": None,
            "confidence": 0.0,
            "rationale": "No predictions available",
        }

    r = pred_df["expected_return"].astype(float)
    i, j, gain_log = _best_interval_max_product(r)
    if j < i:
        return {
            "buy_date": None,
            "sell_date": None,
            "confidence": 0.0,
            "rationale": "No positive interval in forecast horizon",
        }

    cum_prod = float(np.prod(1.0 + r.iloc[i : j + 1]))
    total_ret = cum_prod - 1.0

    # Confidence heuristic: map average absolute return and interval length
    avg_abs = float(r.abs().mean())
    length = int(j - i + 1)
    conf = 1.0 / (1.0 + math.exp(- (avg_abs * 10 + 0.1 * length)))
    conf = max(0.0, min(1.0, conf))

    return {
        "buy_date": str(pred_df.index[i].date()),
        "sell_date": str(pred_df.index[j].date()),
        "confidence": conf,
        "rationale": f"Expected cumulative return {(total_ret*100):.2f}% over {length} days",
    }

