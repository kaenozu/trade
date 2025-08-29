from __future__ import annotations

import os
import json
from dataclasses import dataclass
import re
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score


MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


@dataclass
class ModelMeta:
    features: list
    r2_mean: float
    train_rows: int


def _split_Xy(feat: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = feat.drop(columns=["target"])  # type: ignore
    y = feat["target"]  # type: ignore
    return X, y


def _evaluate_cv(model: HistGradientBoostingRegressor, X: pd.DataFrame, y: pd.Series, splits: int = 5) -> float:
    tscv = TimeSeriesSplit(n_splits=min(splits, max(2, len(X) // 60)))
    scores = []
    for train_idx, test_idx in tscv.split(X):
        model_ = HistGradientBoostingRegressor(**model.get_params())
        model_.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model_.predict(X.iloc[test_idx])
        scores.append(r2_score(y.iloc[test_idx], pred))
    return float(np.nanmean(scores))


_TICKER_RE = re.compile(r"^[A-Za-z0-9._-]{1,15}$")


def _validate_ticker(ticker: str) -> str:
    if not ticker or not _TICKER_RE.match(ticker):
        raise ValueError("Invalid ticker")
    return ticker


def _model_path(ticker: str) -> str:
    _validate_ticker(ticker)
    base = ticker.replace("/", "_").replace("\\", "_")
    path = os.path.join(MODEL_DIR, f"{base}.joblib")
    real = os.path.realpath(path)
    root = os.path.realpath(MODEL_DIR)
    if not real.startswith(root + os.sep):
        raise ValueError("Unsafe model path")
    return real


def train_or_load_model(ticker: str, feat: pd.DataFrame):
    X, y = _split_Xy(feat)

    path = _model_path(ticker)
    if os.path.exists(path):
        try:
            saved = joblib.load(path)
            model = saved["model"]
            meta = saved["meta"]
            if set(meta.features) == set(X.columns):
                return model, meta
        except Exception:
            pass

    model = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=600, l2_regularization=0.0)
    r2 = _evaluate_cv(model, X, y)
    model.fit(X, y)

    meta = ModelMeta(features=list(X.columns), r2_mean=float(r2), train_rows=int(len(X)))
    joblib.dump({"model": model, "meta": meta}, path)
    return model, meta


def predict_future(df_price: pd.DataFrame, feat: pd.DataFrame, model: HistGradientBoostingRegressor, horizon_days: int = 10) -> pd.DataFrame:
    # Start from last available day
    last_date = df_price.index.max()
    X_last = feat.drop(columns=["target"]).iloc[-1]

    # We will iteratively predict returns and update a synthetic price series
    synthetic = df_price.copy()
    current_close = float(synthetic["Close"].iloc[-1])
    last_vol = float(synthetic["Volume"].iloc[-1])

    rows = []
    for i in range(1, horizon_days + 1):
        # Predict using the last known features
        pred_ret = float(model.predict(X_last.values.reshape(1, -1))[0])
        current_close = float(current_close * (1.0 + pred_ret))

        # Append a new day with naive OHLCV (close-only driven)
        new_date = last_date + pd.tseries.offsets.BDay(i)
        synthetic.loc[new_date, ["Open", "High", "Low", "Close", "Volume"]] = [
            current_close,
            current_close,
            current_close,
            current_close,
            last_vol,
        ]

        # Recompute features for the new end-of-series window
        from .features import build_feature_frame  # local import to avoid cycle

        feat_new = build_feature_frame(synthetic)
        X_last = feat_new.drop(columns=["target"]).iloc[-1]

        rows.append({
            "date": new_date.normalize(),
            "expected_return": pred_ret,
            "expected_price": current_close,
        })

    return pd.DataFrame(rows).set_index("date")

