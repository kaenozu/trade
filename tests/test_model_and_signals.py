import pandas as pd

from app.services.features import build_feature_frame
from app.services.model import train_or_load_model, predict_future
from app.services.signals import generate_trade_plan
from .utils import make_synthetic_ohlcv


def test_train_predict_and_trade_plan():
    df = make_synthetic_ohlcv(450)
    feat = build_feature_frame(df)
    model, meta = train_or_load_model("TEST.T", feat)
    assert meta.train_rows > 200
    pred_df = predict_future(df, feat, model, horizon_days=10)
    assert isinstance(pred_df, pd.DataFrame)
    assert {"expected_return", "expected_price"}.issubset(pred_df.columns)
    assert len(pred_df) == 10
    trade = generate_trade_plan(pred_df)
    assert set(trade.keys()) == {"buy_date", "sell_date", "confidence", "rationale"}

