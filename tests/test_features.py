from app.services.features import build_feature_frame
from .utils import make_synthetic_ohlcv


def test_build_feature_frame_shapes():
    df = make_synthetic_ohlcv(300)
    feat = build_feature_frame(df)
    assert "target" in feat.columns
    # Ensure we have a reasonable number of features
    assert feat.shape[1] >= 10
    # No NaNs
    assert feat.isna().sum().sum() == 0
    # Enough rows left after drops
    assert feat.shape[0] > 200

