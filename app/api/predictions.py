"""ML prediction endpoints."""

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..models.api_models import PredictionPoint, PredictionRequest, PredictionResponse, TradePlan
from ..services import data as data_service
from ..services.features import build_feature_frame
from ..services.model import predict_future, train_or_load_model
from ..services.signals import generate_trade_plan

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict_stock(request: PredictionRequest) -> PredictionResponse:
    """Generate ML-based stock predictions and trading plan."""
    try:
        df = data_service.fetch_ohlcv(
            request.ticker, 
            period_days=request.lookback_days + 5
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if len(df) < 200:
        raise HTTPException(
            status_code=400, 
            detail="Not enough data to train model (minimum 200 days required)"
        )

    try:
        # Feature engineering
        features = build_feature_frame(df)
        
        # Train or load model
        model, meta = train_or_load_model(request.ticker, features)
        logger.info("Using model for %s with R2=%.3f", request.ticker, meta.r2_mean)
        
        # Generate predictions
        pred_df = predict_future(df, features, model, horizon_days=request.horizon_days)
        
        # Generate trading plan
        trade_plan_data = generate_trade_plan(pred_df)
        
        # Format response
        predictions = [
            PredictionPoint(
                date=str(pd.to_datetime(idx).date()),
                expected_return=float(row["expected_return"]),
                expected_price=float(row["expected_price"])
            )
            for idx, row in pred_df.iterrows()
        ]
        
        trade_plan = TradePlan(
            buy_date=trade_plan_data["buy_date"],
            sell_date=trade_plan_data["sell_date"],
            confidence=trade_plan_data["confidence"],
            rationale=trade_plan_data["rationale"]
        )
        
        return PredictionResponse(
            ticker=request.ticker,
            horizon_days=request.horizon_days,
            trade_plan=trade_plan,
            predictions=predictions
        )
        
    except Exception as e:
        logger.error("Prediction failed for %s: %s", request.ticker, e)
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        ) from e