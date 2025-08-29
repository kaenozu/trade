"""ML prediction endpoints."""

import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from ..core.security import rate_limit, validate_input
from ..core.services import ServiceContainer, get_container
from ..models.api_models import PredictionPoint, PredictionRequest, PredictionResponse, TradePlan
from ..services import data as data_sync_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
@rate_limit(max_requests=10, window_seconds=60)  # 10 predictions per minute
async def predict_stock(
    http_request: Request,
    request: PredictionRequest,
    container: ServiceContainer = Depends(get_container)
) -> PredictionResponse:
    """Generate ML-based stock predictions and trading plan."""
    logger.info("Processing prediction request for %s", request.ticker)

    data_service = container.get_data_service()
    feature_service = container.get_feature_service()
    model_service = container.get_model_service()
    signal_service = container.get_signal_service()

    try:
        df = None
        fetch_mode = "sync"
        # Prefer async path if available
        if hasattr(data_service, "fetch_ohlcv_async"):
            try:
                df = await data_service.fetch_ohlcv_async(
                    request.ticker,
                    period_days=request.lookback_days + 5,
                )
                if df is not None and not getattr(df, "empty", False):
                    fetch_mode = "async"
            except Exception as e:
                logger.debug("async fetch failed for %s: %s (will fallback)", request.ticker, e)

        # Fallback to legacy module function (enables test monkeypatching)
        if df is None or getattr(df, "empty", False):
            df = await run_in_threadpool(
                data_sync_service.fetch_ohlcv,
                request.ticker,
                request.lookback_days + 5,
            )
            fetch_mode = "sync"
    except Exception as e:
        logger.warning("Data fetch failed for %s: %s", request.ticker, e)
        raise HTTPException(status_code=400, detail=str(e)) from e

    if len(df) < 200:
        raise HTTPException(
            status_code=400,
            detail="Not enough data to train model (minimum 200 days required)"
        )

    try:
        # Feature engineering
        features = feature_service.build_features(df)

        # Train or load model
        model, meta = model_service.train_or_load_model(request.ticker, features)

        # Handle both dict and object types for meta
        if hasattr(meta, 'r2_mean'):
            r2_score = meta.r2_mean
        elif isinstance(meta, dict):
            r2_score = meta.get('r2_mean', 0.0)
        else:
            r2_score = 0.0

        logger.info("Using model for %s with R2=%.3f", request.ticker, r2_score)

        # Generate predictions
        pred_df = model_service.predict_future(df, features, model, horizon_days=request.horizon_days)

        # Generate trading plan
        trade_plan_data = signal_service.generate_trade_plan(pred_df)

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

        # Prepare metadata for response
        try:
            period_start = str(pd.to_datetime(df.index.min()).date())
            period_end = str(pd.to_datetime(df.index.max()).date())
        except Exception:
            period_start = None
            period_end = None

        # Normalize meta to dict-like
        features_list = []
        train_rows = 0
        r2_mean = 0.0
        if hasattr(meta, "features"):
            try:
                features_list = list(getattr(meta, "features"))
            except Exception:
                features_list = []
        if hasattr(meta, "train_rows"):
            try:
                train_rows = int(getattr(meta, "train_rows"))
            except Exception:
                train_rows = 0
        if hasattr(meta, "r2_mean"):
            try:
                r2_mean = float(getattr(meta, "r2_mean"))
            except Exception:
                r2_mean = 0.0
        if isinstance(meta, dict):
            features_list = meta.get("features", features_list) or features_list
            r2_mean = float(meta.get("r2_mean", r2_mean) or r2_mean)
            train_rows = int(meta.get("train_rows", train_rows) or train_rows)

        model_meta = {
            "features": features_list,
            "r2_mean": r2_mean,
            "train_rows": train_rows,
            "period_start": period_start,
            "period_end": period_end,
        }

        provider = None
        try:
            provider = getattr(df, "attrs", {}).get("data_source")  # type: ignore[attr-defined]
        except Exception:
            provider = None
        data_source = {
            "provider": provider or "unknown",
            "mode": fetch_mode,
            "rows": int(len(df)),
        }

        logger.info("Successfully generated %d predictions for %s", len(predictions), request.ticker)
        return PredictionResponse(
            ticker=request.ticker,
            horizon_days=request.horizon_days,
            trade_plan=trade_plan,
            predictions=predictions,
            model_meta=model_meta,
            data_source=data_source,
        )

    except Exception as e:
        logger.error("Prediction failed for %s: %s", request.ticker, e)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        ) from e
