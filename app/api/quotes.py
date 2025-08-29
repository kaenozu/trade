"""Stock quote endpoints."""

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..models.api_models import BulkQuotesResponse, Quote, QuoteItem
from ..services import data as data_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/quote", response_model=Quote)
def get_quote(ticker: str) -> Quote:
    """Get current quote for a single ticker."""
    # Basic input validation
    if not ticker or len(ticker) > 15:
        raise HTTPException(status_code=400, detail="Invalid ticker")
    
    try:
        price, asof = data_service.fetch_last_close_direct(ticker)
        return Quote(ticker=ticker, price=price, asof=asof)
    except Exception:
        # Fallback to OHLCV fetch
        try:
            df = data_service.fetch_ohlcv(ticker, period_days=90)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
            
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No data") from None
            
        last_idx = df.index.max()
        last_close = float(df.loc[last_idx, "Close"])
        return Quote(
            ticker=ticker, 
            price=last_close, 
            asof=str(pd.to_datetime(last_idx).date())
        )


@router.get("/quotes", response_model=BulkQuotesResponse) 
def get_bulk_quotes(tickers: str) -> BulkQuotesResponse:
    """Get quotes for multiple tickers (comma-separated)."""
    if not tickers:
        raise HTTPException(status_code=400, detail="tickers parameter is required")
    
    # Parse and deduplicate tickers
    raw_tickers = [t.strip() for t in tickers.split(",") if t.strip()]
    unique_tickers = []
    for ticker in raw_tickers:
        if ticker not in unique_tickers:
            unique_tickers.append(ticker)
        if len(unique_tickers) >= 300:  # Limit to prevent abuse
            break
    
    quotes: list[QuoteItem] = []
    
    for ticker in unique_tickers:
        try:
            price, asof = data_service.fetch_last_close_direct(ticker)
            quotes.append(QuoteItem(ticker=ticker, price=price, asof=asof))
        except Exception as e:
            logger.debug("Direct quote failed for %s: %s", ticker, e)
            # Fallback to OHLCV
            try:
                df = data_service.fetch_ohlcv(ticker, period_days=60)
                if len(df) > 0:
                    last_idx = df.index.max()
                    last_close = float(df.loc[last_idx, "Close"])
                    quotes.append(QuoteItem(
                        ticker=ticker,
                        price=last_close,
                        asof=str(pd.to_datetime(last_idx).date())
                    ))
                else:
                    quotes.append(QuoteItem(ticker=ticker, error=str(e)))
            except Exception as e2:
                logger.debug("Fallback quote failed for %s: %s", ticker, e2)
                quotes.append(QuoteItem(ticker=ticker, error=str(e2)))
    
    return BulkQuotesResponse(quotes=quotes)