"""Stock quote endpoints."""

import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from ..core.services import ServiceContainer, get_container
from ..models.api_models import BulkQuotesResponse, Quote, QuoteItem

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/quote", response_model=Quote)
async def get_quote(
    ticker: str,
    container: ServiceContainer = Depends(get_container)
) -> Quote:
    """Get current quote for a single ticker."""
    # Basic input validation
    if not ticker or len(ticker) > 15:
        raise HTTPException(status_code=400, detail="Invalid ticker")
    
    data_service = container.get_data_service()
    logger.debug("Fetching quote for %s", ticker)
    
    try:
        price, asof = data_service.fetch_last_close_direct(ticker)
        return Quote(ticker=ticker, price=price, asof=asof)
    except Exception as e:
        logger.debug("Direct quote failed for %s: %s", ticker, e)
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
async def get_bulk_quotes(
    tickers: str,
    container: ServiceContainer = Depends(get_container)
) -> BulkQuotesResponse:
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
    
    data_service = container.get_data_service()
    logger.info("Fetching bulk quotes for %d tickers", len(unique_tickers))
    
    quotes: list[QuoteItem] = []
    
    # Use async concurrent fetching for better performance
    try:
        quote_results = await data_service.fetch_multiple_quotes_async(unique_tickers)
        
        for ticker in unique_tickers:
            result = quote_results.get(ticker)
            
            if isinstance(result, Exception):
                logger.debug("Quote failed for %s: %s", ticker, result)
                quotes.append(QuoteItem(ticker=ticker, error=str(result)))
            else:
                price, asof = result
                quotes.append(QuoteItem(ticker=ticker, price=price, asof=asof))
                
    except Exception as e:
        logger.error("Bulk quote fetch failed: %s", e)
        # Fallback to individual synchronous fetches
        for ticker in unique_tickers:
            try:
                price, asof = data_service.fetch_last_close_direct(ticker)
                quotes.append(QuoteItem(ticker=ticker, price=price, asof=asof))
            except Exception as e:
                logger.debug("Direct quote failed for %s: %s", ticker, e)
                quotes.append(QuoteItem(ticker=ticker, error=str(e)))
    
    logger.info("Successfully fetched %d quotes", len([q for q in quotes if q.price is not None]))
    return BulkQuotesResponse(quotes=quotes)