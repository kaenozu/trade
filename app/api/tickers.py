"""Ticker-related endpoints."""

import logging

from fastapi import APIRouter, Depends

from ..core.services import ServiceContainer, get_container
from ..models.api_models import TickerInfo

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/tickers", response_model=list[TickerInfo])
async def get_tickers(
    q: str | None = None,
    container: ServiceContainer = Depends(get_container)
) -> list[TickerInfo]:
    """Get list of available Japanese stock tickers."""
    ticker_service = container.get_ticker_service()
    logger.debug("Fetching tickers with query: %s", q)
    
    raw_tickers = ticker_service.list_tickers(query=q)
    logger.info("Found %d tickers", len(raw_tickers))
    
    return [
        TickerInfo(
            ticker=ticker["ticker"],
            name=ticker["name"], 
            sector=ticker["sector"]
        )
        for ticker in raw_tickers
    ]