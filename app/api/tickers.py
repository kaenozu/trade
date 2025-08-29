"""Ticker-related endpoints."""


from fastapi import APIRouter

from ..models.api_models import TickerInfo
from ..services.tickers import list_jp_tickers

router = APIRouter()


@router.get("/tickers", response_model=list[TickerInfo])
def get_tickers(q: str | None = None) -> list[TickerInfo]:
    """Get list of available Japanese stock tickers."""
    raw_tickers = list_jp_tickers(query=q)
    return [
        TickerInfo(
            ticker=ticker["ticker"],
            name=ticker["name"], 
            sector=ticker["sector"]
        )
        for ticker in raw_tickers
    ]