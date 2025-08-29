"""API request and response models."""

from typing import Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for stock prediction."""
    ticker: str = Field(..., min_length=1, max_length=15, description="Stock ticker symbol")
    horizon_days: int = Field(10, ge=1, le=30, description="Prediction horizon in days")
    lookback_days: int = Field(400, ge=200, le=1000, description="Training data lookback period in days")


class PredictionPoint(BaseModel):
    """Individual prediction data point."""
    date: str = Field(..., description="Prediction date in YYYY-MM-DD format")
    expected_return: float = Field(..., description="Expected daily return")
    expected_price: float = Field(..., description="Expected stock price")


class TradePlan(BaseModel):
    """Trading recommendation based on predictions."""
    buy_date: Optional[str] = Field(None, description="Recommended buy date")
    sell_date: Optional[str] = Field(None, description="Recommended sell date")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    rationale: str = Field(..., description="Explanation of the trading plan")


class PredictionResponse(BaseModel):
    """Complete prediction response."""
    ticker: str = Field(..., description="Stock ticker symbol")
    horizon_days: int = Field(..., description="Prediction horizon in days")
    trade_plan: TradePlan = Field(..., description="Trading recommendation")
    predictions: list[PredictionPoint] = Field(..., description="Individual predictions")


class Quote(BaseModel):
    """Current stock quote."""
    ticker: str = Field(..., description="Stock ticker symbol")
    price: float = Field(..., description="Current stock price")
    asof: str = Field(..., description="Quote date in YYYY-MM-DD format")


class QuoteItem(BaseModel):
    """Individual quote item for bulk requests."""
    ticker: str = Field(..., description="Stock ticker symbol")
    price: Optional[float] = Field(None, description="Current stock price")
    asof: Optional[str] = Field(None, description="Quote date in YYYY-MM-DD format")
    error: Optional[str] = Field(None, description="Error message if quote failed")


class BulkQuotesResponse(BaseModel):
    """Response for bulk quote requests."""
    quotes: list[QuoteItem] = Field(..., description="List of quotes")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")


class VersionResponse(BaseModel):
    """Version information response."""
    app: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")
    git_sha: Optional[str] = Field(None, description="Git commit SHA")


class TickerInfo(BaseModel):
    """Ticker information."""
    ticker: str = Field(..., description="Stock ticker symbol")
    name: str = Field(..., description="Company name")
    sector: str = Field(..., description="Business sector")
