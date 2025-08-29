"""Async data service with enhanced caching and error handling."""

import asyncio
import logging
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..core.cache import cached_async
from ..core.config import settings
from ..core.exceptions import DataError
from . import data

logger = logging.getLogger(__name__)


class AsyncDataService:
    """Async wrapper for data operations with caching and optimization."""

    def __init__(self):
        self.cache_duration = 300  # 5 minutes for quotes, longer for OHLCV

    @cached_async(max_age_seconds=1800)  # 30 minute cache for OHLCV
    async def fetch_ohlcv_async(self, ticker: str, period_days: int) -> pd.DataFrame:
        """Fetch OHLCV data asynchronously with caching."""
        logger.debug("Fetching OHLCV for %s (%d days)", ticker, period_days)
        
        # Run the synchronous function in a thread pool
        loop = asyncio.get_event_loop()
        try:
            df = await loop.run_in_executor(
                None, 
                data.fetch_ohlcv, 
                ticker, 
                period_days
            )
            
            if df.empty:
                raise DataError(f"No data available for ticker {ticker}")
                
            logger.debug("Fetched %d rows for %s", len(df), ticker)
            return df
            
        except Exception as e:
            logger.error("Failed to fetch OHLCV for %s: %s", ticker, e)
            raise DataError(f"Data fetch failed for {ticker}: {str(e)}") from e

    @cached_async(max_age_seconds=300)  # 5 minute cache for quotes
    async def fetch_last_close_async(self, ticker: str) -> tuple[float, str]:
        """Fetch last close price asynchronously with caching."""
        logger.debug("Fetching last close for %s", ticker)
        
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                data.fetch_last_close_direct,
                ticker
            )
            return result
            
        except Exception as e:
            logger.error("Failed to fetch quote for %s: %s", ticker, e)
            raise DataError(f"Quote fetch failed for {ticker}: {str(e)}") from e

    async def fetch_multiple_quotes_async(self, tickers: list[str]) -> dict[str, tuple[float, str] | Exception]:
        """Fetch multiple quotes concurrently."""
        logger.info("Fetching quotes for %d tickers concurrently", len(tickers))
        
        async def fetch_single_quote(ticker: str) -> tuple[str, tuple[float, str] | Exception]:
            try:
                result = await self.fetch_last_close_async(ticker)
                return ticker, result
            except Exception as e:
                return ticker, e

        # Create tasks for all tickers
        tasks = [fetch_single_quote(ticker) for ticker in tickers]
        
        # Execute concurrently with limit to avoid overwhelming the system
        semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
        
        async def limited_fetch(task):
            async with semaphore:
                return await task
        
        limited_tasks = [limited_fetch(task) for task in tasks]
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)
        
        # Process results
        quote_results = {}
        for result in results:
            if isinstance(result, tuple):
                ticker, quote_or_error = result
                quote_results[ticker] = quote_or_error
            else:
                # Handle cases where gather returned an exception
                logger.error("Unexpected result from concurrent fetch: %s", result)
        
        success_count = sum(1 for v in quote_results.values() if not isinstance(v, Exception))
        logger.info("Successfully fetched %d/%d quotes", success_count, len(tickers))
        
        return quote_results

    async def validate_ticker_async(self, ticker: str) -> str:
        """Validate ticker format asynchronously."""
        if not ticker or len(ticker) < 1:
            raise DataError("Invalid ticker: must be non-empty string")
        
        if len(ticker) > 15:
            raise DataError(f"Invalid ticker: too long ({len(ticker)} characters)")
        
        # Basic format validation
        ticker_pattern = re.compile(r"^[A-Za-z0-9._-]{1,15}$")
        if not ticker_pattern.match(ticker):
            raise DataError(f"Invalid ticker format: {ticker}")
        
        return ticker.upper()

    async def get_data_summary_async(self, ticker: str) -> dict:
        """Get comprehensive data summary for a ticker."""
        try:
            ticker = await self.validate_ticker_async(ticker)
            
            # Fetch both quote and basic OHLCV data concurrently
            quote_task = self.fetch_last_close_async(ticker)
            ohlcv_task = self.fetch_ohlcv_async(ticker, 30)  # Last 30 days
            
            quote_result, ohlcv_result = await asyncio.gather(
                quote_task, ohlcv_task, return_exceptions=True
            )
            
            summary = {"ticker": ticker}
            
            # Process quote result
            if isinstance(quote_result, Exception):
                summary["quote_error"] = str(quote_result)
            else:
                price, asof = quote_result
                summary["current_price"] = price
                summary["price_date"] = asof
            
            # Process OHLCV result
            if isinstance(ohlcv_result, Exception):
                summary["ohlcv_error"] = str(ohlcv_result)
            else:
                df = ohlcv_result
                summary["data_points"] = len(df)
                summary["date_range"] = {
                    "start": str(df.index.min().date()),
                    "end": str(df.index.max().date())
                }
                
                # Basic statistics
                if len(df) > 0:
                    recent_close = float(df["Close"].iloc[-1])
                    summary["statistics"] = {
                        "avg_volume": float(df["Volume"].mean()),
                        "volatility_30d": float(df["Close"].pct_change().std() * np.sqrt(252)),
                        "recent_close": recent_close
                    }
            
            return summary
            
        except Exception as e:
            logger.error("Failed to get data summary for %s: %s", ticker, e)
            raise DataError(f"Data summary failed for {ticker}: {str(e)}") from e


# Global async data service instance
_async_data_service: AsyncDataService | None = None


def get_async_data_service() -> AsyncDataService:
    """Get the global async data service instance."""
    global _async_data_service
    if _async_data_service is None:
        _async_data_service = AsyncDataService()
    return _async_data_service