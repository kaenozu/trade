"""Comprehensive tests for data services."""

import pytest
from unittest.mock import MagicMock, patch, Mock
import pandas as pd
import asyncio
from datetime import datetime, date

from app.services import data as data_service
from app.core.exceptions import DataError
from .utils import make_synthetic_ohlcv


class TestDataService:
    """Test the data service functions."""

    def test_validate_ticker_valid(self):
        """Test ticker validation with valid tickers."""
        assert data_service._validate_ticker("AAPL") == "AAPL"
        assert data_service._validate_ticker("7203.T") == "7203.T"
        assert data_service._validate_ticker("BRK.B") == "BRK.B"

    def test_validate_ticker_invalid(self):
        """Test ticker validation with invalid tickers."""
        with pytest.raises(DataError):
            data_service._validate_ticker("")
        
        with pytest.raises(DataError):
            data_service._validate_ticker("VERYLONGTICKERNAME")
        
        with pytest.raises(DataError):
            data_service._validate_ticker("TICK$R")

    @patch('app.services.data.yf')
    def test_fetch_ohlcv_success(self, mock_yf):
        """Test successful OHLCV fetch."""
        mock_yf.download.return_value = make_synthetic_ohlcv(100)
        
        result = data_service.fetch_ohlcv("AAPL", period_days=100)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert all(col in result.columns for col in ["Open", "High", "Low", "Close", "Volume"])

    @patch('app.services.data.yf')
    def test_fetch_ohlcv_empty_result(self, mock_yf):
        """Test OHLCV fetch with empty result."""
        mock_yf.download.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value.history.return_value = pd.DataFrame()
        
        # Should raise ValueError due to no data
        with pytest.raises(ValueError, match="No data returned for ticker"):
            data_service.fetch_ohlcv("INVALID")

    @patch('app.services.data.yf')
    def test_fetch_ohlcv_with_exception(self, mock_yf):
        """Test OHLCV fetch with yfinance exception."""
        mock_yf.download.side_effect = Exception("Network error")
        mock_yf.Ticker.return_value.history.side_effect = Exception("Network error")
        
        # Should fallback to synthetic if allowed or raise
        with pytest.raises(ValueError):
            data_service.fetch_ohlcv("AAPL")

    @patch('app.services.data._make_session')
    def test_fetch_last_close_direct_success(self, mock_session):
        """Test successful direct price fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "chart": {
                "result": [{
                    "timestamp": [1698076800],  # Example timestamp
                    "indicators": {
                        "quote": [{
                            "close": [150.25]
                        }]
                    }
                }]
            }
        }
        mock_session.return_value.get.return_value = mock_response
        
        price, asof = data_service.fetch_last_close_direct("AAPL")
        assert isinstance(price, float)
        assert isinstance(asof, str)
        assert price == 150.25

    @patch('app.services.data._make_session')
    def test_fetch_last_close_direct_failure(self, mock_session):
        """Test direct price fetch complete failure."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_session.return_value.get.return_value = mock_response
        
        with pytest.raises(ValueError, match="Failed to fetch quote"):
            data_service.fetch_last_close_direct("INVALID")

    def test_cache_path_valid(self):
        """Test cache path generation."""
        path = data_service._cache_path("AAPL", 100)
        assert "AAPL" in path
        assert "100d" in path

    def test_cache_path_sanitization(self):
        """Test cache path character sanitization."""
        # The implementation replaces / and \ with _ for safety
        result = data_service._cache_path("../../../etc/passwd", 100)
        # Should sanitize the path characters
        assert "_" in result  # Slashes should be replaced with underscores
        assert result.endswith("100d.csv")
        # Should still be within cache directory
        cache_dir = data_service.CACHE_DIR
        assert result.startswith(cache_dir)

    @patch('app.services.data.yf')
    def test_fetch_ohlcv_caching(self, mock_yf):
        """Test OHLCV fetch with caching."""
        mock_yf.download.return_value = make_synthetic_ohlcv(100)
        
        # Mock cache file operations
        with patch('app.services.data.os.path.exists', return_value=False):
            result1 = data_service.fetch_ohlcv("AAPL", period_days=100, ttl_seconds=3600)
            
        assert isinstance(result1, pd.DataFrame)
        assert len(result1) > 0


from app.services.async_data import AsyncDataService


class TestAsyncDataService:
    """Test the AsyncDataService class."""

    def setup_method(self):
        self.service = AsyncDataService()

    @pytest.mark.asyncio
    async def test_fetch_multiple_quotes_async_success(self):
        """Test successful async multiple quotes fetch."""
        # Mock the underlying fetch method
        async def mock_fetch_last_close(ticker):
            if ticker == "AAPL":
                return 150.0, "2023-10-26"
            elif ticker == "GOOGL":
                return 2800.0, "2023-10-26"
            else:
                raise ValueError("Invalid ticker")

        with patch.object(self.service, 'fetch_last_close_async', side_effect=mock_fetch_last_close):
            results = await self.service.fetch_multiple_quotes_async(["AAPL", "GOOGL", "INVALID"])
            
            assert "AAPL" in results
            assert "GOOGL" in results
            assert "INVALID" in results
            
            assert results["AAPL"] == (150.0, "2023-10-26")
            assert results["GOOGL"] == (2800.0, "2023-10-26")
            assert isinstance(results["INVALID"], Exception)

    @pytest.mark.asyncio
    async def test_fetch_last_close_async_success(self):
        """Test successful async quote fetch."""
        # Mock the data service function that's called under the hood
        with patch('app.services.data.fetch_last_close_direct', return_value=(150.0, "2023-10-26")):
            price, asof = await self.service.fetch_last_close_async("AAPL")
            assert price == 150.0
            assert asof == "2023-10-26"

    @pytest.mark.asyncio
    async def test_fetch_last_close_async_error(self):
        """Test async quote fetch with error."""
        with patch('app.services.data.fetch_last_close_direct', side_effect=ValueError("API error")):
            with pytest.raises(DataError, match="Quote fetch failed"):
                await self.service.fetch_last_close_async("INVALID")

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_async_success(self):
        """Test successful async OHLCV fetch."""
        mock_df = make_synthetic_ohlcv(30)
        with patch('app.services.data.fetch_ohlcv', return_value=mock_df):
            result = await self.service.fetch_ohlcv_async("AAPL", 30)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 30

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_async_empty(self):
        """Test async OHLCV fetch with empty result."""
        with patch('app.services.data.fetch_ohlcv', return_value=pd.DataFrame()):
            with pytest.raises(DataError, match="No data available"):
                await self.service.fetch_ohlcv_async("INVALID", 30)

    @pytest.mark.asyncio
    async def test_fetch_multiple_quotes_empty_list(self):
        """Test async multiple quotes fetch with empty ticker list."""
        results = await self.service.fetch_multiple_quotes_async([])
        assert results == {}

    @pytest.mark.asyncio
    async def test_fetch_multiple_quotes_concurrency(self):
        """Test that async multiple quotes uses concurrency."""
        call_count = 0
        
        async def mock_fetch_last_close(ticker):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate async work
            return 100.0, "2023-10-26"

        with patch.object(self.service, 'fetch_last_close_async', side_effect=mock_fetch_last_close):
            start_time = asyncio.get_event_loop().time()
            results = await self.service.fetch_multiple_quotes_async(["AAPL", "GOOGL", "MSFT"])
            end_time = asyncio.get_event_loop().time()
            
            # With concurrency, this should be much faster than 3 * 0.01 seconds
            assert end_time - start_time < 0.05
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_validate_ticker_async(self):
        """Test async ticker validation."""
        # Valid ticker
        result = await self.service.validate_ticker_async("AAPL")
        assert result == "AAPL"
        
        # Invalid tickers
        with pytest.raises(DataError):
            await self.service.validate_ticker_async("")
            
        with pytest.raises(DataError):
            await self.service.validate_ticker_async("VERYLONGTICKERNAME")

    @pytest.mark.asyncio
    async def test_get_data_summary_async_success(self):
        """Test successful data summary fetch."""
        mock_df = make_synthetic_ohlcv(30)
        
        with patch.object(self.service, 'fetch_last_close_async', return_value=(150.0, "2023-10-26")), \
             patch.object(self.service, 'fetch_ohlcv_async', return_value=mock_df):
            
            summary = await self.service.get_data_summary_async("AAPL")
            
            assert summary["ticker"] == "AAPL"
            assert summary["current_price"] == 150.0
            assert summary["price_date"] == "2023-10-26"
            assert summary["data_points"] == 30
            assert "statistics" in summary

    @pytest.mark.asyncio 
    async def test_get_data_summary_async_with_errors(self):
        """Test data summary with partial errors."""
        mock_df = make_synthetic_ohlcv(30)
        
        with patch.object(self.service, 'fetch_last_close_async', side_effect=DataError("Quote failed")), \
             patch.object(self.service, 'fetch_ohlcv_async', return_value=mock_df):
            
            summary = await self.service.get_data_summary_async("AAPL")
            
            assert summary["ticker"] == "AAPL"
            assert "quote_error" in summary
            assert summary["data_points"] == 30


class TestDataServiceIntegration:
    """Integration tests for data services."""

    def setup_method(self):
        self.async_service = AsyncDataService()

    def test_service_initialization(self):
        """Test that services initialize correctly."""
        assert self.async_service is not None

    @patch('app.services.data.yf')
    def test_fetch_ohlcv_date_range(self, mock_yf):
        """Test OHLCV fetch with specific date range."""
        mock_yf.download.return_value = make_synthetic_ohlcv(30)
        
        end_date = datetime.now().date()
        result = data_service.fetch_ohlcv("AAPL", period_days=100, end=end_date)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_async_data_integration(self):
        """Test async service integration."""
        # Test that async service can be used with mocked data
        with patch('app.services.data.fetch_last_close_direct', return_value=(100.0, "2023-10-26")):
            price, asof = await self.async_service.fetch_last_close_async("AAPL")
            assert price == 100.0
            assert asof == "2023-10-26"

    def test_cache_path_security(self):
        """Test cache path security validation."""
        # Valid path
        path = data_service._cache_path("AAPL", 30)
        assert "AAPL" in path
        
        # Test path security with path traversal attempt
        result = data_service._cache_path("../../../etc/passwd", 30)
        # Should be safely contained within cache directory
        cache_dir = data_service.CACHE_DIR
        assert result.startswith(cache_dir)
        # Characters should be sanitized
        assert "_" in result  # Should replace / and \ with underscores

    def test_ticker_validation_integration(self):
        """Test ticker validation across services."""
        # Valid tickers
        valid_tickers = ["AAPL", "GOOGL", "7203.T", "BRK.B"]
        for ticker in valid_tickers:
            assert data_service._validate_ticker(ticker) == ticker

        # Invalid tickers - check what exception is actually raised
        invalid_tickers = ["", "VERYLONGTICKERNAME", "TICK$R"]
        for ticker in invalid_tickers:
            try:
                data_service._validate_ticker(ticker)
                # If no exception, test failed
                assert False, f"Expected exception for invalid ticker: {ticker}"
            except (DataError, ValueError) as e:
                # Either DataError or ValueError is acceptable
                assert str(e)  # Just verify exception has a message