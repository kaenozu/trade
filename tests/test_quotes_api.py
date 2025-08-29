"""Comprehensive tests for quotes API endpoints."""

import pytest
from unittest.mock import MagicMock
import pandas as pd
from httpx import AsyncClient, ASGITransport
from fastapi.testclient import TestClient

from app.main import app
from app.services import data as data_service
from app.services.async_data import get_async_data_service
from .utils import make_synthetic_ohlcv


class TestQuoteEndpoint:
    """Test the /quote endpoint."""

    def setup_method(self):
        self.client = TestClient(app)

    def test_quote_success_direct(self, monkeypatch):
        """Test quote endpoint with successful direct fetch."""
        def mock_fetch_direct(ticker: str, timeout: float = 5.0):
            return 1234.56, "2023-10-26"

        monkeypatch.setattr(data_service, "fetch_last_close_direct", mock_fetch_direct)

        response = self.client.get("/quote?ticker=AAPL")
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert data["price"] == 1234.56
        assert data["asof"] == "2023-10-26"

    def test_quote_fallback_to_ohlcv(self, monkeypatch):
        """Test quote endpoint fallback to OHLCV when direct fails."""
        def mock_fetch_direct_fail(ticker: str, timeout: float = 5.0):
            raise ValueError("Direct fetch failed")

        def mock_fetch_ohlcv(ticker: str, period_days: int = 90):
            return make_synthetic_ohlcv(5)

        monkeypatch.setattr(data_service, "fetch_last_close_direct", mock_fetch_direct_fail)
        monkeypatch.setattr(data_service, "fetch_ohlcv", mock_fetch_ohlcv)

        response = self.client.get("/quote?ticker=AAPL")
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert "price" in data
        assert "asof" in data

    def test_quote_no_data(self, monkeypatch):
        """Test quote endpoint when no data is available."""
        def mock_fetch_direct_fail(ticker: str, timeout: float = 5.0):
            raise ValueError("Direct fetch failed")

        def mock_fetch_ohlcv_empty(ticker: str, period_days: int = 90):
            return pd.DataFrame()  # Empty DataFrame

        monkeypatch.setattr(data_service, "fetch_last_close_direct", mock_fetch_direct_fail)
        monkeypatch.setattr(data_service, "fetch_ohlcv", mock_fetch_ohlcv_empty)

        response = self.client.get("/quote?ticker=INVALID")
        assert response.status_code == 400
        # Check response format - might be different error structure
        response_data = response.json()
        if "detail" in response_data:
            assert "No data" in response_data["detail"]
        else:
            # Alternative error format
            assert "error" in str(response_data) or "No data" in str(response_data)

    def test_quote_ohlcv_failure(self, monkeypatch):
        """Test quote endpoint when both direct and OHLCV fail."""
        def mock_fetch_direct_fail(ticker: str, timeout: float = 5.0):
            raise ValueError("Direct fetch failed")

        def mock_fetch_ohlcv_fail(ticker: str, period_days: int = 90):
            raise ValueError("OHLCV fetch failed")

        monkeypatch.setattr(data_service, "fetch_last_close_direct", mock_fetch_direct_fail)
        monkeypatch.setattr(data_service, "fetch_ohlcv", mock_fetch_ohlcv_fail)

        response = self.client.get("/quote?ticker=INVALID")
        assert response.status_code == 400
        # Check response format - might be different error structure
        response_data = response.json()
        if "detail" in response_data:
            assert "OHLCV fetch failed" in response_data["detail"]
        else:
            # Alternative error format
            assert "error" in str(response_data) or "OHLCV fetch failed" in str(response_data)

    def test_quote_empty_ticker(self):
        """Test quote endpoint with empty ticker."""
        response = self.client.get("/quote?ticker=")
        assert response.status_code == 400

    def test_quote_long_ticker(self):
        """Test quote endpoint with too long ticker."""
        response = self.client.get("/quote?ticker=VERYLONGTICKERNAME123456")
        assert response.status_code == 400


class TestBulkQuotesEndpoint:
    """Test the /quotes endpoint."""

    def setup_method(self):
        self.client = TestClient(app)

    @pytest.mark.asyncio
    async def test_bulk_quotes_success_async(self, monkeypatch):
        """Test bulk quotes with successful async fetch."""
        async def mock_fetch_multiple_async(tickers: list):
            return {
                "AAPL": (150.0, "2023-10-26"),
                "GOOGL": (2800.0, "2023-10-26"),
            }

        async_service = get_async_data_service()
        monkeypatch.setattr(async_service, "fetch_multiple_quotes_async", mock_fetch_multiple_async)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/quotes?tickers=AAPL,GOOGL")
            assert response.status_code == 200
            data = response.json()
            assert "quotes" in data
            quotes = data["quotes"]
            assert len(quotes) == 2

    @pytest.mark.asyncio
    async def test_bulk_quotes_with_errors_async(self, monkeypatch):
        """Test bulk quotes with some failures in async fetch."""
        async def mock_fetch_multiple_async(tickers: list):
            return {
                "AAPL": (150.0, "2023-10-26"),
                "INVALID": ValueError("Invalid ticker"),
            }

        async_service = get_async_data_service()
        monkeypatch.setattr(async_service, "fetch_multiple_quotes_async", mock_fetch_multiple_async)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/quotes?tickers=AAPL,INVALID")
            assert response.status_code == 200
            data = response.json()
            quotes = data["quotes"]
            assert len(quotes) == 2
            
            # Check successful quote
            aapl_quote = next((q for q in quotes if q["ticker"] == "AAPL"), None)
            assert aapl_quote is not None
            assert aapl_quote["price"] == 150.0
            
            # Check failed quote
            invalid_quote = next((q for q in quotes if q["ticker"] == "INVALID"), None)
            assert invalid_quote is not None
            assert invalid_quote["error"] is not None

    def test_bulk_quotes_fallback_to_sync(self, monkeypatch):
        """Test bulk quotes fallback to synchronous fetch."""
        async def mock_fetch_multiple_async_fail(tickers: list):
            raise Exception("Async fetch failed")

        def mock_fetch_direct_success(ticker: str, timeout: float = 5.0):
            if ticker == "AAPL":
                return 150.0, "2023-10-26"
            else:
                raise ValueError("Invalid ticker")

        async_service = get_async_data_service()
        monkeypatch.setattr(async_service, "fetch_multiple_quotes_async", mock_fetch_multiple_async_fail)
        monkeypatch.setattr(data_service, "fetch_last_close_direct", mock_fetch_direct_success)

        response = self.client.get("/quotes?tickers=AAPL,INVALID")
        assert response.status_code == 200
        data = response.json()
        quotes = data["quotes"]
        assert len(quotes) == 2

    def test_bulk_quotes_empty_tickers(self):
        """Test bulk quotes with empty tickers parameter."""
        response = self.client.get("/quotes?tickers=")
        assert response.status_code == 400
        response_data = response.json()
        if "detail" in response_data:
            assert "tickers parameter is required" in response_data["detail"]
        else:
            assert "tickers" in str(response_data) or "required" in str(response_data)

    def test_bulk_quotes_deduplication(self, monkeypatch):
        """Test bulk quotes deduplication of tickers."""
        def mock_fetch_direct(ticker: str, timeout: float = 5.0):
            return 100.0, "2023-10-26"

        async def mock_fetch_multiple_async_fail(tickers: list):
            raise Exception("Force fallback")

        async_service = get_async_data_service()
        monkeypatch.setattr(async_service, "fetch_multiple_quotes_async", mock_fetch_multiple_async_fail)
        monkeypatch.setattr(data_service, "fetch_last_close_direct", mock_fetch_direct)

        response = self.client.get("/quotes?tickers=AAPL,AAPL,GOOGL,AAPL")
        assert response.status_code == 200
        data = response.json()
        quotes = data["quotes"]
        # Should have only 2 unique tickers
        tickers = [q["ticker"] for q in quotes]
        assert len(set(tickers)) == 2

    def test_bulk_quotes_limit_protection(self, monkeypatch):
        """Test bulk quotes limit to prevent abuse."""
        def mock_fetch_direct(ticker: str, timeout: float = 5.0):
            return 100.0, "2023-10-26"

        async def mock_fetch_multiple_async_fail(tickers: list):
            raise Exception("Force fallback")

        async_service = get_async_data_service()
        monkeypatch.setattr(async_service, "fetch_multiple_quotes_async", mock_fetch_multiple_async_fail)
        monkeypatch.setattr(data_service, "fetch_last_close_direct", mock_fetch_direct)

        # Create 400 tickers (exceeds 300 limit)
        many_tickers = ",".join([f"TICK{i:03d}" for i in range(400)])
        response = self.client.get(f"/quotes?tickers={many_tickers}")
        assert response.status_code == 200
        data = response.json()
        quotes = data["quotes"]
        # Should be limited to 300
        assert len(quotes) == 300

    def test_bulk_quotes_whitespace_handling(self, monkeypatch):
        """Test bulk quotes with whitespace in tickers."""
        def mock_fetch_direct(ticker: str, timeout: float = 5.0):
            return 100.0, "2023-10-26"

        async def mock_fetch_multiple_async_fail(tickers: list):
            raise Exception("Force fallback")

        async_service = get_async_data_service()
        monkeypatch.setattr(async_service, "fetch_multiple_quotes_async", mock_fetch_multiple_async_fail)
        monkeypatch.setattr(data_service, "fetch_last_close_direct", mock_fetch_direct)

        response = self.client.get("/quotes?tickers= AAPL , GOOGL ,  MSFT  ")
        assert response.status_code == 200
        data = response.json()
        quotes = data["quotes"]
        assert len(quotes) == 3
        tickers = [q["ticker"] for q in quotes]
        assert "AAPL" in tickers
        assert "GOOGL" in tickers
        assert "MSFT" in tickers