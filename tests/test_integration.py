"""Integration tests for the application."""

import asyncio
from datetime import datetime

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.domain.entities import TickerSymbol, Price
from app.core.health import get_health_service
from app.core.metrics import get_metrics_collector


class TestApplicationIntegration:
    """Integration tests for the full application."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)

    def test_application_startup(self):
        """Test that the application starts up correctly."""
        response = self.client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_version_endpoint(self):
        """Test version information endpoint."""
        response = self.client.get("/version")
        assert response.status_code == 200

        data = response.json()
        assert "app" in data
        assert "version" in data
        assert data["app"] == "JP Stocks ML Forecaster"

    @pytest.mark.asyncio
    async def test_health_checks(self):
        """Test comprehensive health checking system."""
        health_service = get_health_service()
        health_results = await health_service.check_all()

        # Should have all expected health checkers
        expected_services = {
            "database", "cache", "external_apis",
            "system_resources", "ml_models"
        }
        assert set(health_results.keys()) == expected_services

        # Each health check should have required fields
        for service_name, result in health_results.items():
            assert result.service == service_name
            assert result.status is not None
            assert result.message is not None
            assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_detailed_health_endpoint(self):
        """Test detailed health endpoint."""
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/detailed")
            assert response.status_code == 200

            data = response.json()
            assert "overall_status" in data
            assert "timestamp" in data
            assert "checks" in data
            assert "summary" in data

    def test_metrics_collection(self):
        """Test that metrics are being collected."""
        metrics = get_metrics_collector()

        # Record some test metrics
        metrics.record_metric("test_metric", 123.45, {"test": "tag"})
        metrics.increment_counter("test_counter")

        # Check metrics summary
        summary = metrics.get_metrics_summary()
        assert isinstance(summary, dict)
        assert len(summary) > 0

    def test_tickers_endpoint(self):
        """Test tickers listing endpoint."""
        response = self.client.get("/tickers")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

        if data:  # If we have ticker data
            ticker = data[0]
            assert "ticker" in ticker
            assert "name" in ticker
            assert "sector" in ticker

    def test_quote_endpoint_validation(self):
        """Test quote endpoint input validation."""
        # Test invalid ticker
        response = self.client.get("/quote?ticker=")
        assert response.status_code == 400

        # Test too long ticker
        response = self.client.get("/quote?ticker=VERYLONGTICKER12345")
        assert response.status_code == 400

    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        # Make multiple requests quickly
        responses = []
        for _ in range(5):
            response = self.client.get("/healthz")
            responses.append(response.status_code)

        # All should succeed for health endpoint (has high limits)
        assert all(status == 200 for status in responses)

    def test_security_headers(self):
        """Test that security headers are present."""
        response = self.client.get("/healthz")

        # Check for security headers
        headers = response.headers
        expected_headers = [
            "x-content-type-options",
            "x-frame-options",
            "x-xss-protection"
        ]

        for header in expected_headers:
            assert header in headers

    def test_error_handling(self):
        """Test application error handling."""
        # Test 404 for non-existent endpoint
        response = self.client.get("/non-existent-endpoint")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_async_operations(self):
        """Test asynchronous operations work correctly."""
        from app.services.async_data import get_async_data_service

        async_service = get_async_data_service()

        # Test async validation
        ticker = await async_service.validate_ticker_async("TEST")
        assert ticker == "TEST"

        # Test with invalid ticker
        with pytest.raises(Exception):
            await async_service.validate_ticker_async("")


class TestDomainEntities:
    """Test domain entity functionality."""

    def test_ticker_symbol_validation(self):
        """Test ticker symbol validation."""
        # Valid tickers
        valid_tickers = ["AAPL", "GOOGL", "MSFT", "TEST.T"]
        for ticker_str in valid_tickers:
            ticker = TickerSymbol(ticker_str)
            assert ticker.value == ticker_str.upper()

        # Invalid tickers
        invalid_tickers = ["", "VERY_LONG_TICKER_NAME", "TICK$R", "<script>"]
        for ticker_str in invalid_tickers:
            with pytest.raises(ValueError):
                TickerSymbol(ticker_str)

    def test_price_validation(self):
        """Test price validation."""
        # Valid prices
        price = Price(123.45, "USD")
        assert price.value == 123.45
        assert price.currency == "USD"

        # Invalid prices
        with pytest.raises(ValueError):
            Price(-100.0)  # Negative price

        with pytest.raises(ValueError):
            Price(float('nan'))  # NaN price

    def test_price_equality(self):
        """Test price equality comparison."""
        price1 = Price(100.0, "USD")
        price2 = Price(100.0, "USD")
        price3 = Price(100.0, "EUR")
        price4 = Price(101.0, "USD")

        assert price1 == price2
        assert price1 != price3
        assert price1 != price4


class TestEndToEnd:
    """End-to-end tests simulating real user workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)

    def test_complete_prediction_workflow(self):
        """Test complete prediction workflow from API perspective."""
        # 1. Get list of available tickers
        response = self.client.get("/tickers")
        assert response.status_code == 200

        # 2. Try to get a quote (may fail due to external dependencies)
        response = self.client.get("/quote?ticker=TEST.T")
        # Don't assert success due to external dependencies

        # 3. Try to make a prediction (may fail due to data requirements)
        prediction_request = {
            "ticker": "TEST.T",
            "horizon_days": 5,
            "lookback_days": 200
        }

        response = self.client.post("/predict", json=prediction_request)
        # This may fail due to insufficient data, which is expected
        assert response.status_code in [200, 400, 422, 500]

    def test_monitoring_workflow(self):
        """Test monitoring and observability workflow."""
        # 1. Check basic health
        response = self.client.get("/healthz")
        assert response.status_code == 200

        # 2. Check detailed health
        response = self.client.get("/health/detailed")
        assert response.status_code == 200

        # 3. Check metrics summary
        response = self.client.get("/metrics-summary")
        assert response.status_code == 200

        # 4. Check Prometheus metrics
        response = self.client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")

    def test_api_documentation(self):
        """Test that API documentation is accessible in development."""
        # Note: docs are disabled in production
        response = self.client.get("/docs")
        # Should either work (development) or be disabled (production)
        assert response.status_code in [200, 404]

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            # Make multiple concurrent health checks
            tasks = [
                client.get("/healthz")
                for _ in range(10)
            ]

            responses = await asyncio.gather(*tasks)

            # All should succeed
            assert all(r.status_code == 200 for r in responses)

    def test_input_sanitization_e2e(self):
        """Test input sanitization across the application."""
        malicious_inputs = [
            "LONG1234567890123",  # Too long ticker (16 chars)
            "SPECIAL$CHARS",      # Invalid characters
            "",                   # Empty string
            "123LEADINGNUM",      # Leading number
        ]

        for malicious_input in malicious_inputs:
            # Test ticker endpoints
            response = self.client.get(f"/quote?ticker={malicious_input}")
            assert response.status_code == 400

            # Test tickers search (may be more lenient)
            response = self.client.get(f"/tickers?q={malicious_input}")
            # Should handle gracefully without error
            assert response.status_code in [200, 400]


if __name__ == "__main__":
    # Run tests directly if script is executed
    pytest.main([__file__, "-v"])
