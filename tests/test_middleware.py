"""Tests for middleware functionality."""

import logging
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.middleware import (
    RequestLoggingMiddleware,
    setup_cors,
    setup_logging,
    setup_prometheus,
    setup_request_logging,
    setup_sentry,
)


class TestSetupCors:
    """Test CORS setup functionality."""

    @patch("app.core.middleware.settings")
    @patch("app.core.middleware.logger")
    def test_setup_cors_success(self, mock_logger, mock_settings, caplog):
        """Test successful CORS setup."""
        mock_settings.cors_origins_list = ["http://localhost:3000"]
        app = FastAPI()
        
        setup_cors(app)
        
        assert len(app.user_middleware) > 0
        mock_logger.info.assert_called()

    @patch("app.core.middleware.settings")
    @patch("app.core.middleware.logger")
    def test_setup_cors_exception(self, mock_logger, mock_settings, caplog):
        """Test CORS setup with exception."""
        mock_settings.cors_origins_list = ["invalid-origin"]
        app = FastAPI()
        
        # Mock the add_middleware method to raise exception
        with patch.object(app, 'add_middleware', side_effect=Exception("CORS error")):
            setup_cors(app)
            
            mock_logger.warning.assert_called()


class TestSetupSentry:
    """Test Sentry setup functionality."""

    @patch("app.core.middleware.settings")
    def test_setup_sentry_disabled(self, mock_settings):
        """Test Sentry setup when disabled."""
        mock_settings.sentry_dsn = None
        
        setup_sentry()  # Should return early without error

    @patch("app.core.middleware.settings")
    @patch("app.core.middleware.logger")
    def test_setup_sentry_success(self, mock_logger, mock_settings, caplog):
        """Test successful Sentry setup."""
        mock_settings.sentry_dsn = "https://test@sentry.io/123"
        mock_settings.sentry_traces_sample_rate = 0.1
        mock_settings.sentry_profiles_sample_rate = 0.1
        mock_settings.sentry_env = "test"
        
        with patch("sentry_sdk.init") as mock_init:
            setup_sentry()
            
            mock_init.assert_called_once()
            mock_logger.info.assert_called()

    @patch("app.core.middleware.settings")
    def test_setup_sentry_import_error(self, mock_settings, caplog):
        """Test Sentry setup with import error."""
        mock_settings.sentry_dsn = "https://test@sentry.io/123"
        
        with patch("builtins.__import__", side_effect=ImportError):
            setup_sentry()
            
            assert "Sentry SDK not available" in caplog.text

    @patch("app.core.middleware.settings")
    @patch("app.core.middleware.logger")
    def test_setup_sentry_exception(self, mock_logger, mock_settings, caplog):
        """Test Sentry setup with exception."""
        mock_settings.sentry_dsn = "https://test@sentry.io/123"
        
        with patch("sentry_sdk.init", side_effect=Exception("Sentry error")):
            setup_sentry()
            
            mock_logger.warning.assert_called()


class TestSetupPrometheus:
    """Test Prometheus setup functionality."""

    @patch("app.core.middleware.settings")
    @patch("app.core.middleware.logger")
    def test_setup_prometheus_disabled(self, mock_logger, mock_settings, caplog):
        """Test Prometheus setup when disabled."""
        mock_settings.metrics_enabled = False
        app = FastAPI()
        
        setup_prometheus(app)
        
        mock_logger.info.assert_called()

    @patch("app.core.middleware.settings")
    @patch("app.core.middleware.logger")
    def test_setup_prometheus_success(self, mock_logger, mock_settings, caplog):
        """Test successful Prometheus setup."""
        mock_settings.metrics_enabled = True
        app = FastAPI()
        
        with patch("builtins.__import__") as mock_import:
            mock_module = Mock()
            mock_instrumentator = Mock()
            mock_instrumentator.instrument.return_value = mock_instrumentator
            mock_instrumentator.expose.return_value = mock_instrumentator
            mock_module.Instrumentator.return_value = mock_instrumentator
            mock_import.return_value = mock_module
            
            setup_prometheus(app)
            
            mock_instrumentator.instrument.assert_called_once_with(app)
            mock_instrumentator.expose.assert_called_once_with(app, include_in_schema=False)
            mock_logger.info.assert_called()

    @patch("app.core.middleware.settings")
    def test_setup_prometheus_import_error(self, mock_settings, caplog):
        """Test Prometheus setup with import error."""
        mock_settings.metrics_enabled = True
        app = FastAPI()
        
        with patch("builtins.__import__", side_effect=ImportError):
            setup_prometheus(app)
            
            assert "Prometheus instrumentator not available" in caplog.text

    @patch("app.core.middleware.settings")
    @patch("app.core.middleware.logger")
    def test_setup_prometheus_exception(self, mock_logger, mock_settings, caplog):
        """Test Prometheus setup with exception."""
        mock_settings.metrics_enabled = True
        app = FastAPI()
        
        with patch("builtins.__import__") as mock_import:
            mock_import.return_value.Instrumentator.side_effect = Exception("Prometheus error")
            
            setup_prometheus(app)
            
            mock_logger.warning.assert_called()


class TestRequestLoggingMiddleware:
    """Test RequestLoggingMiddleware functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.app = FastAPI()
        self.middleware = RequestLoggingMiddleware(self.app)

    @pytest.fixture
    def mock_request(self):
        """Create a mock request."""
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "GET"
        request.client.host = "127.0.0.1"
        request.headers = {"user-agent": "test-agent"}
        return request

    @pytest.fixture
    def mock_response(self):
        """Create a mock response."""
        response = Mock(spec=Response)
        response.status_code = 200
        return response

    @pytest.mark.asyncio
    @patch("app.core.middleware.get_performance_logger")
    async def test_dispatch_success(self, mock_get_logger, mock_request, mock_response):
        """Test successful request processing."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        middleware = RequestLoggingMiddleware(self.app)
        
        async def mock_call_next(request):
            return mock_response
        
        with patch("time.time", side_effect=[1000.0, 1000.5, 1000.5]):
            result = await middleware.dispatch(mock_request, mock_call_next)
        
        mock_logger.log_api_request.assert_called_once_with(
            endpoint="/test",
            method="GET",
            client_ip="127.0.0.1",
            user_agent="test-agent"
        )
        assert result == mock_response

    @pytest.mark.asyncio
    @patch("app.core.middleware.get_performance_logger")
    async def test_dispatch_with_no_client(self, mock_get_logger, mock_response):
        """Test request processing with no client info."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        middleware = RequestLoggingMiddleware(self.app)
        
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "GET"
        request.client = None
        request.headers = {}
        
        async def mock_call_next(request):
            return mock_response
        
        with patch("time.time", side_effect=[1000.0, 1000.5, 1000.5]):
            await middleware.dispatch(request, mock_call_next)
        
        mock_logger.log_api_request.assert_called_once_with(
            endpoint="/test",
            method="GET",
            client_ip="unknown",
            user_agent="unknown"
        )

    @pytest.mark.asyncio
    @patch("app.core.middleware.get_performance_logger")
    @patch("app.core.middleware.logging.getLogger")
    async def test_dispatch_exception(self, mock_get_std_logger, mock_get_logger, mock_request):
        """Test request processing with exception."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_std_logger = Mock()
        mock_get_std_logger.return_value = mock_std_logger
        middleware = RequestLoggingMiddleware(self.app)
        
        async def mock_call_next(request):
            raise ValueError("Test error")
        
        with patch("time.time", side_effect=[1000.0, 1000.5]):
            with pytest.raises(ValueError):
                await middleware.dispatch(mock_request, mock_call_next)
        
        mock_std_logger.error.assert_called_once()
        args = mock_std_logger.error.call_args[0]
        assert "Request failed" in args[0]
        assert "GET" in args
        assert "/test" in args

    @pytest.mark.asyncio
    @patch("app.core.middleware.get_performance_logger")
    @patch("app.core.middleware.logging.getLogger")
    async def test_dispatch_logs_response_details(self, mock_get_std_logger, mock_get_logger, mock_request, mock_response):
        """Test that response details are properly logged."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_std_logger = Mock()
        mock_get_std_logger.return_value = mock_std_logger
        middleware = RequestLoggingMiddleware(self.app)
        
        async def mock_call_next(request):
            return mock_response
        
        with patch("time.time", side_effect=[1000.0, 1000.5]):
            await middleware.dispatch(mock_request, mock_call_next)
        
        mock_std_logger.info.assert_called_once()
        call_args = mock_std_logger.info.call_args
        args, kwargs = call_args
        
        assert "Request completed" in args[0]
        assert "GET" in args
        assert "/test" in args
        assert 200 in args
        
        extra = kwargs.get("extra", {})
        assert extra["method"] == "GET"
        assert extra["path"] == "/test"
        assert extra["status_code"] == 200
        assert "duration_seconds" in extra
        assert extra["type"] == "http_response"


class TestSetupFunctions:
    """Test setup utility functions."""

    def test_setup_logging(self):
        """Test logging setup."""
        with patch("app.core.logging.setup_performance_logging") as mock_setup_performance:
            setup_logging()
            mock_setup_performance.assert_called_once()

    def test_setup_request_logging(self):
        """Test request logging middleware setup."""
        app = FastAPI()
        
        setup_request_logging(app)
        
        # Check that middleware was added
        assert len(app.user_middleware) > 0
        middleware_class = app.user_middleware[0].cls
        assert issubclass(middleware_class, BaseHTTPMiddleware)


class TestIntegration:
    """Test middleware integration with FastAPI."""

    def test_request_logging_middleware_integration(self):
        """Test RequestLoggingMiddleware integration with FastAPI."""
        app = FastAPI()
        
        @app.get("/test")
        def test_endpoint():
            return {"message": "test"}
        
        setup_request_logging(app)
        
        with patch("app.core.middleware.get_performance_logger"):
            client = TestClient(app)
            response = client.get("/test")
            
            assert response.status_code == 200
            assert response.json() == {"message": "test"}

    def test_multiple_middleware_setup(self):
        """Test setting up multiple middleware components."""
        app = FastAPI()
        
        with patch("app.core.middleware.settings") as mock_settings:
            mock_settings.cors_origins_list = ["http://localhost:3000"]
            mock_settings.metrics_enabled = False
            mock_settings.sentry_dsn = None
            
            setup_cors(app)
            setup_prometheus(app)
            setup_sentry()
            setup_request_logging(app)
            
            # Should have CORS and RequestLogging middleware
            assert len(app.user_middleware) >= 1