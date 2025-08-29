"""Comprehensive tests for logging utilities."""

import asyncio
import logging
import time
from unittest.mock import MagicMock, patch, Mock

import pytest

from app.core.logging import (
    PerformanceLogger,
    setup_performance_logging,
    get_performance_logger,
    log_performance,
    log_async_performance
)


class TestPerformanceLogger:
    """Test the PerformanceLogger class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_logger = MagicMock(spec=logging.Logger)
        self.perf_logger = PerformanceLogger(self.mock_logger)

    def test_time_operation_success(self):
        """Test successful operation timing."""
        operation_name = "test_operation"
        context = {"key": "value"}
        
        with self.perf_logger.time_operation(operation_name, **context):
            time.sleep(0.001)  # Short delay
        
        # Should log start and completion
        assert self.mock_logger.info.call_count == 2
        
        # Check start log
        start_call = self.mock_logger.info.call_args_list[0]
        assert start_call[0][0] == "Starting operation: %s"
        assert start_call[0][1] == operation_name
        assert start_call[1]["extra"]["operation"] == operation_name
        assert start_call[1]["extra"]["key"] == "value"
        
        # Check completion log
        completion_call = self.mock_logger.info.call_args_list[1]
        assert "Operation completed: %s" in completion_call[0][0]
        assert completion_call[0][1] == operation_name
        assert completion_call[1]["extra"]["status"] == "success"
        assert "duration_seconds" in completion_call[1]["extra"]

    def test_time_operation_failure(self):
        """Test operation timing with exception."""
        operation_name = "failing_operation"
        
        with pytest.raises(ValueError):
            with self.perf_logger.time_operation(operation_name):
                raise ValueError("Test error")
        
        # Should log start and error
        assert self.mock_logger.info.call_count == 1  # Start only
        assert self.mock_logger.error.call_count == 1  # Error
        
        # Check error log
        error_call = self.mock_logger.error.call_args_list[0]
        assert "Operation failed: %s" in error_call[0][0]
        assert error_call[0][1] == operation_name
        assert error_call[1]["extra"]["status"] == "error"
        assert error_call[1]["extra"]["error"] == "Test error"

    def test_log_api_request(self):
        """Test API request logging."""
        endpoint = "/test/endpoint"
        method = "get"
        extra_data = {"user_id": "123"}
        
        self.perf_logger.log_api_request(endpoint, method, **extra_data)
        
        self.mock_logger.info.assert_called_once()
        call_args = self.mock_logger.info.call_args
        
        assert call_args[0][0] == "API request: %s %s"
        assert call_args[0][1] == "GET"
        assert call_args[0][2] == endpoint
        extra = call_args[1]["extra"]
        assert extra["endpoint"] == endpoint
        assert extra["method"] == "GET"
        assert extra["type"] == "api_request"
        assert extra["user_id"] == "123"

    def test_log_data_fetch(self):
        """Test data fetch logging."""
        ticker = "AAPL"
        data_type = "OHLCV"
        extra_data = {"period": "1d"}
        
        self.perf_logger.log_data_fetch(ticker, data_type, **extra_data)
        
        self.mock_logger.debug.assert_called_once()
        call_args = self.mock_logger.debug.call_args
        
        assert call_args[0][0] == "Fetching %s data for %s"
        assert call_args[0][1] == data_type
        assert call_args[0][2] == ticker
        extra = call_args[1]["extra"]
        assert extra["ticker"] == ticker
        assert extra["data_type"] == data_type
        assert extra["type"] == "data_fetch"
        assert extra["period"] == "1d"

    def test_log_cache_operation_hit(self):
        """Test cache operation logging with hit."""
        operation = "get"
        cache_key = "test_key"
        
        self.perf_logger.log_cache_operation(operation, cache_key, hit=True)
        
        self.mock_logger.debug.assert_called_once()
        call_args = self.mock_logger.debug.call_args
        
        assert "Cache %s: %s%s" in call_args[0][0]
        extra = call_args[1]["extra"]
        assert extra["cache_operation"] == operation
        assert extra["cache_key"] == cache_key
        assert extra["cache_hit"] is True
        assert extra["type"] == "cache_operation"

    def test_log_cache_operation_miss(self):
        """Test cache operation logging with miss."""
        operation = "get"
        cache_key = "test_key"
        
        self.perf_logger.log_cache_operation(operation, cache_key, hit=False)
        
        call_args = self.mock_logger.debug.call_args
        assert "Cache %s: %s%s" in call_args[0][0]
        assert call_args[1]["extra"]["cache_hit"] is False

    def test_log_cache_operation_no_hit_info(self):
        """Test cache operation logging without hit info."""
        operation = "set"
        cache_key = "test_key"
        
        self.perf_logger.log_cache_operation(operation, cache_key)
        
        call_args = self.mock_logger.debug.call_args
        assert "Cache %s: %s%s" in call_args[0][0]
        assert call_args[1]["extra"]["cache_hit"] is None

    def test_log_model_operation(self):
        """Test model operation logging."""
        operation = "prediction"
        ticker = "GOOGL"
        extra_data = {"model_type": "regression", "features": 10}
        
        self.perf_logger.log_model_operation(operation, ticker, **extra_data)
        
        self.mock_logger.info.assert_called_once()
        call_args = self.mock_logger.info.call_args
        
        assert call_args[0][0] == "Model %s for %s"
        assert call_args[0][1] == operation
        assert call_args[0][2] == ticker
        extra = call_args[1]["extra"]
        assert extra["model_operation"] == operation
        assert extra["ticker"] == ticker
        assert extra["type"] == "model_operation"
        assert extra["model_type"] == "regression"
        assert extra["features"] == 10


class TestLoggingSetup:
    """Test logging setup functions."""

    @patch('app.core.logging.logging')
    @patch('app.core.logging.settings')
    def test_setup_performance_logging_info_level(self, mock_settings, mock_logging):
        """Test logging setup with INFO level."""
        mock_settings.log_level = "INFO"
        mock_settings.debug = False
        
        mock_app_logger = MagicMock()
        mock_logging.getLogger.return_value = mock_app_logger
        
        setup_performance_logging()
        
        # Should configure basic logging
        mock_logging.basicConfig.assert_called_once()
        basic_config_call = mock_logging.basicConfig.call_args
        assert basic_config_call[1]["level"] == mock_logging.INFO
        assert "%(asctime)s" in basic_config_call[1]["format"]
        assert basic_config_call[1]["force"] is True

    @patch('app.core.logging.logging')
    @patch('app.core.logging.settings')
    def test_setup_performance_logging_debug_mode(self, mock_settings, mock_logging):
        """Test logging setup in debug mode."""
        mock_settings.log_level = "DEBUG"
        mock_settings.debug = True
        
        mock_loggers = {}
        def mock_get_logger(name):
            if name not in mock_loggers:
                mock_loggers[name] = MagicMock()
            return mock_loggers[name]
        
        mock_logging.getLogger.side_effect = mock_get_logger
        mock_logging.DEBUG = logging.DEBUG
        mock_logging.WARNING = logging.WARNING
        
        setup_performance_logging()
        
        # Should set debug level for specific loggers
        assert "app.services" in mock_loggers
        assert "app.api" in mock_loggers
        mock_loggers["app.services"].setLevel.assert_called_with(mock_logging.DEBUG)
        mock_loggers["app.api"].setLevel.assert_called_with(mock_logging.DEBUG)

    @patch('app.core.logging.logging')
    @patch('app.core.logging.settings')
    def test_setup_performance_logging_external_libraries(self, mock_settings, mock_logging):
        """Test that external library logging is reduced."""
        mock_settings.log_level = "INFO"
        mock_settings.debug = False
        
        mock_loggers = {}
        def mock_get_logger(name):
            if name not in mock_loggers:
                mock_loggers[name] = MagicMock()
            return mock_loggers[name]
        
        mock_logging.getLogger.side_effect = mock_get_logger
        mock_logging.WARNING = logging.WARNING
        
        setup_performance_logging()
        
        # Should reduce external library logging
        for lib in ["urllib3", "requests", "yfinance"]:
            assert lib in mock_loggers
            mock_loggers[lib].setLevel.assert_called_with(mock_logging.WARNING)

    def test_get_performance_logger(self):
        """Test getting a performance logger."""
        logger_name = "test.module"
        
        perf_logger = get_performance_logger(logger_name)
        
        assert isinstance(perf_logger, PerformanceLogger)
        assert perf_logger.logger.name == logger_name


class TestPerformanceDecorators:
    """Test performance logging decorators."""

    @patch('app.core.logging.get_performance_logger')
    def test_log_performance_decorator(self, mock_get_perf_logger):
        """Test sync performance decorator."""
        mock_perf_logger = MagicMock()
        mock_get_perf_logger.return_value = mock_perf_logger
        
        @log_performance("custom_operation")
        def test_function(x, y):
            return x + y
        
        result = test_function(1, 2)
        
        assert result == 3
        mock_get_perf_logger.assert_called_once()
        mock_perf_logger.time_operation.assert_called_once()
        
        # Check context manager call
        time_op_call = mock_perf_logger.time_operation.call_args
        assert time_op_call[0][0] == "custom_operation"
        assert time_op_call[1]["function"] == "test_function"

    @patch('app.core.logging.get_performance_logger')
    def test_log_performance_decorator_default_name(self, mock_get_perf_logger):
        """Test sync performance decorator with default operation name."""
        mock_perf_logger = MagicMock()
        mock_get_perf_logger.return_value = mock_perf_logger
        
        @log_performance()
        def another_function():
            return "test"
        
        result = another_function()
        
        assert result == "test"
        time_op_call = mock_perf_logger.time_operation.call_args
        assert time_op_call[0][0] == "another_function"

    @patch('app.core.logging.get_performance_logger')
    @pytest.mark.asyncio
    async def test_log_async_performance_decorator(self, mock_get_perf_logger):
        """Test async performance decorator."""
        mock_perf_logger = MagicMock()
        mock_get_perf_logger.return_value = mock_perf_logger
        
        @log_async_performance("async_operation")
        async def async_test_function(x, y):
            await asyncio.sleep(0.001)
            return x * y
        
        result = await async_test_function(3, 4)
        
        assert result == 12
        mock_get_perf_logger.assert_called_once()
        mock_perf_logger.time_operation.assert_called_once()
        
        # Check context manager call
        time_op_call = mock_perf_logger.time_operation.call_args
        assert time_op_call[0][0] == "async_operation"
        assert time_op_call[1]["function"] == "async_test_function"

    @patch('app.core.logging.get_performance_logger')
    @pytest.mark.asyncio
    async def test_log_async_performance_decorator_default_name(self, mock_get_perf_logger):
        """Test async performance decorator with default operation name."""
        mock_perf_logger = MagicMock()
        mock_get_perf_logger.return_value = mock_perf_logger
        
        @log_async_performance()
        async def another_async_function():
            return "async_test"
        
        result = await another_async_function()
        
        assert result == "async_test"
        time_op_call = mock_perf_logger.time_operation.call_args
        assert time_op_call[0][0] == "another_async_function"


class TestLoggingIntegration:
    """Integration tests for logging functionality."""

    def test_real_logger_integration(self):
        """Test with real logger instance."""
        # Create a real logger to test integration
        real_logger = logging.getLogger("test.integration")
        perf_logger = PerformanceLogger(real_logger)
        
        # This should not raise any exceptions
        with perf_logger.time_operation("integration_test"):
            time.sleep(0.001)
        
        perf_logger.log_api_request("/test", "POST", user="test_user")
        perf_logger.log_data_fetch("TEST", "quote", source="api")
        perf_logger.log_cache_operation("get", "test_key", hit=True)
        perf_logger.log_model_operation("train", "TEST", epochs=10)

    def test_decorator_exception_handling(self):
        """Test that decorators properly handle exceptions."""
        @log_performance("failing_operation")
        def failing_function():
            raise ValueError("Test failure")
        
        with pytest.raises(ValueError, match="Test failure"):
            failing_function()

    @pytest.mark.asyncio
    async def test_async_decorator_exception_handling(self):
        """Test that async decorators properly handle exceptions."""
        @log_async_performance("failing_async_operation")
        async def failing_async_function():
            raise RuntimeError("Async test failure")
        
        with pytest.raises(RuntimeError, match="Async test failure"):
            await failing_async_function()

    def test_logger_module_name_capture(self):
        """Test that decorators capture the correct module name."""
        # The decorator should use the function's module name
        @log_performance()
        def module_test_function():
            return "module_test"
        
        # This should work without issues
        result = module_test_function()
        assert result == "module_test"