"""Enhanced logging utilities with structured logging and performance monitoring."""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, TypeVar

from .config import settings

F = TypeVar('F', bound=Callable[..., Any])


class PerformanceLogger:
    """Logger for performance monitoring and structured logging."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @contextmanager
    def time_operation(self, operation: str, **context):
        """Context manager to time operations and log performance."""
        start_time = time.time()
        extra_context = {"operation": operation, **context}

        self.logger.info("Starting operation: %s", operation, extra=extra_context)

        try:
            yield
            duration = time.time() - start_time
            self.logger.info(
                "Operation completed: %s (%.3fs)",
                operation,
                duration,
                extra={**extra_context, "duration_seconds": duration, "status": "success"}
            )
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                "Operation failed: %s (%.3fs) - %s",
                operation,
                duration,
                str(e),
                extra={**extra_context, "duration_seconds": duration, "status": "error", "error": str(e)}
            )
            raise

    def log_api_request(self, endpoint: str, method: str, **kwargs):
        """Log API request details."""
        self.logger.info(
            "API request: %s %s",
            method.upper(),
            endpoint,
            extra={
                "endpoint": endpoint,
                "method": method.upper(),
                "type": "api_request",
                **kwargs
            }
        )

    def log_data_fetch(self, ticker: str, data_type: str, **kwargs):
        """Log data fetch operations."""
        self.logger.debug(
            "Fetching %s data for %s",
            data_type,
            ticker,
            extra={
                "ticker": ticker,
                "data_type": data_type,
                "type": "data_fetch",
                **kwargs
            }
        )

    def log_cache_operation(self, operation: str, cache_key: str, hit: bool = None, **kwargs):
        """Log cache operations."""
        self.logger.debug(
            "Cache %s: %s%s",
            operation,
            cache_key,
            f" ({'hit' if hit else 'miss'})" if hit is not None else "",
            extra={
                "cache_operation": operation,
                "cache_key": cache_key,
                "cache_hit": hit,
                "type": "cache_operation",
                **kwargs
            }
        )

    def log_model_operation(self, operation: str, ticker: str, **kwargs):
        """Log ML model operations."""
        self.logger.info(
            "Model %s for %s",
            operation,
            ticker,
            extra={
                "model_operation": operation,
                "ticker": ticker,
                "type": "model_operation",
                **kwargs
            }
        )


def setup_performance_logging() -> None:
    """Setup enhanced logging configuration."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

    # Configure specific loggers
    app_logger = logging.getLogger("app")
    app_logger.setLevel(log_level)

    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)

    if settings.debug:
        # More verbose logging in debug mode
        logging.getLogger("app.services").setLevel(logging.DEBUG)
        logging.getLogger("app.api").setLevel(logging.DEBUG)


def get_performance_logger(name: str) -> PerformanceLogger:
    """Get a performance logger for the given module."""
    logger = logging.getLogger(name)
    return PerformanceLogger(logger)


def log_performance(operation_name: str | None = None):
    """Decorator to log function performance."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_performance_logger(func.__module__)
            op_name = operation_name or f"{func.__name__}"

            with logger.time_operation(op_name, function=func.__name__, func_module=func.__module__):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def log_async_performance(operation_name: str | None = None):
    """Decorator to log async function performance."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = get_performance_logger(func.__module__)
            op_name = operation_name or f"{func.__name__}"

            with logger.time_operation(op_name, function=func.__name__, func_module=func.__module__):
                return await func(*args, **kwargs)
        return wrapper
    return decorator
