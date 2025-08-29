"""Middleware configuration for the FastAPI application."""

import logging
import time

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from .config import settings
from .logging import get_performance_logger

logger = logging.getLogger(__name__)


def setup_cors(app: FastAPI) -> None:
    """Configure CORS middleware."""
    try:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins_list,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
        logger.info("CORS middleware configured with origins: %s", settings.cors_origins_list)
    except Exception as e:
        logger.warning("Failed to setup CORS middleware: %s", e)


def setup_sentry() -> None:
    """Configure Sentry for error tracking."""
    if not settings.sentry_dsn:
        return
        
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            integrations=[FastApiIntegration()],
            traces_sample_rate=settings.sentry_traces_sample_rate,
            profiles_sample_rate=settings.sentry_profiles_sample_rate,
            environment=settings.sentry_env,
        )
        logger.info("Sentry initialized for environment: %s", settings.sentry_env)
    except ImportError:
        logger.warning("Sentry SDK not available, skipping initialization")
    except Exception as e:
        logger.warning("Failed to initialize Sentry: %s", e)


def setup_prometheus(app: FastAPI) -> None:
    """Configure Prometheus metrics collection."""
    if not settings.metrics_enabled:
        logger.info("Prometheus metrics disabled")
        return
        
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
        
        Instrumentator().instrument(app).expose(app, include_in_schema=False)
        logger.info("Prometheus metrics enabled at /metrics")
    except ImportError:
        logger.warning("Prometheus instrumentator not available")
    except Exception as e:
        logger.warning("Failed to setup Prometheus metrics: %s", e)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log HTTP requests and responses."""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = get_performance_logger(__name__)
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Log request and response details with timing."""
        start_time = time.time()
        
        # Log incoming request
        self.logger.log_api_request(
            endpoint=str(request.url.path),
            method=request.method,
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown")
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logging.getLogger(__name__).info(
                "Request completed: %s %s - %d (%.3fs)",
                request.method,
                request.url.path,
                response.status_code,
                duration,
                extra={
                    "method": request.method,
                    "path": str(request.url.path),
                    "status_code": response.status_code,
                    "duration_seconds": duration,
                    "type": "http_response"
                }
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logging.getLogger(__name__).error(
                "Request failed: %s %s - %s (%.3fs)",
                request.method,
                request.url.path,
                str(e),
                duration,
                extra={
                    "method": request.method,
                    "path": str(request.url.path),
                    "error": str(e),
                    "duration_seconds": duration,
                    "type": "http_error"
                }
            )
            raise


def setup_logging() -> None:
    """Configure application logging."""
    from .logging import setup_performance_logging
    setup_performance_logging()


def setup_request_logging(app: FastAPI) -> None:
    """Setup request logging middleware."""
    app.add_middleware(RequestLoggingMiddleware)