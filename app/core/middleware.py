"""Middleware configuration for the FastAPI application."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings

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


def setup_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if settings.debug:
        logging.getLogger("app").setLevel(logging.DEBUG)