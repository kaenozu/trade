"""Refactored main application module with improved architecture."""

import logging

from fastapi import FastAPI

from .api import frontend, health, predictions, quotes, tickers
from .core.config import settings
from .core.error_handlers import setup_error_handlers
from .core.middleware import setup_cors, setup_logging, setup_prometheus, setup_sentry, setup_request_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize Sentry early
setup_sentry()

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="ML-powered Japanese stock prediction and trading signal generator",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Setup middleware
setup_request_logging(app)
setup_cors(app)
setup_prometheus(app)

# Setup error handlers
setup_error_handlers(app)

# Include routers
app.include_router(health.router, tags=["System"])
app.include_router(tickers.router, tags=["Market Data"])
app.include_router(quotes.router, tags=["Market Data"])
app.include_router(predictions.router, tags=["ML Predictions"])
app.include_router(frontend.router, tags=["Frontend"])

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    if settings.git_sha:
        logger.info("Git SHA: %s", settings.git_sha)
    
    # Log configuration
    logger.info("Cache directory: %s", settings.cache_directory)
    logger.info("Model directory: %s", settings.model_directory)
    logger.info("Synthetic data: %s", "enabled" if settings.allow_synthetic_data else "disabled")
    logger.info("Metrics: %s", "enabled" if settings.metrics_enabled else "disabled")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info("Shutting down %s", settings.app_name)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    )