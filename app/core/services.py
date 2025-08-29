"""Service implementations with dependency injection support."""

import logging
from typing import Any

import pandas as pd

from ..services import data, features, model, signals, tickers
from .interfaces import (
    DataServiceInterface,
    FeatureServiceInterface,
    ModelServiceInterface,
    SignalServiceInterface,
    TickerServiceInterface,
)

logger = logging.getLogger(__name__)


class DataService(DataServiceInterface):
    """Default data service implementation with async support."""

    def __init__(self):
        from ..services.async_data import get_async_data_service
        self._async_service = get_async_data_service()

    def fetch_ohlcv(self, ticker: str, period_days: int) -> pd.DataFrame:
        """Fetch OHLCV data for a ticker."""
        logger.debug("Fetching OHLCV data for %s (period: %d days)", ticker, period_days)
        return data.fetch_ohlcv(ticker, period_days)

    def fetch_last_close_direct(self, ticker: str) -> tuple[float, str]:
        """Fetch last close price and date directly."""
        logger.debug("Fetching last close price for %s", ticker)
        return data.fetch_last_close_direct(ticker)

    async def fetch_ohlcv_async(self, ticker: str, period_days: int) -> pd.DataFrame:
        """Fetch OHLCV data for a ticker asynchronously."""
        return await self._async_service.fetch_ohlcv_async(ticker, period_days)

    async def fetch_last_close_async(self, ticker: str) -> tuple[float, str]:
        """Fetch last close price and date directly asynchronously."""
        return await self._async_service.fetch_last_close_async(ticker)

    async def fetch_multiple_quotes_async(self, tickers: list[str]) -> dict[str, tuple[float, str] | Exception]:
        """Fetch multiple quotes concurrently."""
        return await self._async_service.fetch_multiple_quotes_async(tickers)


class FeatureService(FeatureServiceInterface):
    """Default feature engineering service implementation."""

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build feature frame from OHLCV data."""
        logger.debug("Building features for %d rows of data", len(df))
        return features.build_feature_frame(df)


class ModelService(ModelServiceInterface):
    """Default ML model service implementation."""

    def train_or_load_model(self, ticker: str, feat: pd.DataFrame) -> tuple[Any, dict]:
        """Train or load a model for the ticker."""
        logger.debug("Training/loading model for %s", ticker)
        return model.train_or_load_model(ticker, feat)

    def predict_future(
        self, 
        df: pd.DataFrame, 
        feat: pd.DataFrame, 
        model: Any, 
        horizon_days: int
    ) -> pd.DataFrame:
        """Generate future predictions."""
        from ..services import model as model_module
        logger.debug("Generating %d-day predictions", horizon_days)
        return model_module.predict_future(df, feat, model, horizon_days)


class SignalService(SignalServiceInterface):
    """Default trading signal service implementation."""

    def generate_trade_plan(self, pred_df: pd.DataFrame) -> dict:
        """Generate trade plan from predictions."""
        logger.debug("Generating trade plan from %d predictions", len(pred_df))
        return signals.generate_trade_plan(pred_df)


class TickerService(TickerServiceInterface):
    """Default ticker information service implementation."""

    def list_tickers(self, query: str | None = None, limit: int = 200) -> list[dict]:
        """List available tickers."""
        logger.debug("Listing tickers (query: %s, limit: %d)", query, limit)
        return tickers.list_jp_tickers(query, limit)


class ServiceContainer:
    """Container for service dependencies."""

    def __init__(
        self,
        data_service: DataServiceInterface | None = None,
        feature_service: FeatureServiceInterface | None = None,
        model_service: ModelServiceInterface | None = None,
        signal_service: SignalServiceInterface | None = None,
        ticker_service: TickerServiceInterface | None = None,
    ):
        self.data_service = data_service or DataService()
        self.feature_service = feature_service or FeatureService()
        self.model_service = model_service or ModelService()
        self.signal_service = signal_service or SignalService()
        self.ticker_service = ticker_service or TickerService()

    def get_data_service(self) -> DataServiceInterface:
        """Get data service instance."""
        return self.data_service

    def get_feature_service(self) -> FeatureServiceInterface:
        """Get feature service instance."""
        return self.feature_service

    def get_model_service(self) -> ModelServiceInterface:
        """Get model service instance."""
        return self.model_service

    def get_signal_service(self) -> SignalServiceInterface:
        """Get signal service instance."""
        return self.signal_service

    def get_ticker_service(self) -> TickerServiceInterface:
        """Get ticker service instance."""
        return self.ticker_service


# Global service container instance
_service_container: ServiceContainer | None = None


def get_container() -> ServiceContainer:
    """Get the global service container."""
    global _service_container
    if _service_container is None:
        _service_container = ServiceContainer()
    return _service_container


def set_container(container: ServiceContainer) -> None:
    """Set the global service container (for testing)."""
    global _service_container
    _service_container = container