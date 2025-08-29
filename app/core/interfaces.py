"""Service interfaces for dependency injection."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class DataServiceInterface(ABC):
    """Interface for data retrieval services."""

    @abstractmethod
    def fetch_ohlcv(self, ticker: str, period_days: int) -> pd.DataFrame:
        """Fetch OHLCV data for a ticker."""
        pass

    @abstractmethod
    def fetch_last_close_direct(self, ticker: str) -> tuple[float, str]:
        """Fetch last close price and date directly."""
        pass

    @abstractmethod
    async def fetch_ohlcv_async(self, ticker: str, period_days: int) -> pd.DataFrame:
        """Fetch OHLCV data for a ticker asynchronously."""
        pass

    @abstractmethod
    async def fetch_last_close_async(self, ticker: str) -> tuple[float, str]:
        """Fetch last close price and date directly asynchronously."""
        pass

    @abstractmethod
    async def fetch_multiple_quotes_async(self, tickers: list[str]) -> dict[str, tuple[float, str] | Exception]:
        """Fetch multiple quotes concurrently."""
        pass


class FeatureServiceInterface(ABC):
    """Interface for feature engineering services."""

    @abstractmethod
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build feature frame from OHLCV data."""
        pass


class ModelServiceInterface(ABC):
    """Interface for ML model services."""

    @abstractmethod
    def train_or_load_model(self, ticker: str, feat: pd.DataFrame) -> tuple[Any, dict]:
        """Train or load a model for the ticker."""
        pass

    @abstractmethod
    def predict_future(
        self,
        df: pd.DataFrame,
        feat: pd.DataFrame,
        model: Any,
        horizon_days: int
    ) -> pd.DataFrame:
        """Generate future predictions."""
        pass


class SignalServiceInterface(ABC):
    """Interface for trading signal services."""

    @abstractmethod
    def generate_trade_plan(self, pred_df: pd.DataFrame) -> dict:
        """Generate trade plan from predictions."""
        pass


class TickerServiceInterface(ABC):
    """Interface for ticker information services."""

    @abstractmethod
    def list_tickers(self, query: str | None = None, limit: int = 200) -> list[dict]:
        """List available tickers."""
        pass
