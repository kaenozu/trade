"""Domain entities representing core business objects."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd


class TickerSymbol:
    """Value object representing a validated stock ticker symbol."""
    
    _PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,15}$")
    
    def __init__(self, value: str):
        if not value or not isinstance(value, str):
            raise ValueError("Ticker must be a non-empty string")
        
        if not self._PATTERN.match(value):
            raise ValueError(f"Invalid ticker format: {value}")
        
        self._value = value.upper().strip()
    
    @property
    def value(self) -> str:
        return self._value
    
    def __str__(self) -> str:
        return self._value
    
    def __repr__(self) -> str:
        return f"TickerSymbol('{self._value}')"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, TickerSymbol):
            return self._value == other._value
        return False
    
    def __hash__(self) -> int:
        return hash(self._value)


class Price:
    """Value object representing a price with validation."""
    
    def __init__(self, value: float, currency: str = "JPY"):
        if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
            raise ValueError("Price must be a valid number")
        
        if value < 0:
            raise ValueError("Price cannot be negative")
        
        self._value = float(value)
        self._currency = currency.upper()
    
    @property
    def value(self) -> float:
        return self._value
    
    @property
    def currency(self) -> str:
        return self._currency
    
    def __str__(self) -> str:
        return f"{self._value:.2f} {self._currency}"
    
    def __repr__(self) -> str:
        return f"Price({self._value}, '{self._currency}')"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Price):
            return (
                abs(self._value - other._value) < 1e-6 
                and self._currency == other._currency
            )
        return False


@dataclass(frozen=True)
class Quote:
    """Domain entity representing a stock quote."""
    
    ticker: TickerSymbol
    price: Price
    timestamp: datetime
    volume: Optional[int] = None
    
    def __post_init__(self):
        if self.volume is not None and self.volume < 0:
            raise ValueError("Volume cannot be negative")
    
    @property
    def age_seconds(self) -> float:
        """Get the age of this quote in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()
    
    def is_stale(self, max_age_seconds: int = 300) -> bool:
        """Check if the quote is stale."""
        return self.age_seconds > max_age_seconds


@dataclass
class MarketData:
    """Domain entity representing market data for a ticker."""
    
    ticker: TickerSymbol
    data_frame: pd.DataFrame
    period_days: int
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in self.data_frame.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(self.data_frame) == 0:
            raise ValueError("Market data cannot be empty")
    
    @property
    def latest_price(self) -> Price:
        """Get the latest closing price."""
        latest_close = float(self.data_frame['Close'].iloc[-1])
        return Price(latest_close)
    
    @property
    def date_range(self) -> tuple[date, date]:
        """Get the date range of the data."""
        start_date = self.data_frame.index.min().date()
        end_date = self.data_frame.index.max().date()
        return start_date, end_date
    
    @property
    def total_data_points(self) -> int:
        """Get the total number of data points."""
        return len(self.data_frame)
    
    def get_returns(self, periods: int = 1) -> pd.Series:
        """Calculate returns for the specified periods."""
        return self.data_frame['Close'].pct_change(periods).dropna()
    
    def get_volatility(self, window: int = 20) -> float:
        """Calculate annualized volatility."""
        returns = self.get_returns()
        return float(returns.rolling(window).std().iloc[-1] * np.sqrt(252))


class PredictionType(Enum):
    """Types of predictions available."""
    PRICE = "price"
    RETURN = "return"
    DIRECTION = "direction"


@dataclass(frozen=True)
class PredictionPoint:
    """Domain entity representing a single prediction point."""
    
    date: date
    prediction_type: PredictionType
    expected_value: float
    confidence: float
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class Prediction:
    """Domain entity representing a complete prediction for a ticker."""
    
    ticker: TickerSymbol
    created_at: datetime
    horizon_days: int
    prediction_points: List[PredictionPoint]
    model_metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.horizon_days <= 0:
            raise ValueError("Horizon days must be positive")
        
        if not self.prediction_points:
            raise ValueError("Prediction must contain at least one prediction point")
        
        # Sort prediction points by date
        self.prediction_points.sort(key=lambda p: p.date)
    
    @property
    def average_confidence(self) -> float:
        """Get the average confidence across all prediction points."""
        confidences = [p.confidence for p in self.prediction_points]
        return float(np.mean(confidences))
    
    @property
    def prediction_period(self) -> tuple[date, date]:
        """Get the date range of the predictions."""
        start_date = self.prediction_points[0].date
        end_date = self.prediction_points[-1].date
        return start_date, end_date


class TradeAction(Enum):
    """Types of trade actions."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(frozen=True)
class TradeSignal:
    """Domain entity representing a trading signal."""
    
    ticker: TickerSymbol
    action: TradeAction
    signal_date: date
    target_date: Optional[date] = None
    confidence: float = 0.5
    rationale: str = ""
    expected_return: Optional[float] = None
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        if self.target_date and self.target_date < self.signal_date:
            raise ValueError("Target date cannot be before signal date")


@dataclass
class TradePlan:
    """Domain entity representing a complete trading plan."""
    
    ticker: TickerSymbol
    created_at: datetime
    signals: List[TradeSignal]
    rationale: str = ""
    expected_total_return: Optional[float] = None
    
    def __post_init__(self):
        if not self.signals:
            raise ValueError("Trade plan must contain at least one signal")
        
        # Sort signals by date
        self.signals.sort(key=lambda s: s.signal_date)
    
    @property
    def buy_signals(self) -> List[TradeSignal]:
        """Get all buy signals."""
        return [s for s in self.signals if s.action == TradeAction.BUY]
    
    @property
    def sell_signals(self) -> List[TradeSignal]:
        """Get all sell signals."""
        return [s for s in self.signals if s.action == TradeAction.SELL]
    
    @property
    def average_confidence(self) -> float:
        """Get the average confidence across all signals."""
        confidences = [s.confidence for s in self.signals]
        return float(np.mean(confidences))
    
    @property
    def plan_period(self) -> tuple[date, date]:
        """Get the date range of the trade plan."""
        start_date = self.signals[0].signal_date
        end_date = max(s.target_date or s.signal_date for s in self.signals)
        return start_date, end_date