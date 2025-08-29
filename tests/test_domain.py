"""Tests for domain layer functionality."""

import pytest
from datetime import datetime, date
import pandas as pd
import numpy as np

from app.domain.entities import (
    TickerSymbol, Price, Quote, MarketData, PredictionType, PredictionPoint,
    Prediction, TradeAction, TradeSignal, TradePlan
)
from app.domain.services import PredictionService, TradePlanService, RiskService


class TestDomainEntities:
    """Test domain entities."""

    def test_ticker_symbol_creation(self):
        """Test ticker symbol creation and validation."""
        # Valid ticker symbols
        valid_symbols = ["AAPL", "googl", "TEST.T", "BRK-A", "SPX_INDEX"]

        for symbol_str in valid_symbols:
            ticker = TickerSymbol(symbol_str)
            assert ticker.value == symbol_str.upper().strip()
            assert str(ticker) == symbol_str.upper().strip()

        # Invalid ticker symbols
        invalid_symbols = ["", "   ", "A" * 20, "TICK$R", None]

        for symbol_str in invalid_symbols:
            with pytest.raises(ValueError):
                TickerSymbol(symbol_str)

    def test_ticker_symbol_equality(self):
        """Test ticker symbol equality and hashing."""
        ticker1 = TickerSymbol("AAPL")
        ticker2 = TickerSymbol("aapl")
        ticker3 = TickerSymbol("GOOGL")

        assert ticker1 == ticker2
        assert ticker1 != ticker3
        assert hash(ticker1) == hash(ticker2)
        assert hash(ticker1) != hash(ticker3)

    def test_price_validation(self):
        """Test price validation."""
        # Valid prices
        price = Price(100.50, "USD")
        assert price.value == 100.50
        assert price.currency == "USD"

        # Test equality
        price2 = Price(100.50, "USD")
        assert price == price2

        # Invalid prices
        invalid_values = [-100.0, float('nan'), float('inf'), "not_a_number"]

        for value in invalid_values:
            with pytest.raises(ValueError):
                Price(value)

    def test_quote_creation(self):
        """Test quote creation."""
        ticker = TickerSymbol("AAPL")
        price = Price(150.75, "USD")
        timestamp = datetime.now()

        quote = Quote(
            ticker=ticker,
            price=price,
            timestamp=timestamp,
            volume=1000000
        )

        assert quote.ticker == ticker
        assert quote.price == price
        assert quote.timestamp == timestamp
        assert quote.volume == 1000000
        assert not quote.is_stale(max_age_seconds=60)

        # Test with negative volume
        with pytest.raises(ValueError):
            Quote(ticker=ticker, price=price, timestamp=timestamp, volume=-100)

    def test_market_data_validation(self):
        """Test market data validation."""
        ticker = TickerSymbol("TEST")

        # Valid market data
        df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [102, 103, 104],
            'Low': [99, 100, 101],
            'Close': [101, 102, 103],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))

        market_data = MarketData(ticker=ticker, data_frame=df, period_days=30)

        assert market_data.ticker == ticker
        assert market_data.total_data_points == 3
        assert market_data.latest_price.value == 103.0

        start_date, end_date = market_data.date_range
        assert isinstance(start_date, date)
        assert isinstance(end_date, date)

        returns = market_data.get_returns()
        assert len(returns) == 2  # One less due to pct_change

        volatility = market_data.get_volatility(window=2)
        assert isinstance(volatility, float)

        # Test with missing columns
        invalid_df = pd.DataFrame({'Close': [100, 101, 102]})
        with pytest.raises(ValueError, match="Missing required columns"):
            MarketData(ticker=ticker, data_frame=invalid_df, period_days=30)

        # Test with empty DataFrame
        empty_df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        with pytest.raises(ValueError, match="Market data cannot be empty"):
            MarketData(ticker=ticker, data_frame=empty_df, period_days=30)

    def test_prediction_point_validation(self):
        """Test prediction point validation."""
        # Valid prediction point
        point = PredictionPoint(
            date=date(2023, 1, 1),
            prediction_type=PredictionType.PRICE,
            expected_value=100.50,
            confidence=0.85
        )

        assert point.date == date(2023, 1, 1)
        assert point.prediction_type == PredictionType.PRICE
        assert point.expected_value == 100.50
        assert point.confidence == 0.85

        # Invalid confidence values
        with pytest.raises(ValueError, match="Confidence must be between"):
            PredictionPoint(
                date=date(2023, 1, 1),
                prediction_type=PredictionType.PRICE,
                expected_value=100.0,
                confidence=1.5
            )

        with pytest.raises(ValueError):
            PredictionPoint(
                date=date(2023, 1, 1),
                prediction_type=PredictionType.PRICE,
                expected_value=100.0,
                confidence=-0.1
            )

    def test_prediction_creation(self):
        """Test prediction creation and methods."""
        ticker = TickerSymbol("TEST")
        points = [
            PredictionPoint(
                date=date(2023, 1, 1),
                prediction_type=PredictionType.PRICE,
                expected_value=100.0,
                confidence=0.8
            ),
            PredictionPoint(
                date=date(2023, 1, 2),
                prediction_type=PredictionType.RETURN,
                expected_value=0.02,
                confidence=0.7
            )
        ]

        prediction = Prediction(
            ticker=ticker,
            created_at=datetime.now(),
            horizon_days=5,
            prediction_points=points
        )

        assert prediction.ticker == ticker
        assert len(prediction.prediction_points) == 2
        assert prediction.average_confidence == 0.75

        start_date, end_date = prediction.prediction_period
        assert start_date == date(2023, 1, 1)
        assert end_date == date(2023, 1, 2)

        # Test validation errors
        with pytest.raises(ValueError, match="Horizon days must be positive"):
            Prediction(
                ticker=ticker,
                created_at=datetime.now(),
                horizon_days=0,
                prediction_points=points
            )

        with pytest.raises(ValueError, match="must contain at least one"):
            Prediction(
                ticker=ticker,
                created_at=datetime.now(),
                horizon_days=5,
                prediction_points=[]
            )

    def test_trade_signal_validation(self):
        """Test trade signal validation."""
        ticker = TickerSymbol("TEST")

        # Valid trade signal
        signal = TradeSignal(
            ticker=ticker,
            action=TradeAction.BUY,
            signal_date=date(2023, 1, 1),
            target_date=date(2023, 1, 5),
            confidence=0.8,
            rationale="Expected price increase",
            expected_return=0.05
        )

        assert signal.ticker == ticker
        assert signal.action == TradeAction.BUY
        assert signal.confidence == 0.8

        # Test validation errors
        with pytest.raises(ValueError, match="Target date cannot be before"):
            TradeSignal(
                ticker=ticker,
                action=TradeAction.BUY,
                signal_date=date(2023, 1, 5),
                target_date=date(2023, 1, 1)  # Before signal date
            )

    def test_trade_plan_creation(self):
        """Test trade plan creation and methods."""
        ticker = TickerSymbol("TEST")
        signals = [
            TradeSignal(
                ticker=ticker,
                action=TradeAction.BUY,
                signal_date=date(2023, 1, 1),
                confidence=0.8,
                expected_return=0.05
            ),
            TradeSignal(
                ticker=ticker,
                action=TradeAction.SELL,
                signal_date=date(2023, 1, 5),
                confidence=0.7,
                expected_return=-0.02
            )
        ]

        plan = TradePlan(
            ticker=ticker,
            created_at=datetime.now(),
            signals=signals,
            expected_total_return=0.03
        )

        assert plan.ticker == ticker
        assert len(plan.signals) == 2
        assert plan.average_confidence == 0.75
        assert len(plan.buy_signals) == 1
        assert len(plan.sell_signals) == 1

        start_date, end_date = plan.plan_period
        assert start_date == date(2023, 1, 1)
        assert end_date == date(2023, 1, 5)


class TestDomainServices:
    """Test domain services."""

    def test_prediction_service(self):
        """Test prediction service functionality."""
        service = PredictionService()
        ticker = TickerSymbol("TEST")

        # Create mock prediction DataFrame
        pred_df = pd.DataFrame({
            'expected_price': [100.0, 101.0, 102.0],
            'expected_return': [0.01, 0.01, 0.01]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))

        model_metadata = {'confidence': 0.8, 'r2_mean': 0.75}

        prediction = service.create_prediction_from_dataframe(
            ticker=ticker,
            pred_df=pred_df,
            horizon_days=3,
            model_metadata=model_metadata
        )

        assert prediction.ticker == ticker
        assert prediction.horizon_days == 3
        assert len(prediction.prediction_points) == 6  # 3 price + 3 return points
        assert prediction.model_metadata == model_metadata

    def test_trade_plan_service(self):
        """Test trade plan service functionality."""
        service = TradePlanService()
        ticker = TickerSymbol("TEST")

        # Create mock prediction
        points = [
            PredictionPoint(
                date=date(2023, 1, 1),
                prediction_type=PredictionType.RETURN,
                expected_value=0.03,  # Above min_return
                confidence=0.8
            ),
            PredictionPoint(
                date=date(2023, 1, 2),
                prediction_type=PredictionType.RETURN,
                expected_value=-0.03,  # Below -min_return
                confidence=0.7
            )
        ]

        prediction = Prediction(
            ticker=ticker,
            created_at=datetime.now(),
            horizon_days=2,
            prediction_points=points
        )

        trade_plan = service.create_trade_plan_from_prediction(
            prediction=prediction,
            min_confidence=0.6,
            min_return=0.02
        )

        assert trade_plan.ticker == ticker
        assert len(trade_plan.signals) >= 1  # Should have at least one signal

        # Test with low confidence predictions
        low_confidence_points = [
            PredictionPoint(
                date=date(2023, 1, 1),
                prediction_type=PredictionType.RETURN,
                expected_value=0.05,
                confidence=0.4  # Below min_confidence
            )
        ]

        low_conf_prediction = Prediction(
            ticker=ticker,
            created_at=datetime.now(),
            horizon_days=1,
            prediction_points=low_confidence_points
        )

        low_conf_plan = service.create_trade_plan_from_prediction(low_conf_prediction)

        # Should default to HOLD
        assert any(s.action == TradeAction.HOLD for s in low_conf_plan.signals)

    def test_risk_service(self):
        """Test risk service functionality."""
        service = RiskService()
        ticker = TickerSymbol("TEST")

        # Create mock market data
        df = pd.DataFrame({
            'Open': np.random.normal(100, 5, 100),
            'High': np.random.normal(102, 5, 100),
            'Low': np.random.normal(98, 5, 100),
            'Close': np.random.normal(100, 5, 100),
            'Volume': np.random.randint(1000, 2000, 100)
        }, index=pd.date_range('2023-01-01', periods=100, freq='D'))

        market_data = MarketData(ticker=ticker, data_frame=df, period_days=100)

        # Create mock trade plan
        signals = [
            TradeSignal(
                ticker=ticker,
                action=TradeAction.BUY,
                signal_date=date(2023, 1, 1),
                confidence=0.8
            )
        ]

        trade_plan = TradePlan(
            ticker=ticker,
            created_at=datetime.now(),
            signals=signals
        )

        # Test position risk assessment
        risk_assessment = service.assess_position_risk(trade_plan, market_data)

        assert 'volatility' in risk_assessment
        assert 'max_potential_loss' in risk_assessment
        assert 'risk_score' in risk_assessment
        assert 'recommendation' in risk_assessment
        assert risk_assessment['recommendation'] in ['LOW_RISK', 'MEDIUM_RISK', 'HIGH_RISK']

        # Test portfolio risk calculation
        portfolio_risk = service.calculate_portfolio_risk([trade_plan], {ticker: market_data})

        assert 'total_risk' in portfolio_risk
        assert 'diversification_score' in portfolio_risk
        assert isinstance(portfolio_risk['total_risk'], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
