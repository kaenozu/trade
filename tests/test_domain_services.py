"""Comprehensive tests for domain services."""

import logging
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.domain.entities import (
    MarketData,
    Prediction,
    PredictionPoint,
    PredictionType,
    TickerSymbol,
    TradeAction,
    TradePlan,
    TradeSignal,
)
from app.domain.services import PredictionService, TradePlanService, RiskService


class TestPredictionService:
    """Test the PredictionService class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = PredictionService()
        self.ticker = TickerSymbol("AAPL")

    def test_create_prediction_from_dataframe_price_and_return(self):
        """Test creating prediction with both price and return data."""
        # Create sample DataFrame
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
        data = {
            'expected_price': [150.0, 152.0, 151.0],
            'expected_return': [0.01, 0.013, -0.007]
        }
        pred_df = pd.DataFrame(data, index=dates)
        
        model_metadata = {'confidence': 0.8, 'model_type': 'regression'}
        horizon_days = 3
        
        prediction = self.service.create_prediction_from_dataframe(
            self.ticker, pred_df, horizon_days, model_metadata
        )
        
        assert prediction.ticker == self.ticker
        assert prediction.horizon_days == horizon_days
        assert prediction.model_metadata == model_metadata
        assert len(prediction.prediction_points) == 6  # 3 price + 3 return points
        
        # Check price points
        price_points = [p for p in prediction.prediction_points if p.prediction_type == PredictionType.PRICE]
        assert len(price_points) == 3
        assert price_points[0].expected_value == 150.0
        assert price_points[0].confidence == 0.8
        
        # Check return points
        return_points = [p for p in prediction.prediction_points if p.prediction_type == PredictionType.RETURN]
        assert len(return_points) == 3
        assert return_points[0].expected_value == 0.01
        assert return_points[0].confidence == 0.8

    def test_create_prediction_price_only(self):
        """Test creating prediction with price data only."""
        dates = pd.date_range(start="2023-01-01", periods=2, freq="D")
        data = {'expected_price': [150.0, 152.0]}
        pred_df = pd.DataFrame(data, index=dates)
        
        model_metadata = {'confidence': 0.7}
        
        prediction = self.service.create_prediction_from_dataframe(
            self.ticker, pred_df, 2, model_metadata
        )
        
        assert len(prediction.prediction_points) == 2
        assert all(p.prediction_type == PredictionType.PRICE for p in prediction.prediction_points)

    def test_create_prediction_return_only(self):
        """Test creating prediction with return data only."""
        dates = pd.date_range(start="2023-01-01", periods=2, freq="D")
        data = {'expected_return': [0.02, -0.01]}
        pred_df = pd.DataFrame(data, index=dates)
        
        model_metadata = {'confidence': 0.6}
        
        prediction = self.service.create_prediction_from_dataframe(
            self.ticker, pred_df, 2, model_metadata
        )
        
        assert len(prediction.prediction_points) == 2
        assert all(p.prediction_type == PredictionType.RETURN for p in prediction.prediction_points)

    def test_create_prediction_confidence_bounds(self):
        """Test confidence value is properly bounded between 0 and 1."""
        dates = pd.date_range(start="2023-01-01", periods=1, freq="D")
        pred_df = pd.DataFrame({'expected_price': [150.0]}, index=dates)
        
        # Test confidence > 1
        model_metadata = {'confidence': 1.5}
        prediction = self.service.create_prediction_from_dataframe(
            self.ticker, pred_df, 1, model_metadata
        )
        assert prediction.prediction_points[0].confidence == 1.0
        
        # Test confidence < 0
        model_metadata = {'confidence': -0.2}
        prediction = self.service.create_prediction_from_dataframe(
            self.ticker, pred_df, 1, model_metadata
        )
        assert prediction.prediction_points[0].confidence == 0.0

    def test_calculate_prediction_accuracy_price(self):
        """Test accuracy calculation for price predictions."""
        # Create prediction
        prediction_points = [
            PredictionPoint(
                date=date(2023, 1, 1),
                prediction_type=PredictionType.PRICE,
                expected_value=150.0,
                confidence=0.8
            ),
            PredictionPoint(
                date=date(2023, 1, 2),
                prediction_type=PredictionType.PRICE,
                expected_value=152.0,
                confidence=0.7
            )
        ]
        prediction = Prediction(
            ticker=self.ticker,
            created_at=datetime.now(),
            horizon_days=2,
            prediction_points=prediction_points,
            model_metadata={}
        )
        
        # Create actual market data
        actual_dates = pd.date_range(start="2023-01-01", periods=2, freq="D")
        actual_data = pd.DataFrame({
            'Open': [148.0, 151.0],
            'High': [152.0, 155.0],
            'Low': [147.0, 150.0],
            'Close': [149.0, 153.0],  # Actual prices
            'Volume': [1000000, 1100000]
        }, index=actual_dates)
        
        market_data = MarketData(ticker=self.ticker, data_frame=actual_data, period_days=2)
        
        # Calculate accuracy
        accuracy = self.service.calculate_prediction_accuracy(prediction, market_data)
        
        assert accuracy['data_points_matched'] == 2
        assert accuracy['total_prediction_points'] == 2
        assert accuracy['price_mae'] is not None
        assert accuracy['price_mape'] is not None
        assert accuracy['return_mae'] is None

    def test_calculate_prediction_accuracy_return(self):
        """Test accuracy calculation for return predictions."""
        # Create return prediction
        prediction_points = [
            PredictionPoint(
                date=date(2023, 1, 2),  # Need previous day data for return calc
                prediction_type=PredictionType.RETURN,
                expected_value=0.02,
                confidence=0.8
            )
        ]
        prediction = Prediction(
            ticker=self.ticker,
            created_at=datetime.now(),
            horizon_days=1,
            prediction_points=prediction_points,
            model_metadata={}
        )
        
        # Create actual market data with both days
        actual_dates = pd.date_range(start="2023-01-01", periods=2, freq="D")
        actual_data = pd.DataFrame({
            'Open': [99.0, 102.0],
            'High': [101.0, 104.0],
            'Low': [98.0, 101.0],
            'Close': [100.0, 103.0],  # 3% actual return
            'Volume': [1000000, 1100000]
        }, index=actual_dates)
        
        market_data = MarketData(ticker=self.ticker, data_frame=actual_data, period_days=2)
        
        accuracy = self.service.calculate_prediction_accuracy(prediction, market_data)
        
        assert accuracy['data_points_matched'] == 1
        assert accuracy['return_mae'] is not None
        assert accuracy['price_mae'] is None

    def test_calculate_prediction_accuracy_no_matching_data(self):
        """Test accuracy calculation when no actual data matches."""
        prediction_points = [
            PredictionPoint(
                date=date(2023, 12, 31),  # Date not in actual data
                prediction_type=PredictionType.PRICE,
                expected_value=150.0,
                confidence=0.8
            )
        ]
        prediction = Prediction(
            ticker=self.ticker,
            created_at=datetime.now(),
            horizon_days=1,
            prediction_points=prediction_points,
            model_metadata={}
        )
        
        # Create actual data with different dates
        actual_dates = pd.date_range(start="2023-01-01", periods=2, freq="D")
        actual_data = pd.DataFrame({
            'Open': [99.0, 100.5],
            'High': [101.0, 102.0],
            'Low': [98.0, 100.0],
            'Close': [100.0, 101.0],
            'Volume': [1000000, 1100000]
        }, index=actual_dates)
        market_data = MarketData(ticker=self.ticker, data_frame=actual_data, period_days=2)
        
        accuracy = self.service.calculate_prediction_accuracy(prediction, market_data)
        
        assert accuracy['data_points_matched'] == 0
        assert accuracy['price_mae'] is None
        assert accuracy['return_mae'] is None


class TestTradePlanService:
    """Test the TradePlanService class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = TradePlanService()
        self.ticker = TickerSymbol("GOOGL")

    def test_create_trade_plan_with_confident_returns(self):
        """Test creating trade plan with confident return predictions."""
        # Create prediction with good returns
        prediction_points = [
            PredictionPoint(
                date=date(2023, 1, 1),
                prediction_type=PredictionType.RETURN,
                expected_value=0.03,  # 3% return
                confidence=0.8
            ),
            PredictionPoint(
                date=date(2023, 1, 2),
                prediction_type=PredictionType.RETURN,
                expected_value=-0.025,  # -2.5% return
                confidence=0.7
            )
        ]
        prediction = Prediction(
            ticker=self.ticker,
            created_at=datetime.now(),
            horizon_days=2,
            prediction_points=prediction_points,
            model_metadata={}
        )
        
        trade_plan = self.service.create_trade_plan_from_prediction(prediction)
        
        assert trade_plan.ticker == self.ticker
        assert len(trade_plan.signals) >= 1  # Should have at least one signal
        
        # Should have BUY signal for positive return
        buy_signals = [s for s in trade_plan.signals if s.action == TradeAction.BUY]
        assert len(buy_signals) >= 1
        assert buy_signals[0].expected_return == 0.03

    def test_create_trade_plan_low_confidence(self):
        """Test creating trade plan with low confidence predictions."""
        prediction_points = [
            PredictionPoint(
                date=date(2023, 1, 1),
                prediction_type=PredictionType.RETURN,
                expected_value=0.05,
                confidence=0.4  # Below default threshold of 0.6
            )
        ]
        prediction = Prediction(
            ticker=self.ticker,
            created_at=datetime.now(),
            horizon_days=1,
            prediction_points=prediction_points,
            model_metadata={}
        )
        
        trade_plan = self.service.create_trade_plan_from_prediction(prediction)
        
        # Should result in HOLD signal due to low confidence
        assert len(trade_plan.signals) == 1
        assert trade_plan.signals[0].action == TradeAction.HOLD

    def test_create_trade_plan_no_return_predictions(self):
        """Test creating trade plan with no return predictions."""
        prediction_points = [
            PredictionPoint(
                date=date(2023, 1, 1),
                prediction_type=PredictionType.PRICE,  # Price, not return
                expected_value=150.0,
                confidence=0.8
            )
        ]
        prediction = Prediction(
            ticker=self.ticker,
            created_at=datetime.now(),
            horizon_days=1,
            prediction_points=prediction_points,
            model_metadata={}
        )
        
        trade_plan = self.service.create_trade_plan_from_prediction(prediction)
        
        # Should result in HOLD signal
        assert len(trade_plan.signals) == 1
        assert trade_plan.signals[0].action == TradeAction.HOLD

    def test_create_trade_plan_custom_thresholds(self):
        """Test creating trade plan with custom thresholds."""
        prediction_points = [
            PredictionPoint(
                date=date(2023, 1, 1),
                prediction_type=PredictionType.RETURN,
                expected_value=0.015,  # 1.5% return
                confidence=0.5
            )
        ]
        prediction = Prediction(
            ticker=self.ticker,
            created_at=datetime.now(),
            horizon_days=1,
            prediction_points=prediction_points,
            model_metadata={}
        )
        
        trade_plan = self.service.create_trade_plan_from_prediction(
            prediction, min_confidence=0.4, min_return=0.01
        )
        
        # Should generate BUY signal with custom thresholds
        buy_signals = [s for s in trade_plan.signals if s.action == TradeAction.BUY]
        assert len(buy_signals) >= 1

    def test_optimize_trade_plan_high_transaction_cost(self):
        """Test trade plan optimization with high transaction costs."""
        # Create trade plan with small expected returns
        signals = [
            TradeSignal(
                ticker=self.ticker,
                action=TradeAction.BUY,
                signal_date=date(2023, 1, 1),
                confidence=0.8,
                rationale="Small return",
                expected_return=0.002  # 0.2% return
            ),
            TradeSignal(
                ticker=self.ticker,
                action=TradeAction.SELL,
                signal_date=date(2023, 1, 2),
                confidence=0.8,
                rationale="Exit position",
                expected_return=None
            )
        ]
        
        trade_plan = TradePlan(
            ticker=self.ticker,
            created_at=datetime.now(),
            signals=signals,
            rationale="Test plan"
        )
        
        # Mock market data
        market_data = MagicMock(spec=MarketData)
        
        # High transaction cost should remove the signal
        optimized_plan = self.service.optimize_trade_plan(
            trade_plan, market_data, transaction_cost=0.005
        )
        
        # Should result in HOLD only due to high transaction costs
        hold_signals = [s for s in optimized_plan.signals if s.action == TradeAction.HOLD]
        assert len(hold_signals) >= 1

    def test_optimize_trade_plan_low_transaction_cost(self):
        """Test trade plan optimization with low transaction costs."""
        signals = [
            TradeSignal(
                ticker=self.ticker,
                action=TradeAction.BUY,
                signal_date=date(2023, 1, 1),
                confidence=0.8,
                rationale="Good return",
                expected_return=0.05  # 5% return
            )
        ]
        
        trade_plan = TradePlan(
            ticker=self.ticker,
            created_at=datetime.now(),
            signals=signals,
            rationale="Test plan"
        )
        
        market_data = MagicMock(spec=MarketData)
        
        optimized_plan = self.service.optimize_trade_plan(
            trade_plan, market_data, transaction_cost=0.001
        )
        
        # Should keep the signal due to low transaction costs
        buy_signals = [s for s in optimized_plan.signals if s.action == TradeAction.BUY]
        assert len(buy_signals) >= 1

    def test_optimize_trade_plan_hold_signals_preserved(self):
        """Test that HOLD signals are always preserved in optimization."""
        signals = [
            TradeSignal(
                ticker=self.ticker,
                action=TradeAction.HOLD,
                signal_date=date(2023, 1, 1),
                confidence=0.5,
                rationale="Hold position"
            )
        ]
        
        trade_plan = TradePlan(
            ticker=self.ticker,
            created_at=datetime.now(),
            signals=signals,
            rationale="Hold plan"
        )
        
        market_data = MagicMock(spec=MarketData)
        
        optimized_plan = self.service.optimize_trade_plan(trade_plan, market_data)
        
        assert len(optimized_plan.signals) == 1
        assert optimized_plan.signals[0].action == TradeAction.HOLD


class TestRiskService:
    """Test the RiskService class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = RiskService()
        self.ticker1 = TickerSymbol("AAPL")
        self.ticker2 = TickerSymbol("GOOGL")

    def create_mock_market_data(self, ticker: TickerSymbol, volatility: float) -> MarketData:
        """Create mock market data with specified volatility."""
        market_data = MagicMock(spec=MarketData)
        market_data.ticker = ticker
        market_data.get_volatility.return_value = volatility
        return market_data

    def create_mock_trade_plan(self, ticker: TickerSymbol, expected_return: float = None, confidence: float = 0.8) -> TradePlan:
        """Create mock trade plan."""
        signals = [
            TradeSignal(
                ticker=ticker,
                action=TradeAction.BUY,
                signal_date=date(2023, 1, 1),
                confidence=confidence,
                rationale="Test signal"
            )
        ]
        
        trade_plan = MagicMock(spec=TradePlan)
        trade_plan.ticker = ticker
        trade_plan.signals = signals
        trade_plan.expected_total_return = expected_return
        trade_plan.average_confidence = confidence
        return trade_plan

    def test_calculate_portfolio_risk_multiple_positions(self):
        """Test portfolio risk calculation with multiple positions."""
        # Create trade plans
        trade_plans = [
            self.create_mock_trade_plan(self.ticker1, expected_return=0.05, confidence=0.8),
            self.create_mock_trade_plan(self.ticker2, expected_return=0.03, confidence=0.7)
        ]
        
        # Create market data
        market_data = {
            self.ticker1: self.create_mock_market_data(self.ticker1, volatility=0.2),
            self.ticker2: self.create_mock_market_data(self.ticker2, volatility=0.15)
        }
        
        risk_metrics = self.service.calculate_portfolio_risk(trade_plans, market_data)
        
        assert 'total_risk' in risk_metrics
        assert 'average_volatility' in risk_metrics
        assert 'diversification_score' in risk_metrics
        assert 'expected_portfolio_return' in risk_metrics
        assert 'sharpe_estimate' in risk_metrics
        
        assert risk_metrics['total_risk'] > 0
        assert risk_metrics['diversification_score'] > 0
        assert risk_metrics['expected_portfolio_return'] == 0.04  # Average of 0.05 and 0.03

    def test_calculate_portfolio_risk_empty_plans(self):
        """Test portfolio risk calculation with empty trade plans."""
        risk_metrics = self.service.calculate_portfolio_risk([], {})
        
        assert risk_metrics['total_risk'] == 0.0
        assert risk_metrics['diversification_score'] == 0.0

    def test_calculate_portfolio_risk_no_market_data(self):
        """Test portfolio risk calculation when market data is missing."""
        trade_plans = [self.create_mock_trade_plan(self.ticker1)]
        
        risk_metrics = self.service.calculate_portfolio_risk(trade_plans, {})
        
        assert risk_metrics['total_risk'] == 0.0
        assert risk_metrics['diversification_score'] == 0.0

    def test_assess_position_risk_high_volatility(self):
        """Test position risk assessment with high volatility."""
        trade_plan = self.create_mock_trade_plan(self.ticker1, confidence=0.6)
        market_data = self.create_mock_market_data(self.ticker1, volatility=0.3)  # High volatility
        
        risk_assessment = self.service.assess_position_risk(trade_plan, market_data)
        
        assert risk_assessment['volatility'] == 0.3
        assert risk_assessment['max_potential_loss'] == 0.6  # 2 * volatility
        assert risk_assessment['exceeds_drawdown_threshold'] is True  # > 0.1 default threshold
        assert risk_assessment['recommendation'] == 'HIGH_RISK'

    def test_assess_position_risk_low_volatility(self):
        """Test position risk assessment with low volatility."""
        trade_plan = self.create_mock_trade_plan(self.ticker1, confidence=0.9)
        market_data = self.create_mock_market_data(self.ticker1, volatility=0.05)  # Low volatility
        
        risk_assessment = self.service.assess_position_risk(trade_plan, market_data)
        
        assert risk_assessment['volatility'] == 0.05
        assert risk_assessment['max_potential_loss'] == 0.1  # 2 * volatility
        assert risk_assessment['exceeds_drawdown_threshold'] is False  # <= 0.1 threshold
        assert risk_assessment['recommendation'] == 'LOW_RISK'

    def test_assess_position_risk_medium_risk(self):
        """Test position risk assessment for medium risk scenario."""
        trade_plan = self.create_mock_trade_plan(self.ticker1, confidence=0.7)
        market_data = self.create_mock_market_data(self.ticker1, volatility=0.1)
        
        risk_assessment = self.service.assess_position_risk(trade_plan, market_data)
        
        # Risk score = volatility * (1 - confidence) = 0.1 * 0.3 = 0.03
        expected_risk_score = 0.1 * (1.0 - 0.7)
        assert risk_assessment['risk_score'] == expected_risk_score
        
        # max_potential_loss = 0.1 * 2 = 0.2 > 0.1 (default threshold), so HIGH_RISK
        assert risk_assessment['max_potential_loss'] == 0.2
        assert risk_assessment['exceeds_drawdown_threshold'] is True
        assert risk_assessment['recommendation'] == 'HIGH_RISK'

    def test_assess_position_risk_custom_threshold(self):
        """Test position risk assessment with custom drawdown threshold."""
        trade_plan = self.create_mock_trade_plan(self.ticker1)
        market_data = self.create_mock_market_data(self.ticker1, volatility=0.1)
        
        # Use very low threshold
        risk_assessment = self.service.assess_position_risk(
            trade_plan, market_data, max_drawdown_threshold=0.05
        )
        
        assert risk_assessment['max_potential_loss'] == 0.2  # 2 * 0.1
        assert risk_assessment['exceeds_drawdown_threshold'] is True  # 0.2 > 0.05


class TestDomainServicesIntegration:
    """Integration tests for domain services."""

    def setup_method(self):
        """Set up test fixtures."""
        self.prediction_service = PredictionService()
        self.trade_plan_service = TradePlanService()
        self.risk_service = RiskService()
        self.ticker = TickerSymbol("TSLA")

    def test_end_to_end_workflow(self):
        """Test complete workflow from prediction to risk assessment."""
        # 1. Create prediction
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
        pred_data = {
            'expected_return': [0.03, -0.01, 0.02],
            'expected_price': [200.0, 198.0, 202.0]
        }
        pred_df = pd.DataFrame(pred_data, index=dates)
        
        prediction = self.prediction_service.create_prediction_from_dataframe(
            self.ticker, pred_df, 3, {'confidence': 0.8}
        )
        
        # 2. Create trade plan
        trade_plan = self.trade_plan_service.create_trade_plan_from_prediction(prediction)
        
        # 3. Assess risk
        mock_market_data = MagicMock(spec=MarketData)
        mock_market_data.get_volatility.return_value = 0.15
        
        risk_assessment = self.risk_service.assess_position_risk(trade_plan, mock_market_data)
        
        # Verify complete workflow
        assert prediction.ticker == self.ticker
        assert len(prediction.prediction_points) == 6  # 3 returns + 3 prices
        assert trade_plan.ticker == self.ticker
        assert len(trade_plan.signals) > 0
        assert 'recommendation' in risk_assessment

    def test_logging_integration(self):
        """Test that domain services use logging properly."""
        with patch('app.domain.services.logger') as mock_logger:
            # Create a simple prediction
            dates = pd.date_range(start="2023-01-01", periods=1, freq="D")
            pred_df = pd.DataFrame({'expected_price': [150.0]}, index=dates)
            
            self.prediction_service.create_prediction_from_dataframe(
                self.ticker, pred_df, 1, {}
            )
            
            # Should have logged the operation
            mock_logger.debug.assert_called()