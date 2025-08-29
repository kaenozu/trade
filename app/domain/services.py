"""Domain services containing business logic."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from .entities import (
    MarketData,
    Prediction,
    PredictionPoint,
    PredictionType,
    TickerSymbol,
    TradeAction,
    TradePlan,
    TradeSignal,
)

logger = logging.getLogger(__name__)


class PredictionService:
    """Domain service for generating and managing predictions."""

    def create_prediction_from_dataframe(
        self,
        ticker: TickerSymbol,
        pred_df: pd.DataFrame,
        horizon_days: int,
        model_metadata: dict
    ) -> Prediction:
        """Create a prediction from a DataFrame with expected returns and prices."""
        logger.debug("Creating prediction for %s with %d data points", ticker, len(pred_df))

        prediction_points = []

        for idx, row in pred_df.iterrows():
            # Price prediction
            if 'expected_price' in row:
                price_point = PredictionPoint(
                    date=pd.to_datetime(idx).date(),
                    prediction_type=PredictionType.PRICE,
                    expected_value=float(row['expected_price']),
                    confidence=min(1.0, max(0.0, float(model_metadata.get('confidence', 0.5))))
                )
                prediction_points.append(price_point)

            # Return prediction
            if 'expected_return' in row:
                return_point = PredictionPoint(
                    date=pd.to_datetime(idx).date(),
                    prediction_type=PredictionType.RETURN,
                    expected_value=float(row['expected_return']),
                    confidence=min(1.0, max(0.0, float(model_metadata.get('confidence', 0.5))))
                )
                prediction_points.append(return_point)

        return Prediction(
            ticker=ticker,
            created_at=datetime.now(),
            horizon_days=horizon_days,
            prediction_points=prediction_points,
            model_metadata=model_metadata
        )

    def calculate_prediction_accuracy(
        self,
        prediction: Prediction,
        actual_data: MarketData
    ) -> dict:
        """Calculate accuracy metrics for a prediction against actual data."""
        logger.debug("Calculating accuracy for prediction of %s", prediction.ticker)

        # Filter prediction points that have corresponding actual data
        actual_prices = {}
        for idx, row in actual_data.data_frame.iterrows():
            actual_prices[pd.to_datetime(idx).date()] = float(row['Close'])

        price_errors = []
        return_errors = []

        for point in prediction.prediction_points:
            if point.date in actual_prices:
                if point.prediction_type == PredictionType.PRICE:
                    actual_price = actual_prices[point.date]
                    error = abs(point.expected_value - actual_price) / actual_price
                    price_errors.append(error)

                elif point.prediction_type == PredictionType.RETURN:
                    # Calculate actual return (need previous day's price)
                    prev_date = point.date - timedelta(days=1)
                    if prev_date in actual_prices:
                        actual_return = (actual_prices[point.date] - actual_prices[prev_date]) / actual_prices[prev_date]
                        error = abs(point.expected_value - actual_return)
                        return_errors.append(error)

        return {
            'price_mae': float(np.mean(price_errors)) if price_errors else None,
            'price_mape': float(np.mean(price_errors)) * 100 if price_errors else None,
            'return_mae': float(np.mean(return_errors)) if return_errors else None,
            'data_points_matched': len(price_errors) + len(return_errors),
            'total_prediction_points': len(prediction.prediction_points)
        }


class TradePlanService:
    """Domain service for generating and managing trade plans."""

    def create_trade_plan_from_prediction(
        self,
        prediction: Prediction,
        min_confidence: float = 0.6,
        min_return: float = 0.02
    ) -> TradePlan:
        """Create a trade plan from a prediction."""
        logger.debug(
            "Creating trade plan for %s with min_confidence=%.2f, min_return=%.2f",
            prediction.ticker, min_confidence, min_return
        )

        # Extract return predictions
        return_points = [
            p for p in prediction.prediction_points
            if p.prediction_type == PredictionType.RETURN and p.confidence >= min_confidence
        ]

        if not return_points:
            # Create a HOLD signal if no confident predictions
            hold_signal = TradeSignal(
                ticker=prediction.ticker,
                action=TradeAction.HOLD,
                signal_date=datetime.now().date(),
                confidence=0.5,
                rationale="No confident predictions available"
            )

            return TradePlan(
                ticker=prediction.ticker,
                created_at=prediction.created_at,
                signals=[hold_signal],
                rationale="Insufficient confidence in predictions"
            )

        signals = []
        current_position = None  # None, 'long', or 'short'

        for point in return_points:
            signal_action = None

            if point.expected_value >= min_return:
                # Positive expected return - consider buying
                if current_position != 'long':
                    signal_action = TradeAction.BUY
                    current_position = 'long'

            elif point.expected_value <= -min_return:
                # Negative expected return - consider selling
                if current_position == 'long':
                    signal_action = TradeAction.SELL
                    current_position = None

            if signal_action:
                signal = TradeSignal(
                    ticker=prediction.ticker,
                    action=signal_action,
                    signal_date=point.date,
                    confidence=point.confidence,
                    rationale=f"Expected return: {point.expected_value:.3f}",
                    expected_return=point.expected_value
                )
                signals.append(signal)

        # If we end with a long position, add a sell signal
        if current_position == 'long' and signals:
            last_date = return_points[-1].date
            sell_signal = TradeSignal(
                ticker=prediction.ticker,
                action=TradeAction.SELL,
                signal_date=last_date,
                confidence=0.7,
                rationale="End of prediction period"
            )
            signals.append(sell_signal)

        # Calculate expected total return
        buy_signals = [s for s in signals if s.action == TradeAction.BUY]
        sell_signals = [s for s in signals if s.action == TradeAction.SELL]

        expected_total_return = None
        if buy_signals and sell_signals:
            # Simple calculation: sum of expected returns for buy signals
            expected_returns = [
                s.expected_return for s in buy_signals
                if s.expected_return is not None
            ]
            if expected_returns:
                expected_total_return = float(np.sum(expected_returns))

        if not signals:
            # Fallback HOLD signal
            hold_signal = TradeSignal(
                ticker=prediction.ticker,
                action=TradeAction.HOLD,
                signal_date=datetime.now().date(),
                confidence=0.5,
                rationale="No actionable signals generated"
            )
            signals = [hold_signal]

        return TradePlan(
            ticker=prediction.ticker,
            created_at=prediction.created_at,
            signals=signals,
            rationale=f"Generated {len(signals)} signals based on prediction confidence >= {min_confidence}",
            expected_total_return=expected_total_return
        )

    def optimize_trade_plan(
        self,
        trade_plan: TradePlan,
        market_data: MarketData,
        transaction_cost: float = 0.001
    ) -> TradePlan:
        """Optimize a trade plan considering transaction costs and market conditions."""
        logger.debug("Optimizing trade plan for %s", trade_plan.ticker)

        # Simple optimization: remove signals that don't meet cost thresholds
        optimized_signals = []

        for signal in trade_plan.signals:
            if signal.action == TradeAction.HOLD:
                optimized_signals.append(signal)
                continue

            # Check if expected return justifies transaction costs
            if signal.expected_return is not None:
                if abs(signal.expected_return) > transaction_cost * 2:
                    optimized_signals.append(signal)
                else:
                    logger.debug(
                        "Removing signal due to low expected return vs transaction cost: %.4f",
                        signal.expected_return
                    )

        if not optimized_signals:
            # Add a HOLD signal if all signals were removed
            hold_signal = TradeSignal(
                ticker=trade_plan.ticker,
                action=TradeAction.HOLD,
                signal_date=datetime.now().date(),
                confidence=0.5,
                rationale="All signals removed due to transaction cost optimization"
            )
            optimized_signals = [hold_signal]

        # Recalculate expected total return
        optimized_buy_signals = [s for s in optimized_signals if s.action == TradeAction.BUY]
        expected_total_return = None

        if optimized_buy_signals:
            expected_returns = [
                s.expected_return for s in optimized_buy_signals
                if s.expected_return is not None
            ]
            if expected_returns:
                # Subtract transaction costs
                gross_return = float(np.sum(expected_returns))
                num_transactions = len([s for s in optimized_signals if s.action != TradeAction.HOLD])
                total_cost = num_transactions * transaction_cost
                expected_total_return = gross_return - total_cost

        return TradePlan(
            ticker=trade_plan.ticker,
            created_at=trade_plan.created_at,
            signals=optimized_signals,
            rationale=f"Optimized plan with {len(optimized_signals)} signals (transaction cost: {transaction_cost:.3f})",
            expected_total_return=expected_total_return
        )


class RiskService:
    """Domain service for risk assessment and management."""

    def calculate_portfolio_risk(
        self,
        trade_plans: List[TradePlan],
        market_data: dict[TickerSymbol, MarketData]
    ) -> dict:
        """Calculate portfolio-level risk metrics."""
        logger.debug("Calculating portfolio risk for %d trade plans", len(trade_plans))

        if not trade_plans:
            return {'total_risk': 0.0, 'diversification_score': 0.0}

        # Calculate individual position risks
        position_risks = []
        expected_returns = []

        for plan in trade_plans:
            if plan.ticker in market_data:
                data = market_data[plan.ticker]
                volatility = data.get_volatility()
                position_risks.append(volatility)

                if plan.expected_total_return is not None:
                    expected_returns.append(plan.expected_total_return)

        if not position_risks:
            return {'total_risk': 0.0, 'diversification_score': 0.0}

        # Simple portfolio risk calculation (assuming equal weights)
        avg_volatility = float(np.mean(position_risks))

        # Diversification score based on number of positions
        diversification_score = min(1.0, len(trade_plans) / 10.0)

        # Portfolio risk with diversification benefit
        portfolio_risk = avg_volatility * (1.0 - diversification_score * 0.2)

        return {
            'total_risk': portfolio_risk,
            'average_volatility': avg_volatility,
            'diversification_score': diversification_score,
            'expected_portfolio_return': float(np.mean(expected_returns)) if expected_returns else 0.0,
            'sharpe_estimate': (
                float(np.mean(expected_returns)) / portfolio_risk
                if expected_returns and portfolio_risk > 0 else 0.0
            )
        }

    def assess_position_risk(
        self,
        trade_plan: TradePlan,
        market_data: MarketData,
        max_drawdown_threshold: float = 0.1
    ) -> dict:
        """Assess risk for a single position."""
        logger.debug("Assessing position risk for %s", trade_plan.ticker)

        # Calculate historical volatility
        volatility = market_data.get_volatility()

        # Calculate maximum potential loss (simplified)
        max_loss = volatility * 2.0  # 2 standard deviations

        # Risk score based on volatility and confidence
        avg_confidence = trade_plan.average_confidence
        risk_score = volatility * (1.0 - avg_confidence)

        # Check if max loss exceeds threshold
        exceeds_threshold = max_loss > max_drawdown_threshold

        return {
            'volatility': volatility,
            'max_potential_loss': max_loss,
            'risk_score': risk_score,
            'exceeds_drawdown_threshold': exceeds_threshold,
            'average_confidence': avg_confidence,
            'recommendation': (
                'HIGH_RISK' if exceeds_threshold or risk_score > 0.3
                else 'MEDIUM_RISK' if risk_score > 0.15
                else 'LOW_RISK'
            )
        }
