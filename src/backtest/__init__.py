"""
Backtest module for trend backtesting framework.
Provides comprehensive backtesting engine and performance metrics calculation.
"""

from .engine import BacktestEngine
from .metrics import PerformanceMetrics
from .portfolio import Portfolio

__all__ = ['BacktestEngine', 'PerformanceMetrics', 'Portfolio']