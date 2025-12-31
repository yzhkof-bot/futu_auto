"""
Strategy module for trend backtesting framework.
Provides base strategy class and concrete trend-following implementations.
"""

from .base_strategy import BaseStrategy, StrategySignal
from .ma_strategy import MovingAverageStrategy
from .breakout_strategy import BreakoutStrategy
from .trend_following_strategy import TrendFollowingStrategy
from .multi_indicator_strategy import MultiIndicatorStrategy
from .multi_signal_strategy import MultiSignalStrategy, get_strategy

__all__ = [
    'BaseStrategy', 
    'StrategySignal',
    'MovingAverageStrategy', 
    'BreakoutStrategy',
    'TrendFollowingStrategy',
    'MultiIndicatorStrategy',
    'MultiSignalStrategy',
    'get_strategy'
]