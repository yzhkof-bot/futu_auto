"""
Base strategy class providing a standardized interface for all trading strategies.
Improves upon the existing trend_filter_strategy.py with better modularity.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import pandas as pd
import numpy as np

class SignalType(Enum):
    """Enumeration for signal types."""
    BUY = 1
    SELL = -1
    HOLD = 0

@dataclass
class StrategySignal:
    """
    Data class representing a trading signal.
    """
    timestamp: pd.Timestamp
    signal_type: SignalType
    strength: float  # Signal strength (0-1)
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    metadata: Optional[Dict] = None

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class provides a standardized interface and common functionality
    that all strategies should implement or can inherit.
    """
    
    def __init__(self, name: str, parameters: Dict = None):
        """
        Initialize the base strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy-specific parameters
        """
        self.name = name
        self.parameters = parameters or {}
        self.signals = []
        self.current_position = 0
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trade_history = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Validate parameters
        self._validate_parameters()
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[StrategySignal]:
        """
        Generate trading signals based on market data.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            
        Returns:
            List of StrategySignal objects
        """
        pass
    
    @abstractmethod
    def _validate_parameters(self) -> None:
        """
        Validate strategy parameters.
        Should raise ValueError if parameters are invalid.
        """
        pass
    
    def calculate_position_size(self, 
                              signal: StrategySignal, 
                              account_balance: float,
                              risk_per_trade: float = 0.02) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            signal: Trading signal
            account_balance: Current account balance
            risk_per_trade: Risk per trade as fraction of balance
            
        Returns:
            Position size (number of shares or contracts)
        """
        
        if signal.stop_loss is None:
            # Default to 2% risk if no stop loss specified
            return account_balance * risk_per_trade / signal.price
        
        # Calculate position size based on stop loss
        risk_amount = account_balance * risk_per_trade
        risk_per_share = abs(signal.price - signal.stop_loss)
        
        if risk_per_share > 0:
            position_size = risk_amount / risk_per_share
        else:
            position_size = account_balance * 0.1 / signal.price  # Default 10% allocation
        
        return position_size
    
    def calculate_stop_loss(self, 
                          entry_price: float, 
                          signal_type: SignalType,
                          atr: Optional[float] = None,
                          atr_multiplier: float = 2.0) -> float:
        """
        Calculate stop loss level.
        
        Args:
            entry_price: Entry price
            signal_type: Buy or sell signal
            atr: Average True Range value
            atr_multiplier: ATR multiplier for stop distance
            
        Returns:
            Stop loss price
        """
        
        if atr is not None:
            stop_distance = atr * atr_multiplier
        else:
            # Default to 2% stop loss
            stop_distance = entry_price * 0.02
        
        if signal_type == SignalType.BUY:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def calculate_take_profit(self, 
                            entry_price: float, 
                            stop_loss: float,
                            signal_type: SignalType,
                            risk_reward_ratio: float = 2.0) -> float:
        """
        Calculate take profit level based on risk-reward ratio.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            signal_type: Buy or sell signal
            risk_reward_ratio: Risk to reward ratio
            
        Returns:
            Take profit price
        """
        
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio
        
        if signal_type == SignalType.BUY:
            return entry_price + reward
        else:
            return entry_price - reward
    
    def update_trailing_stop(self, 
                           current_price: float, 
                           signal_type: SignalType,
                           trailing_percent: float = 0.05) -> Optional[float]:
        """
        Update trailing stop loss.
        
        Args:
            current_price: Current market price
            signal_type: Current position type
            trailing_percent: Trailing stop percentage
            
        Returns:
            New stop loss price or None if no update
        """
        
        if self.stop_loss is None:
            return None
        
        if signal_type == SignalType.BUY:
            # For long positions, trail stop up
            new_stop = current_price * (1 - trailing_percent)
            if new_stop > self.stop_loss:
                self.stop_loss = new_stop
                return new_stop
        else:
            # For short positions, trail stop down
            new_stop = current_price * (1 + trailing_percent)
            if new_stop < self.stop_loss:
                self.stop_loss = new_stop
                return new_stop
        
        return None
    
    def should_exit_position(self, 
                           current_price: float, 
                           current_data: pd.Series) -> Tuple[bool, str]:
        """
        Check if current position should be exited.
        
        Args:
            current_price: Current market price
            current_data: Current market data row
            
        Returns:
            Tuple of (should_exit, reason)
        """
        
        if self.current_position == 0:
            return False, "No position"
        
        # Check stop loss
        if self.stop_loss is not None:
            if self.current_position > 0 and current_price <= self.stop_loss:
                return True, "Stop loss hit"
            elif self.current_position < 0 and current_price >= self.stop_loss:
                return True, "Stop loss hit"
        
        # Check take profit
        if self.take_profit is not None:
            if self.current_position > 0 and current_price >= self.take_profit:
                return True, "Take profit hit"
            elif self.current_position < 0 and current_price <= self.take_profit:
                return True, "Take profit hit"
        
        return False, "Continue holding"
    
    def enter_position(self, signal: StrategySignal) -> None:
        """
        Enter a new position based on signal.
        
        Args:
            signal: Trading signal to execute
        """
        
        self.current_position = signal.position_size or 1
        self.entry_price = signal.price
        self.stop_loss = signal.stop_loss
        self.take_profit = signal.take_profit
        
        # Record trade entry
        trade_record = {
            'timestamp': signal.timestamp,
            'action': 'entry',
            'signal_type': signal.signal_type,
            'price': signal.price,
            'position_size': self.current_position,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }
        self.trade_history.append(trade_record)
    
    def exit_position(self, exit_price: float, exit_time: pd.Timestamp, reason: str) -> Dict:
        """
        Exit current position.
        
        Args:
            exit_price: Exit price
            exit_time: Exit timestamp
            reason: Reason for exit
            
        Returns:
            Trade result dictionary
        """
        
        if self.current_position == 0:
            return {}
        
        # Calculate trade result
        if self.current_position > 0:  # Long position
            pnl = (exit_price - self.entry_price) * abs(self.current_position)
            return_pct = (exit_price - self.entry_price) / self.entry_price
        else:  # Short position
            pnl = (self.entry_price - exit_price) * abs(self.current_position)
            return_pct = (self.entry_price - exit_price) / self.entry_price
        
        # Update statistics
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Record trade exit
        trade_result = {
            'entry_time': self.trade_history[-1]['timestamp'],
            'exit_time': exit_time,
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'position_size': self.current_position,
            'pnl': pnl,
            'return_pct': return_pct,
            'exit_reason': reason,
            'holding_period': (exit_time - self.trade_history[-1]['timestamp']).days
        }
        
        self.trade_history.append({
            'timestamp': exit_time,
            'action': 'exit',
            'price': exit_price,
            'reason': reason,
            'pnl': pnl,
            'return_pct': return_pct
        })
        
        # Reset position
        self.current_position = 0
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        
        return trade_result
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate strategy performance metrics.
        
        Returns:
            Dictionary with performance statistics
        """
        
        if self.total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0
            }
        
        # Extract trade results
        trade_results = [t for t in self.trade_history if t.get('action') == 'exit']
        
        if not trade_results:
            return {'total_trades': 0, 'win_rate': 0}
        
        returns = [t['return_pct'] for t in trade_results]
        pnls = [t['pnl'] for t in trade_results]
        
        win_rate = self.winning_trades / self.total_trades
        avg_return = np.mean(returns)
        total_return = np.sum(returns)
        avg_win = np.mean([r for r in returns if r > 0]) if self.winning_trades > 0 else 0
        avg_loss = np.mean([r for r in returns if r < 0]) if self.losing_trades > 0 else 0
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'total_pnl': sum(pnls)
        }
    
    def reset_strategy(self) -> None:
        """Reset strategy state for new backtest."""
        
        self.signals = []
        self.current_position = 0
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trade_history = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get parameter bounds for optimization.
        Should be implemented by concrete strategies.
        
        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
        return {}
    
    def set_parameters(self, parameters: Dict) -> None:
        """
        Update strategy parameters.
        
        Args:
            parameters: New parameter values
        """
        self.parameters.update(parameters)
        self._validate_parameters()
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"{self.name}({self.parameters})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(name='{self.name}', parameters={self.parameters})"