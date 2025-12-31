"""
Moving Average Strategy implementation.
Enhanced version of the moving average logic from existing code.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, StrategySignal, SignalType

class MovingAverageStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy with RSI filter and ATR-based stops.
    
    This strategy generates buy signals when short MA crosses above long MA
    and RSI is in acceptable range. Sell signals when short MA crosses below
    long MA or RSI reaches extreme levels.
    """
    
    def __init__(self, 
                 short_period: int = 10,
                 long_period: int = 30,
                 ma_type: str = 'SMA',
                 rsi_period: int = 14,
                 rsi_lower: int = 30,
                 rsi_upper: int = 70,
                 rsi_oversold: int = 20,
                 rsi_overbought: int = 80,
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0,
                 risk_reward_ratio: float = 2.0):
        """
        Initialize Moving Average Strategy.
        
        Args:
            short_period: Short moving average period
            long_period: Long moving average period
            ma_type: Type of moving average ('SMA' or 'EMA')
            rsi_period: RSI calculation period
            rsi_lower: RSI lower threshold for entry
            rsi_upper: RSI upper threshold for entry
            rsi_oversold: RSI oversold level for exit
            rsi_overbought: RSI overbought level for exit
            atr_period: ATR calculation period
            atr_multiplier: ATR multiplier for stop loss
            risk_reward_ratio: Risk to reward ratio for take profit
        """
        
        parameters = {
            'short_period': short_period,
            'long_period': long_period,
            'ma_type': ma_type,
            'rsi_period': rsi_period,
            'rsi_lower': rsi_lower,
            'rsi_upper': rsi_upper,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier,
            'risk_reward_ratio': risk_reward_ratio
        }
        
        super().__init__("MovingAverageStrategy", parameters)
    
    def _validate_parameters(self) -> None:
        """Validate strategy parameters."""
        
        params = self.parameters
        
        if params['short_period'] >= params['long_period']:
            raise ValueError("Short period must be less than long period")
        
        if params['short_period'] < 2:
            raise ValueError("Short period must be at least 2")
        
        if params['long_period'] < 5:
            raise ValueError("Long period must be at least 5")
        
        if params['ma_type'] not in ['SMA', 'EMA']:
            raise ValueError("MA type must be 'SMA' or 'EMA'")
        
        if not (0 < params['rsi_lower'] < params['rsi_upper'] < 100):
            raise ValueError("RSI thresholds must be: 0 < lower < upper < 100")
        
        if params['atr_multiplier'] <= 0:
            raise ValueError("ATR multiplier must be positive")
        
        if params['risk_reward_ratio'] <= 0:
            raise ValueError("Risk reward ratio must be positive")
    
    def generate_signals(self, data: pd.DataFrame) -> List[StrategySignal]:
        """
        Generate trading signals based on MA crossover with RSI filter.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            
        Returns:
            List of StrategySignal objects
        """
        
        # Ensure required indicators are present
        data = self._ensure_indicators(data)
        
        signals = []
        params = self.parameters
        
        short_ma_col = f"{params['ma_type']}_{params['short_period']}"
        long_ma_col = f"{params['ma_type']}_{params['long_period']}"
        rsi_col = f"RSI_{params['rsi_period']}"
        atr_col = f"ATR_{params['atr_period']}"
        
        for i in range(1, len(data)):
            current_row = data.iloc[i]
            previous_row = data.iloc[i-1]
            
            current_price = current_row['Close']
            current_time = data.index[i]
            
            # Get indicator values
            short_ma_curr = current_row[short_ma_col]
            long_ma_curr = current_row[long_ma_col]
            short_ma_prev = previous_row[short_ma_col]
            long_ma_prev = previous_row[long_ma_col]
            rsi = current_row[rsi_col]
            atr = current_row[atr_col]
            
            # Skip if any indicator is NaN
            if pd.isna([short_ma_curr, long_ma_curr, rsi, atr]).any():
                continue
            
            signal_type = SignalType.HOLD
            signal_strength = 0.0
            
            # Check for bullish crossover
            if (short_ma_prev <= long_ma_prev and 
                short_ma_curr > long_ma_curr and
                params['rsi_lower'] < rsi < params['rsi_upper']):
                
                signal_type = SignalType.BUY
                signal_strength = self._calculate_signal_strength(
                    short_ma_curr, long_ma_curr, rsi, 'buy'
                )
            
            # Check for bearish crossover or RSI extreme
            elif ((short_ma_prev >= long_ma_prev and short_ma_curr < long_ma_curr) or
                  rsi < params['rsi_oversold'] or rsi > params['rsi_overbought']):
                
                signal_type = SignalType.SELL
                signal_strength = self._calculate_signal_strength(
                    short_ma_curr, long_ma_curr, rsi, 'sell'
                )
            
            # Generate signal if not HOLD
            if signal_type != SignalType.HOLD:
                
                # Calculate stop loss and take profit
                stop_loss = self.calculate_stop_loss(
                    current_price, signal_type, atr, params['atr_multiplier']
                )
                
                take_profit = self.calculate_take_profit(
                    current_price, stop_loss, signal_type, params['risk_reward_ratio']
                )
                
                signal = StrategySignal(
                    timestamp=current_time,
                    signal_type=signal_type,
                    strength=signal_strength,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'short_ma': short_ma_curr,
                        'long_ma': long_ma_curr,
                        'rsi': rsi,
                        'atr': atr
                    }
                )
                
                signals.append(signal)
        
        return signals
    
    def _ensure_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required indicators are calculated."""
        
        data_copy = data.copy()
        params = self.parameters
        
        # Calculate moving averages if not present
        short_ma_col = f"{params['ma_type']}_{params['short_period']}"
        long_ma_col = f"{params['ma_type']}_{params['long_period']}"
        
        if short_ma_col not in data_copy.columns:
            if params['ma_type'] == 'SMA':
                data_copy[short_ma_col] = data_copy['Close'].rolling(params['short_period']).mean()
            else:  # EMA
                data_copy[short_ma_col] = data_copy['Close'].ewm(span=params['short_period']).mean()
        
        if long_ma_col not in data_copy.columns:
            if params['ma_type'] == 'SMA':
                data_copy[long_ma_col] = data_copy['Close'].rolling(params['long_period']).mean()
            else:  # EMA
                data_copy[long_ma_col] = data_copy['Close'].ewm(span=params['long_period']).mean()
        
        # Calculate RSI if not present
        rsi_col = f"RSI_{params['rsi_period']}"
        if rsi_col not in data_copy.columns:
            delta = data_copy['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(params['rsi_period']).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(params['rsi_period']).mean()
            rs = gain / loss
            data_copy[rsi_col] = 100 - (100 / (1 + rs))
        
        # Calculate ATR if not present
        atr_col = f"ATR_{params['atr_period']}"
        if atr_col not in data_copy.columns:
            data_copy['TR'] = np.maximum(
                data_copy['High'] - data_copy['Low'],
                np.maximum(
                    abs(data_copy['High'] - data_copy['Close'].shift(1)),
                    abs(data_copy['Low'] - data_copy['Close'].shift(1))
                )
            )
            data_copy[atr_col] = data_copy['TR'].rolling(params['atr_period']).mean()
        
        return data_copy
    
    def _calculate_signal_strength(self, 
                                 short_ma: float, 
                                 long_ma: float, 
                                 rsi: float, 
                                 signal_direction: str) -> float:
        """
        Calculate signal strength based on indicator values.
        
        Args:
            short_ma: Short moving average value
            long_ma: Long moving average value
            rsi: RSI value
            signal_direction: 'buy' or 'sell'
            
        Returns:
            Signal strength between 0 and 1
        """
        
        # MA separation strength (normalized)
        ma_separation = abs(short_ma - long_ma) / long_ma
        ma_strength = min(ma_separation / 0.05, 1.0)  # Cap at 5% separation
        
        # RSI strength
        if signal_direction == 'buy':
            # Stronger signal when RSI is closer to middle range
            rsi_strength = 1.0 - abs(rsi - 50) / 50
        else:
            # Stronger signal when RSI is at extremes
            if rsi > 70:
                rsi_strength = (rsi - 70) / 30
            elif rsi < 30:
                rsi_strength = (30 - rsi) / 30
            else:
                rsi_strength = 0.5
        
        # Combine strengths (weighted average)
        total_strength = (ma_strength * 0.6 + rsi_strength * 0.4)
        
        return min(max(total_strength, 0.1), 1.0)  # Ensure between 0.1 and 1.0
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        
        return {
            'short_period': (3, 20),
            'long_period': (15, 100),
            'rsi_lower': (20, 40),
            'rsi_upper': (60, 80),
            'atr_multiplier': (1.0, 4.0),
            'risk_reward_ratio': (1.0, 4.0)
        }
    
    def get_current_signals_summary(self, data: pd.DataFrame) -> Dict:
        """
        Get current market condition summary.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Dictionary with current signal information
        """
        
        if len(data) == 0:
            return {}
        
        data = self._ensure_indicators(data)
        latest = data.iloc[-1]
        params = self.parameters
        
        short_ma_col = f"{params['ma_type']}_{params['short_period']}"
        long_ma_col = f"{params['ma_type']}_{params['long_period']}"
        rsi_col = f"RSI_{params['rsi_period']}"
        
        short_ma = latest[short_ma_col]
        long_ma = latest[long_ma_col]
        rsi = latest[rsi_col]
        
        # Determine trend
        if short_ma > long_ma:
            trend = "Bullish"
        elif short_ma < long_ma:
            trend = "Bearish"
        else:
            trend = "Neutral"
        
        # RSI condition
        if rsi > params['rsi_overbought']:
            rsi_condition = "Overbought"
        elif rsi < params['rsi_oversold']:
            rsi_condition = "Oversold"
        elif params['rsi_lower'] < rsi < params['rsi_upper']:
            rsi_condition = "Neutral"
        else:
            rsi_condition = "Extreme"
        
        return {
            'timestamp': data.index[-1],
            'price': latest['Close'],
            'trend': trend,
            'short_ma': short_ma,
            'long_ma': long_ma,
            'ma_separation_pct': ((short_ma - long_ma) / long_ma) * 100,
            'rsi': rsi,
            'rsi_condition': rsi_condition,
            'signal_ready': (trend == "Bullish" and rsi_condition == "Neutral")
        }