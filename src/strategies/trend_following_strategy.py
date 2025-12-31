"""
Comprehensive Trend Following Strategy.
Enhanced version of the existing trend_filter_strategy.py with improved modularity.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, StrategySignal, SignalType

class TrendFollowingStrategy(BaseStrategy):
    """
    Advanced Trend Following Strategy combining multiple indicators.
    
    This strategy uses:
    - Multiple EMA trend confirmation
    - RSI momentum filtering
    - MACD signal confirmation
    - ATR-based position sizing and stops
    - Dynamic trailing stops
    """
    
    def __init__(self,
                 ema_fast: int = 10,
                 ema_medium: int = 20,
                 ema_slow: int = 50,
                 rsi_period: int = 14,
                 rsi_lower: int = 30,
                 rsi_upper: int = 70,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0,
                 trailing_stop_pct: float = 0.12,
                 min_trend_strength: float = 0.5,
                 risk_reward_ratio: float = 2.0):
        """
        Initialize Trend Following Strategy.
        
        Args:
            ema_fast: Fast EMA period
            ema_medium: Medium EMA period  
            ema_slow: Slow EMA period
            rsi_period: RSI calculation period
            rsi_lower: RSI lower threshold
            rsi_upper: RSI upper threshold
            macd_fast: MACD fast EMA period
            macd_slow: MACD slow EMA period
            macd_signal: MACD signal line period
            atr_period: ATR calculation period
            atr_multiplier: ATR multiplier for stops
            trailing_stop_pct: Trailing stop percentage
            min_trend_strength: Minimum trend strength required
            risk_reward_ratio: Risk to reward ratio
        """
        
        parameters = {
            'ema_fast': ema_fast,
            'ema_medium': ema_medium,
            'ema_slow': ema_slow,
            'rsi_period': rsi_period,
            'rsi_lower': rsi_lower,
            'rsi_upper': rsi_upper,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_signal': macd_signal,
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier,
            'trailing_stop_pct': trailing_stop_pct,
            'min_trend_strength': min_trend_strength,
            'risk_reward_ratio': risk_reward_ratio
        }
        
        super().__init__("TrendFollowingStrategy", parameters)
    
    def _validate_parameters(self) -> None:
        """Validate strategy parameters."""
        
        params = self.parameters
        
        if not (params['ema_fast'] < params['ema_medium'] < params['ema_slow']):
            raise ValueError("EMA periods must be: fast < medium < slow")
        
        if params['ema_fast'] < 3:
            raise ValueError("Fast EMA period must be at least 3")
        
        if not (0 < params['rsi_lower'] < params['rsi_upper'] < 100):
            raise ValueError("RSI thresholds must be: 0 < lower < upper < 100")
        
        if not (params['macd_fast'] < params['macd_slow']):
            raise ValueError("MACD fast period must be less than slow period")
        
        if not (0 < params['trailing_stop_pct'] < 0.5):
            raise ValueError("Trailing stop percentage must be between 0 and 0.5")
        
        if not (0 < params['min_trend_strength'] <= 1.0):
            raise ValueError("Minimum trend strength must be between 0 and 1")
        
        if params['atr_multiplier'] <= 0:
            raise ValueError("ATR multiplier must be positive")
    
    def generate_signals(self, data: pd.DataFrame) -> List[StrategySignal]:
        """
        Generate trend following signals with multiple confirmations.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            
        Returns:
            List of StrategySignal objects
        """
        
        # Ensure required indicators are present
        data = self._ensure_indicators(data)
        
        signals = []
        params = self.parameters
        
        # Column names for indicators
        ema_fast_col = f"EMA_{params['ema_fast']}"
        ema_medium_col = f"EMA_{params['ema_medium']}"
        ema_slow_col = f"EMA_{params['ema_slow']}"
        rsi_col = f"RSI_{params['rsi_period']}"
        atr_col = f"ATR_{params['atr_period']}"
        
        for i in range(max(params['ema_slow'], params['macd_slow']) + 1, len(data)):
            current_row = data.iloc[i]
            previous_row = data.iloc[i-1]
            
            current_price = current_row['Close']
            current_time = data.index[i]
            
            # Get indicator values
            ema_fast = current_row[ema_fast_col]
            ema_medium = current_row[ema_medium_col]
            ema_slow = current_row[ema_slow_col]
            rsi = current_row[rsi_col]
            macd = current_row['MACD']
            macd_signal = current_row['MACD_Signal']
            macd_hist = current_row['MACD_Histogram']
            atr = current_row[atr_col]
            
            # Skip if any indicator is NaN
            if pd.isna([ema_fast, ema_medium, ema_slow, rsi, macd, atr]).any():
                continue
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(
                current_price, ema_fast, ema_medium, ema_slow, macd_hist
            )
            
            signal_type = SignalType.HOLD
            signal_strength = 0.0
            
            # Check for bullish trend signal
            if (self._is_bullish_trend(ema_fast, ema_medium, ema_slow) and
                self._is_bullish_momentum(rsi, macd, macd_signal, macd_hist) and
                trend_strength >= params['min_trend_strength']):
                
                signal_type = SignalType.BUY
                signal_strength = self._calculate_signal_strength(
                    trend_strength, rsi, macd_hist, 'buy'
                )
            
            # Check for bearish trend signal  
            elif (self._is_bearish_trend(ema_fast, ema_medium, ema_slow) and
                  self._is_bearish_momentum(rsi, macd, macd_signal, macd_hist) and
                  trend_strength >= params['min_trend_strength']):
                
                signal_type = SignalType.SELL
                signal_strength = self._calculate_signal_strength(
                    trend_strength, rsi, macd_hist, 'sell'
                )
            
            # Generate signal if conditions met
            if signal_type != SignalType.HOLD:
                
                # Calculate stop loss using ATR
                stop_loss = self.calculate_stop_loss(
                    current_price, signal_type, atr, params['atr_multiplier']
                )
                
                # Calculate take profit
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
                        'ema_fast': ema_fast,
                        'ema_medium': ema_medium,
                        'ema_slow': ema_slow,
                        'rsi': rsi,
                        'macd': macd,
                        'macd_signal': macd_signal,
                        'macd_histogram': macd_hist,
                        'atr': atr,
                        'trend_strength': trend_strength
                    }
                )
                
                signals.append(signal)
        
        return signals
    
    def _ensure_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required indicators are calculated."""
        
        data_copy = data.copy()
        params = self.parameters
        
        # Calculate EMAs if not present
        for period in [params['ema_fast'], params['ema_medium'], params['ema_slow']]:
            ema_col = f"EMA_{period}"
            if ema_col not in data_copy.columns:
                data_copy[ema_col] = data_copy['Close'].ewm(span=period).mean()
        
        # Calculate RSI if not present
        rsi_col = f"RSI_{params['rsi_period']}"
        if rsi_col not in data_copy.columns:
            delta = data_copy['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(params['rsi_period']).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(params['rsi_period']).mean()
            rs = gain / loss
            data_copy[rsi_col] = 100 - (100 / (1 + rs))
        
        # Calculate MACD if not present
        if 'MACD' not in data_copy.columns:
            ema_fast = data_copy['Close'].ewm(span=params['macd_fast']).mean()
            ema_slow = data_copy['Close'].ewm(span=params['macd_slow']).mean()
            data_copy['MACD'] = ema_fast - ema_slow
            data_copy['MACD_Signal'] = data_copy['MACD'].ewm(span=params['macd_signal']).mean()
            data_copy['MACD_Histogram'] = data_copy['MACD'] - data_copy['MACD_Signal']
        
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
    
    def _is_bullish_trend(self, ema_fast: float, ema_medium: float, ema_slow: float) -> bool:
        """Check if EMAs are in bullish alignment."""
        return ema_fast > ema_medium > ema_slow
    
    def _is_bearish_trend(self, ema_fast: float, ema_medium: float, ema_slow: float) -> bool:
        """Check if EMAs are in bearish alignment."""
        return ema_fast < ema_medium < ema_slow
    
    def _is_bullish_momentum(self, rsi: float, macd: float, macd_signal: float, macd_hist: float) -> bool:
        """Check for bullish momentum conditions."""
        params = self.parameters
        return (rsi > params['rsi_lower'] and 
                rsi < params['rsi_upper'] and
                macd > macd_signal and
                macd_hist > 0)
    
    def _is_bearish_momentum(self, rsi: float, macd: float, macd_signal: float, macd_hist: float) -> bool:
        """Check for bearish momentum conditions."""
        params = self.parameters
        return (rsi < params['rsi_upper'] and 
                rsi > params['rsi_lower'] and
                macd < macd_signal and
                macd_hist < 0)
    
    def _calculate_trend_strength(self, 
                                price: float,
                                ema_fast: float, 
                                ema_medium: float, 
                                ema_slow: float,
                                macd_hist: float) -> float:
        """
        Calculate trend strength based on EMA separation and MACD.
        
        Returns:
            Trend strength between 0 and 1
        """
        
        # EMA separation strength
        fast_medium_sep = abs(ema_fast - ema_medium) / ema_medium
        medium_slow_sep = abs(ema_medium - ema_slow) / ema_slow
        price_fast_sep = abs(price - ema_fast) / ema_fast
        
        # Normalize separations (cap at 5%)
        ema_strength = min((fast_medium_sep + medium_slow_sep + price_fast_sep) / 0.15, 1.0)
        
        # MACD histogram strength (normalized)
        macd_strength = min(abs(macd_hist) / (price * 0.01), 1.0)  # Cap at 1% of price
        
        # Combined trend strength
        trend_strength = (ema_strength * 0.7 + macd_strength * 0.3)
        
        return min(max(trend_strength, 0.0), 1.0)
    
    def _calculate_signal_strength(self, 
                                 trend_strength: float,
                                 rsi: float,
                                 macd_hist: float,
                                 direction: str) -> float:
        """Calculate overall signal strength."""
        
        # RSI strength (closer to 50 is stronger for trend following)
        rsi_strength = 1.0 - abs(rsi - 50) / 50
        
        # MACD histogram strength
        macd_strength = min(abs(macd_hist) * 100, 1.0)  # Normalize
        
        # Combine all strengths
        total_strength = (
            trend_strength * 0.5 +
            rsi_strength * 0.3 +
            macd_strength * 0.2
        )
        
        return min(max(total_strength, 0.1), 1.0)
    
    def should_exit_position(self, 
                           current_price: float, 
                           current_data: pd.Series) -> Tuple[bool, str]:
        """
        Enhanced exit logic with trailing stops and trend reversal detection.
        
        Args:
            current_price: Current market price
            current_data: Current market data row
            
        Returns:
            Tuple of (should_exit, reason)
        """
        
        # First check base class exit conditions
        should_exit, reason = super().should_exit_position(current_price, current_data)
        
        if should_exit:
            return should_exit, reason
        
        if self.current_position == 0:
            return False, "No position"
        
        # Update trailing stop
        params = self.parameters
        signal_type = SignalType.BUY if self.current_position > 0 else SignalType.SELL
        
        new_stop = self.update_trailing_stop(
            current_price, signal_type, params['trailing_stop_pct']
        )
        
        if new_stop is not None:
            # Check if trailing stop was hit
            if signal_type == SignalType.BUY and current_price <= self.stop_loss:
                return True, "Trailing stop hit"
            elif signal_type == SignalType.SELL and current_price >= self.stop_loss:
                return True, "Trailing stop hit"
        
        # Check for trend reversal
        data = self._ensure_indicators(pd.DataFrame([current_data]).T)
        if len(data) > 0:
            latest = data.iloc[-1]
            
            ema_fast = latest[f"EMA_{params['ema_fast']}"]
            ema_medium = latest[f"EMA_{params['ema_medium']}"]
            ema_slow = latest[f"EMA_{params['ema_slow']}"]
            
            # Check for trend reversal
            if self.current_position > 0:  # Long position
                if not self._is_bullish_trend(ema_fast, ema_medium, ema_slow):
                    return True, "Trend reversal detected"
            else:  # Short position
                if not self._is_bearish_trend(ema_fast, ema_medium, ema_slow):
                    return True, "Trend reversal detected"
        
        return False, "Continue holding"
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        
        return {
            'ema_fast': (5, 20),
            'ema_medium': (15, 40),
            'ema_slow': (30, 100),
            'rsi_lower': (20, 40),
            'rsi_upper': (60, 80),
            'atr_multiplier': (1.0, 4.0),
            'trailing_stop_pct': (0.05, 0.25),
            'min_trend_strength': (0.3, 0.8),
            'risk_reward_ratio': (1.5, 4.0)
        }
    
    def get_trend_analysis(self, data: pd.DataFrame) -> Dict:
        """
        Get comprehensive trend analysis for current market conditions.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Dictionary with trend analysis
        """
        
        if len(data) == 0:
            return {}
        
        data = self._ensure_indicators(data)
        latest = data.iloc[-1]
        params = self.parameters
        
        # Get indicator values
        price = latest['Close']
        ema_fast = latest[f"EMA_{params['ema_fast']}"]
        ema_medium = latest[f"EMA_{params['ema_medium']}"]
        ema_slow = latest[f"EMA_{params['ema_slow']}"]
        rsi = latest[f"RSI_{params['rsi_period']}"]
        macd = latest['MACD']
        macd_signal = latest['MACD_Signal']
        macd_hist = latest['MACD_Histogram']
        
        # Determine trend direction
        if self._is_bullish_trend(ema_fast, ema_medium, ema_slow):
            trend_direction = "Bullish"
        elif self._is_bearish_trend(ema_fast, ema_medium, ema_slow):
            trend_direction = "Bearish"
        else:
            trend_direction = "Sideways"
        
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(
            price, ema_fast, ema_medium, ema_slow, macd_hist
        )
        
        # Momentum analysis
        momentum_bullish = self._is_bullish_momentum(rsi, macd, macd_signal, macd_hist)
        momentum_bearish = self._is_bearish_momentum(rsi, macd, macd_signal, macd_hist)
        
        if momentum_bullish:
            momentum_direction = "Bullish"
        elif momentum_bearish:
            momentum_direction = "Bearish"
        else:
            momentum_direction = "Neutral"
        
        return {
            'timestamp': data.index[-1],
            'price': price,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'momentum_direction': momentum_direction,
            'ema_alignment': {
                'fast': ema_fast,
                'medium': ema_medium,
                'slow': ema_slow,
                'bullish_alignment': self._is_bullish_trend(ema_fast, ema_medium, ema_slow),
                'bearish_alignment': self._is_bearish_trend(ema_fast, ema_medium, ema_slow)
            },
            'momentum_indicators': {
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_histogram': macd_hist
            },
            'signal_ready': (
                trend_strength >= params['min_trend_strength'] and
                (momentum_bullish or momentum_bearish)
            )
        }