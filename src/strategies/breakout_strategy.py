"""
Breakout Strategy implementation.
Enhanced version incorporating volume confirmation and multiple timeframe analysis.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, StrategySignal, SignalType

class BreakoutStrategy(BaseStrategy):
    """
    Price Breakout Strategy with volume confirmation and volatility filtering.
    
    This strategy identifies breakouts from consolidation periods using:
    - Price breakouts above/below recent highs/lows
    - Volume confirmation for breakout validity
    - ATR-based volatility filtering
    - RSI momentum confirmation
    """
    
    def __init__(self,
                 breakout_period: int = 20,
                 volume_period: int = 20,
                 min_volume_ratio: float = 1.5,
                 rsi_period: int = 14,
                 rsi_momentum_threshold: int = 50,
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0,
                 min_consolidation_days: int = 5,
                 max_consolidation_range: float = 0.05,
                 risk_reward_ratio: float = 2.5):
        """
        Initialize Breakout Strategy.
        
        Args:
            breakout_period: Period for calculating breakout levels
            volume_period: Period for volume average calculation
            min_volume_ratio: Minimum volume ratio for confirmation
            rsi_period: RSI calculation period
            rsi_momentum_threshold: RSI threshold for momentum confirmation
            atr_period: ATR calculation period
            atr_multiplier: ATR multiplier for stop loss
            min_consolidation_days: Minimum days of consolidation required
            max_consolidation_range: Maximum price range for consolidation
            risk_reward_ratio: Risk to reward ratio for take profit
        """
        
        parameters = {
            'breakout_period': breakout_period,
            'volume_period': volume_period,
            'min_volume_ratio': min_volume_ratio,
            'rsi_period': rsi_period,
            'rsi_momentum_threshold': rsi_momentum_threshold,
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier,
            'min_consolidation_days': min_consolidation_days,
            'max_consolidation_range': max_consolidation_range,
            'risk_reward_ratio': risk_reward_ratio
        }
        
        super().__init__("BreakoutStrategy", parameters)
    
    def _validate_parameters(self) -> None:
        """Validate strategy parameters."""
        
        params = self.parameters
        
        if params['breakout_period'] < 5:
            raise ValueError("Breakout period must be at least 5")
        
        if params['volume_period'] < 5:
            raise ValueError("Volume period must be at least 5")
        
        if params['min_volume_ratio'] <= 1.0:
            raise ValueError("Minimum volume ratio must be greater than 1.0")
        
        if params['min_consolidation_days'] < 3:
            raise ValueError("Minimum consolidation days must be at least 3")
        
        if not (0 < params['max_consolidation_range'] < 0.2):
            raise ValueError("Max consolidation range must be between 0 and 0.2")
        
        if params['atr_multiplier'] <= 0:
            raise ValueError("ATR multiplier must be positive")
        
        if params['risk_reward_ratio'] <= 0:
            raise ValueError("Risk reward ratio must be positive")
    
    def generate_signals(self, data: pd.DataFrame) -> List[StrategySignal]:
        """
        Generate breakout signals with volume and momentum confirmation.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            
        Returns:
            List of StrategySignal objects
        """
        
        # Ensure required indicators are present
        data = self._ensure_indicators(data)
        
        signals = []
        params = self.parameters
        
        # Calculate breakout levels
        data = self._calculate_breakout_levels(data)
        
        for i in range(params['breakout_period'], len(data)):
            current_row = data.iloc[i]
            current_price = current_row['Close']
            current_time = data.index[i]
            
            # Get indicator values
            breakout_high = current_row['Breakout_High']
            breakout_low = current_row['Breakout_Low']
            volume_ratio = current_row['Volume_Ratio']
            rsi = current_row[f"RSI_{params['rsi_period']}"]
            atr = current_row[f"ATR_{params['atr_period']}"]
            consolidation_quality = current_row['Consolidation_Quality']
            
            # Skip if any indicator is NaN
            if pd.isna([breakout_high, breakout_low, volume_ratio, rsi, atr]).any():
                continue
            
            signal_type = SignalType.HOLD
            signal_strength = 0.0
            
            # Check for bullish breakout (relaxed conditions)
            if (current_price > breakout_high and
                volume_ratio >= params['min_volume_ratio'] * 0.8 and  # Slightly relaxed volume requirement
                rsi > params['rsi_momentum_threshold'] - 5 and
                consolidation_quality > 0.2):  # Relaxed consolidation requirement
                
                signal_type = SignalType.BUY
                signal_strength = self._calculate_breakout_strength(
                    current_price, breakout_high, volume_ratio, rsi, consolidation_quality, 'buy'
                )
            
            # Check for bearish breakout (relaxed conditions)
            elif (current_price < breakout_low and
                  volume_ratio >= params['min_volume_ratio'] * 0.8 and
                  rsi < params['rsi_momentum_threshold'] + 5 and
                  consolidation_quality > 0.2):
                
                signal_type = SignalType.SELL
                signal_strength = self._calculate_breakout_strength(
                    current_price, breakout_low, volume_ratio, rsi, consolidation_quality, 'sell'
                )
            
            # Generate signal if breakout detected
            if signal_type != SignalType.HOLD:
                
                # Calculate stop loss (use breakout level as reference)
                if signal_type == SignalType.BUY:
                    stop_loss = max(breakout_high - atr * params['atr_multiplier'], breakout_low)
                else:
                    stop_loss = min(breakout_low + atr * params['atr_multiplier'], breakout_high)
                
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
                        'breakout_high': breakout_high,
                        'breakout_low': breakout_low,
                        'volume_ratio': volume_ratio,
                        'rsi': rsi,
                        'atr': atr,
                        'consolidation_quality': consolidation_quality
                    }
                )
                
                signals.append(signal)
        
        return signals
    
    def _ensure_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required indicators are calculated."""
        
        data_copy = data.copy()
        params = self.parameters
        
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
        
        # Calculate volume ratio if not present
        if 'Volume_Ratio' not in data_copy.columns and 'Volume' in data_copy.columns:
            volume_avg = data_copy['Volume'].rolling(params['volume_period']).mean()
            data_copy['Volume_Ratio'] = data_copy['Volume'] / volume_avg
        
        return data_copy
    
    def _calculate_breakout_levels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate breakout levels and consolidation quality."""
        
        params = self.parameters
        data_copy = data.copy()
        
        # Rolling highs and lows for breakout levels
        data_copy['Breakout_High'] = data_copy['High'].rolling(params['breakout_period']).max().shift(1)
        data_copy['Breakout_Low'] = data_copy['Low'].rolling(params['breakout_period']).min().shift(1)
        
        # Calculate consolidation quality with more relaxed criteria
        consolidation_scores = []
        
        for i in range(len(data_copy)):
            if i < params['breakout_period']:
                consolidation_scores.append(0.5)  # Default to moderate quality
                continue
            
            # Get recent price data
            recent_data = data_copy.iloc[i-params['breakout_period']:i]
            
            # Calculate price range relative to average price
            price_range = (recent_data['High'].max() - recent_data['Low'].min())
            avg_price = recent_data['Close'].mean()
            relative_range = price_range / avg_price
            
            # Calculate volatility consistency
            daily_ranges = (recent_data['High'] - recent_data['Low']) / recent_data['Close']
            range_consistency = max(0.1, 1.0 - daily_ranges.std() * 10)  # More forgiving
            
            # Calculate consolidation score - more relaxed threshold (15% instead of 5%)
            effective_max_range = max(params['max_consolidation_range'], 0.15)
            if relative_range <= effective_max_range:
                consolidation_score = (1.0 - relative_range / effective_max_range) * range_consistency
            else:
                # Even if range is large, give some score based on consistency
                consolidation_score = range_consistency * 0.3
            
            consolidation_scores.append(max(0.1, min(1.0, consolidation_score)))
        
        data_copy['Consolidation_Quality'] = consolidation_scores
        
        return data_copy
    
    def _calculate_breakout_strength(self, 
                                   current_price: float,
                                   breakout_level: float,
                                   volume_ratio: float,
                                   rsi: float,
                                   consolidation_quality: float,
                                   direction: str) -> float:
        """
        Calculate breakout signal strength.
        
        Args:
            current_price: Current market price
            breakout_level: Breakout level (high or low)
            volume_ratio: Volume ratio vs average
            rsi: RSI value
            consolidation_quality: Quality of prior consolidation
            direction: 'buy' or 'sell'
            
        Returns:
            Signal strength between 0 and 1
        """
        
        # Price breakout strength (how far beyond breakout level)
        if direction == 'buy':
            price_strength = min((current_price - breakout_level) / breakout_level / 0.02, 1.0)
        else:
            price_strength = min((breakout_level - current_price) / breakout_level / 0.02, 1.0)
        
        # Volume strength (normalized)
        volume_strength = min((volume_ratio - 1.0) / 2.0, 1.0)  # Cap at 3x average volume
        
        # RSI momentum strength
        if direction == 'buy':
            rsi_strength = min((rsi - 50) / 30, 1.0)  # Stronger as RSI increases above 50
        else:
            rsi_strength = min((50 - rsi) / 30, 1.0)  # Stronger as RSI decreases below 50
        
        # Consolidation quality strength
        consolidation_strength = consolidation_quality
        
        # Weighted combination
        total_strength = (
            price_strength * 0.3 +
            volume_strength * 0.3 +
            rsi_strength * 0.2 +
            consolidation_strength * 0.2
        )
        
        return min(max(total_strength, 0.1), 1.0)
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        
        return {
            'breakout_period': (10, 50),
            'min_volume_ratio': (1.2, 3.0),
            'rsi_momentum_threshold': (40, 60),
            'atr_multiplier': (1.5, 4.0),
            'min_consolidation_days': (3, 15),
            'max_consolidation_range': (0.02, 0.15),
            'risk_reward_ratio': (1.5, 4.0)
        }
    
    def identify_consolidation_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Identify current consolidation patterns in the data.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            List of consolidation pattern dictionaries
        """
        
        data = self._ensure_indicators(data)
        data = self._calculate_breakout_levels(data)
        
        patterns = []
        params = self.parameters
        
        # Look for recent consolidation patterns
        for i in range(params['breakout_period'], len(data)):
            consolidation_quality = data.iloc[i]['Consolidation_Quality']
            
            if consolidation_quality > 0.7:  # High quality consolidation
                pattern_data = data.iloc[i-params['breakout_period']:i+1]
                
                pattern = {
                    'start_date': pattern_data.index[0],
                    'end_date': pattern_data.index[-1],
                    'duration_days': len(pattern_data),
                    'high': pattern_data['High'].max(),
                    'low': pattern_data['Low'].min(),
                    'range_pct': ((pattern_data['High'].max() - pattern_data['Low'].min()) / 
                                pattern_data['Close'].mean()) * 100,
                    'quality_score': consolidation_quality,
                    'avg_volume': pattern_data['Volume'].mean() if 'Volume' in pattern_data.columns else None,
                    'breakout_levels': {
                        'resistance': data.iloc[i]['Breakout_High'],
                        'support': data.iloc[i]['Breakout_Low']
                    }
                }
                
                patterns.append(pattern)
        
        return patterns
    
    def get_current_breakout_status(self, data: pd.DataFrame) -> Dict:
        """
        Get current breakout status and levels.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Dictionary with current breakout information
        """
        
        if len(data) == 0:
            return {}
        
        data = self._ensure_indicators(data)
        data = self._calculate_breakout_levels(data)
        
        latest = data.iloc[-1]
        params = self.parameters
        
        current_price = latest['Close']
        breakout_high = latest['Breakout_High']
        breakout_low = latest['Breakout_Low']
        consolidation_quality = latest['Consolidation_Quality']
        
        # Calculate distances to breakout levels
        distance_to_resistance = ((breakout_high - current_price) / current_price) * 100
        distance_to_support = ((current_price - breakout_low) / current_price) * 100
        
        # Determine current status
        if current_price > breakout_high:
            status = "Bullish Breakout"
        elif current_price < breakout_low:
            status = "Bearish Breakout"
        elif consolidation_quality > 0.6:
            status = "Consolidating"
        else:
            status = "Trending"
        
        return {
            'timestamp': data.index[-1],
            'current_price': current_price,
            'status': status,
            'breakout_high': breakout_high,
            'breakout_low': breakout_low,
            'distance_to_resistance_pct': distance_to_resistance,
            'distance_to_support_pct': distance_to_support,
            'consolidation_quality': consolidation_quality,
            'volume_ratio': latest.get('Volume_Ratio', None),
            'rsi': latest.get(f"RSI_{params['rsi_period']}", None)
        }