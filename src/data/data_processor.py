"""
Data processor for technical analysis and feature engineering.
Extends the basic indicator calculations from existing code.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Advanced data processor for technical analysis and feature engineering.
    """
    
    def __init__(self):
        """Initialize the data processor."""
        pass
    
    def add_technical_indicators(self, 
                               data: pd.DataFrame, 
                               indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Add comprehensive technical indicators to the data.
        
        Args:
            data: OHLCV DataFrame
            indicators: List of indicators to calculate. If None, adds all.
            
        Returns:
            DataFrame with technical indicators added
        """
        
        if indicators is None:
            indicators = [
                'sma', 'ema', 'rsi', 'macd', 'atr', 'bollinger', 
                'stochastic', 'williams_r', 'cci', 'momentum'
            ]
        
        result_data = data.copy()
        
        # Moving Averages
        if 'sma' in indicators:
            result_data = self._add_moving_averages(result_data, ma_type='sma')
        
        if 'ema' in indicators:
            result_data = self._add_moving_averages(result_data, ma_type='ema')
        
        # Momentum Indicators
        if 'rsi' in indicators:
            result_data = self._add_rsi(result_data)
        
        if 'macd' in indicators:
            result_data = self._add_macd(result_data)
        
        if 'stochastic' in indicators:
            result_data = self._add_stochastic(result_data)
        
        if 'williams_r' in indicators:
            result_data = self._add_williams_r(result_data)
        
        if 'cci' in indicators:
            result_data = self._add_cci(result_data)
        
        if 'momentum' in indicators:
            result_data = self._add_momentum(result_data)
        
        # Volatility Indicators
        if 'atr' in indicators:
            result_data = self._add_atr(result_data)
        
        if 'bollinger' in indicators:
            result_data = self._add_bollinger_bands(result_data)
        
        # Add basic price features
        result_data = self._add_price_features(result_data)
        
        return result_data
    
    def _add_moving_averages(self, data: pd.DataFrame, ma_type: str = 'sma') -> pd.DataFrame:
        """Add various moving averages."""
        
        periods = [5, 10, 20, 50, 100, 200]
        
        for period in periods:
            if ma_type == 'sma':
                data[f'SMA_{period}'] = data['Close'].rolling(period).mean()
            elif ma_type == 'ema':
                data[f'EMA_{period}'] = data['Close'].ewm(span=period).mean()
        
        return data
    
    def _add_rsi(self, data: pd.DataFrame, periods: List[int] = [14, 21]) -> pd.DataFrame:
        """Add RSI indicators."""
        
        for period in periods:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        return data
    
    def _add_macd(self, data: pd.DataFrame, 
                  fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD indicators."""
        
        ema_fast = data['Close'].ewm(span=fast).mean()
        ema_slow = data['Close'].ewm(span=slow).mean()
        
        data['MACD'] = ema_fast - ema_slow
        data['MACD_Signal'] = data['MACD'].ewm(span=signal).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        return data
    
    def _add_atr(self, data: pd.DataFrame, periods: List[int] = [14, 21]) -> pd.DataFrame:
        """Add Average True Range indicators."""
        
        # Calculate True Range
        data['TR'] = np.maximum(
            data['High'] - data['Low'],
            np.maximum(
                abs(data['High'] - data['Close'].shift(1)),
                abs(data['Low'] - data['Close'].shift(1))
            )
        )
        
        for period in periods:
            data[f'ATR_{period}'] = data['TR'].rolling(period).mean()
        
        return data
    
    def _add_bollinger_bands(self, data: pd.DataFrame, 
                           period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Add Bollinger Bands."""
        
        sma = data['Close'].rolling(period).mean()
        std = data['Close'].rolling(period).std()
        
        data[f'BB_Upper_{period}'] = sma + (std * std_dev)
        data[f'BB_Lower_{period}'] = sma - (std * std_dev)
        data[f'BB_Middle_{period}'] = sma
        data[f'BB_Width_{period}'] = data[f'BB_Upper_{period}'] - data[f'BB_Lower_{period}']
        data[f'BB_Position_{period}'] = (data['Close'] - data[f'BB_Lower_{period}']) / data[f'BB_Width_{period}']
        
        return data
    
    def _add_stochastic(self, data: pd.DataFrame, 
                       k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        
        low_min = data['Low'].rolling(k_period).min()
        high_max = data['High'].rolling(k_period).max()
        
        data[f'Stoch_K_{k_period}'] = 100 * (data['Close'] - low_min) / (high_max - low_min)
        data[f'Stoch_D_{k_period}'] = data[f'Stoch_K_{k_period}'].rolling(d_period).mean()
        
        return data
    
    def _add_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Williams %R."""
        
        high_max = data['High'].rolling(period).max()
        low_min = data['Low'].rolling(period).min()
        
        data[f'Williams_R_{period}'] = -100 * (high_max - data['Close']) / (high_max - low_min)
        
        return data
    
    def _add_cci(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Commodity Channel Index."""
        
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        data[f'CCI_{period}'] = (typical_price - sma_tp) / (0.015 * mad)
        
        return data
    
    def _add_momentum(self, data: pd.DataFrame, periods: List[int] = [10, 20]) -> pd.DataFrame:
        """Add momentum indicators."""
        
        for period in periods:
            data[f'Momentum_{period}'] = data['Close'] / data['Close'].shift(period) - 1
            data[f'ROC_{period}'] = data['Close'].pct_change(period)
        
        return data
    
    def _add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features."""
        
        # Daily returns
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Price ranges
        data['Daily_Range'] = (data['High'] - data['Low']) / data['Close']
        data['Gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        
        # Volume features
        if 'Volume' in data.columns:
            data['Volume_SMA_20'] = data['Volume'].rolling(20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_20']
            data['Price_Volume'] = data['Close'] * data['Volume']
        
        # Volatility measures
        data['Volatility_10'] = data['Returns'].rolling(10).std() * np.sqrt(252)
        data['Volatility_20'] = data['Returns'].rolling(20).std() * np.sqrt(252)
        
        return data
    
    def calculate_support_resistance(self, 
                                   data: pd.DataFrame, 
                                   window: int = 20,
                                   min_touches: int = 2) -> Dict[str, List[float]]:
        """
        Calculate support and resistance levels using pivot points.
        
        Args:
            data: OHLCV DataFrame
            window: Window for pivot detection
            min_touches: Minimum touches required for a level
            
        Returns:
            Dictionary with support and resistance levels
        """
        
        highs = data['High']
        lows = data['Low']
        
        # Find pivot highs and lows
        pivot_highs = []
        pivot_lows = []
        
        for i in range(window, len(data) - window):
            # Check for pivot high
            if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
                pivot_highs.append(highs.iloc[i])
            
            # Check for pivot low
            if lows.iloc[i] == lows.iloc[i-window:i+window+1].min():
                pivot_lows.append(lows.iloc[i])
        
        # Cluster similar levels
        resistance_levels = self._cluster_levels(pivot_highs, min_touches)
        support_levels = self._cluster_levels(pivot_lows, min_touches)
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }
    
    def _cluster_levels(self, levels: List[float], 
                       min_touches: int, tolerance: float = 0.02) -> List[float]:
        """Cluster similar price levels."""
        
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                current_cluster.append(level)
            else:
                if len(current_cluster) >= min_touches:
                    clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        # Don't forget the last cluster
        if len(current_cluster) >= min_touches:
            clusters.append(np.mean(current_cluster))
        
        return clusters
    
    def add_regime_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime indicators (trending vs. sideways).
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with regime indicators
        """
        
        # ADX for trend strength
        data = self._add_adx(data)
        
        # Trend direction indicators
        data['Trend_SMA'] = np.where(
            data['SMA_20'] > data['SMA_50'], 1,
            np.where(data['SMA_20'] < data['SMA_50'], -1, 0)
        )
        
        # Volatility regime
        data['Vol_Regime'] = np.where(
            data['Volatility_20'] > data['Volatility_20'].rolling(60).mean(), 1, 0
        )
        
        return data
    
    def _add_adx(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average Directional Index."""
        
        # Calculate directional movement
        data['DM_Plus'] = np.where(
            (data['High'] - data['High'].shift(1)) > (data['Low'].shift(1) - data['Low']),
            np.maximum(data['High'] - data['High'].shift(1), 0), 0
        )
        
        data['DM_Minus'] = np.where(
            (data['Low'].shift(1) - data['Low']) > (data['High'] - data['High'].shift(1)),
            np.maximum(data['Low'].shift(1) - data['Low'], 0), 0
        )
        
        # Smooth the directional movements
        data['DI_Plus'] = 100 * (data['DM_Plus'].rolling(period).mean() / data[f'ATR_{period}'])
        data['DI_Minus'] = 100 * (data['DM_Minus'].rolling(period).mean() / data[f'ATR_{period}'])
        
        # Calculate ADX
        data['DX'] = 100 * abs(data['DI_Plus'] - data['DI_Minus']) / (data['DI_Plus'] + data['DI_Minus'])
        data['ADX'] = data['DX'].rolling(period).mean()
        
        return data
    
    def prepare_features_for_ml(self, 
                              data: pd.DataFrame, 
                              target_column: str = 'Returns',
                              feature_columns: Optional[List[str]] = None,
                              lookback_periods: List[int] = [1, 5, 10]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for machine learning models.
        
        Args:
            data: DataFrame with indicators
            target_column: Column to use as target variable
            feature_columns: Specific columns to use as features
            lookback_periods: Periods for creating lagged features
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        
        if feature_columns is None:
            # Auto-select numeric columns (excluding OHLCV)
            exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', target_column]
            feature_columns = [col for col in data.select_dtypes(include=[np.number]).columns 
                             if col not in exclude_cols]
        
        features = pd.DataFrame(index=data.index)
        
        # Add current features
        for col in feature_columns:
            if col in data.columns:
                features[col] = data[col]
        
        # Add lagged features
        for col in feature_columns:
            if col in data.columns:
                for lag in lookback_periods:
                    features[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        # Add rolling statistics
        for col in feature_columns:
            if col in data.columns:
                features[f'{col}_mean_5'] = data[col].rolling(5).mean()
                features[f'{col}_std_5'] = data[col].rolling(5).std()
        
        # Target variable (shifted for prediction)
        target = data[target_column].shift(-1)  # Predict next period
        
        # Remove rows with NaN values
        valid_idx = features.dropna().index.intersection(target.dropna().index)
        features = features.loc[valid_idx]
        target = target.loc[valid_idx]
        
        return features, target
    
    def calculate_correlation_matrix(self, 
                                   data: pd.DataFrame, 
                                   method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix for numeric columns.
        
        Args:
            data: DataFrame with numeric data
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation matrix DataFrame
        """
        
        numeric_data = data.select_dtypes(include=[np.number])
        return numeric_data.corr(method=method)
    
    def detect_outliers(self, 
                       data: pd.DataFrame, 
                       columns: Optional[List[str]] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers in the data.
        
        Args:
            data: DataFrame to analyze
            columns: Columns to check for outliers
            method: Method to use ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean DataFrame indicating outliers
        """
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
        
        outliers = pd.DataFrame(False, index=data.index, columns=columns)
        
        for col in columns:
            if col in data.columns:
                if method == 'iqr':
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    outliers[col] = (data[col] < lower_bound) | (data[col] > upper_bound)
                
                elif method == 'zscore':
                    z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                    outliers[col] = z_scores > threshold
        
        return outliers