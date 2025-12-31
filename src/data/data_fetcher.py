"""
Enhanced data fetcher with robust error handling, caching, and data validation.
Improves upon the basic yfinance usage in demo_yfinance.py.
"""

import os
import sys
import time
import pickle
import hashlib
import warnings
import functools
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

warnings.filterwarnings('ignore')

# 强制刷新输出
print = functools.partial(print, flush=True)

class DataFetcher:
    """
    Robust data fetcher with caching, error handling, and data validation.
    """
    
    def __init__(self, cache_dir: str = ".cache", cache_expiry_hours: int = 24):
        """
        Initialize DataFetcher with caching configuration.
        
        Args:
            cache_dir: Directory to store cached data
            cache_expiry_hours: Hours after which cached data expires
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_expiry_hours = cache_expiry_hours
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
    def fetch_stock_data(self, 
                        symbol: str, 
                        start_date: str, 
                        end_date: str,
                        interval: str = '1d',
                        use_cache: bool = True,
                        validate_data: bool = True) -> pd.DataFrame:
        """
        Fetch stock data with robust error handling and caching.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1d', '1h', '5m', etc.)
            use_cache: Whether to use cached data
            validate_data: Whether to validate data quality
            
        Returns:
            DataFrame with OHLCV data and basic indicators
        """
        
        # Generate cache key
        cache_key = self._generate_cache_key(symbol, start_date, end_date, interval)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Try to load from cache first
        if use_cache and self._is_cache_valid(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                print(f"Loaded {symbol} data from cache")
                return cached_data
            except Exception as e:
                print(f"Cache read error: {e}. Fetching fresh data.")
        
        # Fetch fresh data
        data = self._fetch_with_retry(symbol, start_date, end_date, interval)
        
        if data.empty:
            raise ValueError(f"No data available for {symbol} from {start_date} to {end_date}")
        
        # Validate data quality
        if validate_data:
            data = self._validate_and_clean_data(data, symbol)
        
        # Cache the data
        if use_cache:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                print(f"Cached {symbol} data")
            except Exception as e:
                print(f"Cache write error: {e}")
        
        return data
    
    def fetch_multiple_symbols(self, 
                             symbols: List[str], 
                             start_date: str, 
                             end_date: str,
                             interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols efficiently.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval
            
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        
        results = {}
        failed_symbols = []
        
        for symbol in symbols:
            try:
                print(f"Fetching data for {symbol}...")
                data = self.fetch_stock_data(symbol, start_date, end_date, interval)
                results[symbol] = data
            except Exception as e:
                print(f"Failed to fetch {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if failed_symbols:
            print(f"Failed to fetch data for: {failed_symbols}")
        
        return results
    
    def get_benchmark_data(self, 
                          start_date: str, 
                          end_date: str,
                          benchmark: str = 'SPY') -> pd.DataFrame:
        """
        Fetch benchmark data for performance comparison.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            benchmark: Benchmark symbol (default: SPY)
            
        Returns:
            DataFrame with benchmark data
        """
        
        return self.fetch_stock_data(benchmark, start_date, end_date)
    
    def _fetch_with_retry(self, 
                         symbol: str, 
                         start_date: str, 
                         end_date: str, 
                         interval: str) -> pd.DataFrame:
        """
        Fetch data with exponential backoff retry logic.
        """
        
        for attempt in range(self.max_retries):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date, 
                    end=end_date, 
                    interval=interval,
                    auto_adjust=True,
                    prepost=False
                )
                
                if not data.empty:
                    print(f"Successfully fetched {symbol} data (attempt {attempt + 1})")
                    return data
                else:
                    print(f"No data returned for {symbol} (attempt {attempt + 1})")
                    
            except Exception as e:
                print(f"Fetch attempt {attempt + 1} failed for {symbol}: {e}")
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
        
        print(f"All retry attempts failed for {symbol}")
        return pd.DataFrame()
    
    def _validate_and_clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate and clean the fetched data.
        
        Args:
            data: Raw data from yfinance
            symbol: Stock symbol for logging
            
        Returns:
            Cleaned and validated DataFrame
        """
        
        original_length = len(data)
        
        # Remove rows with all NaN values
        data = data.dropna(how='all')
        
        # Check for minimum data requirements
        if len(data) < 50:
            raise ValueError(f"Insufficient data for {symbol}: only {len(data)} rows")
        
        # Validate OHLCV data integrity
        invalid_rows = (
            (data['High'] < data['Low']) |
            (data['High'] < data['Open']) |
            (data['High'] < data['Close']) |
            (data['Low'] > data['Open']) |
            (data['Low'] > data['Close']) |
            (data['Volume'] < 0)
        )
        
        if invalid_rows.any():
            print(f"Warning: Found {invalid_rows.sum()} invalid rows in {symbol} data")
            data = data[~invalid_rows]
        
        # Handle missing values with forward fill (limited)
        data = data.fillna(method='ffill', limit=5)
        
        # Remove any remaining NaN rows
        data = data.dropna()
        
        # Check for extreme price movements (potential data errors)
        returns = data['Close'].pct_change()
        extreme_moves = abs(returns) > 0.5  # 50% single-day moves
        
        if extreme_moves.any():
            print(f"Warning: Found {extreme_moves.sum()} extreme price movements in {symbol}")
            # Optionally remove or flag these days
        
        # Ensure minimum data after cleaning
        if len(data) < 30:
            raise ValueError(f"Insufficient clean data for {symbol}: only {len(data)} rows after cleaning")
        
        cleaned_rows = original_length - len(data)
        if cleaned_rows > 0:
            print(f"Cleaned {cleaned_rows} invalid rows from {symbol} data")
        
        return data
    
    def _generate_cache_key(self, 
                           symbol: str, 
                           start_date: str, 
                           end_date: str, 
                           interval: str) -> str:
        """
        Generate a unique cache key for the data request.
        """
        
        key_string = f"{symbol}_{start_date}_{end_date}_{interval}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """
        Check if cached data is still valid (not expired).
        """
        
        if not cache_file.exists():
            return False
        
        # Check file age
        file_age = time.time() - cache_file.stat().st_mtime
        max_age = self.cache_expiry_hours * 3600  # Convert to seconds
        
        return file_age < max_age
    
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear cached data files.
        
        Args:
            symbol: If provided, clear cache only for this symbol. 
                   If None, clear all cache.
        """
        
        if symbol:
            # Clear cache for specific symbol
            for cache_file in self.cache_dir.glob(f"*{symbol}*.pkl"):
                cache_file.unlink()
                print(f"Cleared cache for {cache_file.name}")
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            print("Cleared all cached data")
    
    def get_cache_info(self) -> Dict[str, any]:
        """
        Get information about cached data.
        
        Returns:
            Dictionary with cache statistics
        """
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cache_directory': str(self.cache_dir),
            'cached_files': len(cache_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'expiry_hours': self.cache_expiry_hours
        }
    
    def update_symbol_data(self, 
                          symbol: str, 
                          existing_data: pd.DataFrame,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Update existing data with new data points (incremental update).
        
        Args:
            symbol: Stock symbol
            existing_data: Previously fetched data
            end_date: End date for update (default: today)
            
        Returns:
            Updated DataFrame with new data appended
        """
        
        if existing_data.empty:
            raise ValueError("Existing data is empty")
        
        # Determine the last date in existing data
        last_date = existing_data.index[-1].strftime('%Y-%m-%d')
        
        # Set end date to today if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Calculate start date for incremental fetch (day after last data)
        start_date = (existing_data.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Fetch new data
        new_data = self.fetch_stock_data(
            symbol, start_date, end_date, use_cache=False
        )
        
        if new_data.empty:
            print(f"No new data available for {symbol}")
            return existing_data
        
        # Combine existing and new data
        combined_data = pd.concat([existing_data, new_data])
        
        # Remove any duplicate dates
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        
        print(f"Added {len(new_data)} new data points for {symbol}")
        return combined_data