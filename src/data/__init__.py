"""
Data module for trend backtesting framework.
Provides robust data fetching, caching, and preprocessing capabilities.
"""

from .data_fetcher import DataFetcher
from .data_processor import DataProcessor

__all__ = ['DataFetcher', 'DataProcessor']