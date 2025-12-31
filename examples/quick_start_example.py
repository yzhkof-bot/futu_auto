#!/usr/bin/env python3
"""
Quick start example for the trend backtesting framework.
This is a simplified example to get started quickly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_fetcher import DataFetcher
from src.data.data_processor import DataProcessor
from src.strategies.ma_strategy import MovingAverageStrategy
from src.backtest.engine import BacktestEngine

def quick_backtest(symbol="AAPL", start_date="2022-01-01", end_date="2023-12-31"):
    """
    Run a quick backtest with minimal setup.
    
    Args:
        symbol: Stock symbol to test
        start_date: Start date for backtest
        end_date: End date for backtest
    """
    
    print(f"Quick Backtest: {symbol} from {start_date} to {end_date}")
    print("-" * 50)
    
    # 1. Fetch and process data
    print("Fetching data...")
    data_fetcher = DataFetcher()
    data = data_fetcher.fetch_stock_data(symbol, start_date, end_date)
    
    data_processor = DataProcessor()
    processed_data = data_processor.add_technical_indicators(data)
    
    print(f"Data loaded: {len(processed_data)} days")
    
    # 2. Create strategy
    strategy = MovingAverageStrategy(
        short_period=10,
        long_period=30,
        rsi_lower=30,
        rsi_upper=70
    )
    
    # 3. Run backtest
    print("Running backtest...")
    engine = BacktestEngine(initial_capital=100000)
    results = engine.run_backtest(processed_data, strategy, symbol)
    
    # 4. Display results
    print("\nResults:")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate_pct']:.1f}%")
    
    return results

if __name__ == "__main__":
    # Run quick example
    results = quick_backtest()
    
    print("\nQuick start example completed!")
    print("For more advanced features, see complete_backtest_example.py")