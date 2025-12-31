#!/usr/bin/env python3
"""
Strategy optimization example showing parameter tuning and walk-forward analysis.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

from src.data.data_fetcher import DataFetcher
from src.data.data_processor import DataProcessor
from src.strategies.ma_strategy import MovingAverageStrategy
from src.backtest.engine import BacktestEngine

def optimize_ma_strategy(data, param_ranges):
    """
    Optimize moving average strategy parameters using grid search.
    
    Args:
        data: Processed market data
        param_ranges: Dictionary with parameter ranges
        
    Returns:
        DataFrame with optimization results
    """
    
    print("Running parameter optimization...")
    
    results = []
    total_combinations = np.prod([len(v) for v in param_ranges.values()])
    current = 0
    
    # Grid search over all parameter combinations
    for short_period, long_period, rsi_lower, rsi_upper in product(
        param_ranges['short_period'],
        param_ranges['long_period'], 
        param_ranges['rsi_lower'],
        param_ranges['rsi_upper']
    ):
        current += 1
        
        # Skip invalid combinations
        if short_period >= long_period:
            continue
            
        if current % 10 == 0:
            print(f"Progress: {current}/{total_combinations}")
        
        try:
            # Create strategy with current parameters
            strategy = MovingAverageStrategy(
                short_period=short_period,
                long_period=long_period,
                rsi_lower=rsi_lower,
                rsi_upper=rsi_upper
            )
            
            # Run backtest
            engine = BacktestEngine(initial_capital=100000)
            backtest_result = engine.run_backtest(data, strategy, "TEST")
            
            # Store results
            results.append({
                'short_period': short_period,
                'long_period': long_period,
                'rsi_lower': rsi_lower,
                'rsi_upper': rsi_upper,
                'total_return': backtest_result['total_return_pct'],
                'sharpe_ratio': backtest_result['sharpe_ratio'],
                'max_drawdown': backtest_result['max_drawdown_pct'],
                'num_trades': backtest_result['total_trades'],
                'win_rate': backtest_result['win_rate_pct']
            })
            
        except Exception as e:
            print(f"Error with parameters {short_period}/{long_period}: {e}")
            continue
    
    return pd.DataFrame(results)

def walk_forward_analysis(data, best_params, optimization_window=252, test_window=63):
    """
    Perform walk-forward analysis to test parameter stability.
    
    Args:
        data: Market data
        best_params: Best parameters from optimization
        optimization_window: Days for parameter optimization
        test_window: Days for out-of-sample testing
        
    Returns:
        DataFrame with walk-forward results
    """
    
    print("Running walk-forward analysis...")
    
    results = []
    start_idx = optimization_window
    
    while start_idx + test_window < len(data):
        print(f"Testing period: {data.index[start_idx]} to {data.index[start_idx + test_window]}")
        
        # Test period data
        test_data = data.iloc[start_idx:start_idx + test_window]
        
        # Create strategy with best parameters
        strategy = MovingAverageStrategy(**best_params)
        
        # Run backtest on test period
        engine = BacktestEngine(initial_capital=100000)
        backtest_result = engine.run_backtest(test_data, strategy, "WF_TEST")
        
        # Store results
        results.append({
            'start_date': test_data.index[0],
            'end_date': test_data.index[-1],
            'total_return': backtest_result['total_return_pct'],
            'sharpe_ratio': backtest_result['sharpe_ratio'],
            'max_drawdown': backtest_result['max_drawdown_pct'],
            'num_trades': backtest_result['total_trades']
        })
        
        start_idx += test_window
    
    return pd.DataFrame(results)

def main():
    """Main optimization example."""
    
    print("STRATEGY OPTIMIZATION EXAMPLE")
    print("=" * 50)
    
    # Configuration
    symbol = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    
    # Fetch and process data
    print(f"Fetching data for {symbol}...")
    data_fetcher = DataFetcher()
    data = data_fetcher.fetch_stock_data(symbol, start_date, end_date)
    
    data_processor = DataProcessor()
    processed_data = data_processor.add_technical_indicators(data)
    
    print(f"Data loaded: {len(processed_data)} days")
    
    # Define parameter ranges for optimization
    param_ranges = {
        'short_period': [5, 10, 15, 20],
        'long_period': [20, 30, 40, 50],
        'rsi_lower': [25, 30, 35],
        'rsi_upper': [65, 70, 75]
    }
    
    print(f"Parameter combinations to test: {np.prod([len(v) for v in param_ranges.values()])}")
    
    # Run optimization
    optimization_results = optimize_ma_strategy(processed_data, param_ranges)
    
    if len(optimization_results) == 0:
        print("No successful optimization results!")
        return
    
    # Find best parameters
    best_by_sharpe = optimization_results.loc[optimization_results['sharpe_ratio'].idxmax()]
    best_by_return = optimization_results.loc[optimization_results['total_return'].idxmax()]
    
    print("\nOptimization Results:")
    print("-" * 30)
    print("Best by Sharpe Ratio:")
    print(f"  Parameters: Short={best_by_sharpe['short_period']}, Long={best_by_sharpe['long_period']}, RSI={best_by_sharpe['rsi_lower']}-{best_by_sharpe['rsi_upper']}")
    print(f"  Sharpe Ratio: {best_by_sharpe['sharpe_ratio']:.3f}")
    print(f"  Total Return: {best_by_sharpe['total_return']:.2f}%")
    print(f"  Max Drawdown: {best_by_sharpe['max_drawdown']:.2f}%")
    
    print("\nBest by Total Return:")
    print(f"  Parameters: Short={best_by_return['short_period']}, Long={best_by_return['long_period']}, RSI={best_by_return['rsi_lower']}-{best_by_return['rsi_upper']}")
    print(f"  Total Return: {best_by_return['total_return']:.2f}%")
    print(f"  Sharpe Ratio: {best_by_return['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {best_by_return['max_drawdown']:.2f}%")
    
    # Create optimization heatmap
    print("\nCreating optimization heatmap...")
    try:
        # Pivot table for heatmap (Short MA vs Long MA, averaged across RSI parameters)
        pivot_data = optimization_results.groupby(['short_period', 'long_period'])['sharpe_ratio'].mean().unstack()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto')
        plt.colorbar(label='Sharpe Ratio')
        plt.title('Parameter Optimization Heatmap - Sharpe Ratio')
        plt.xlabel('Long MA Period')
        plt.ylabel('Short MA Period')
        
        # Set tick labels
        plt.xticks(range(len(pivot_data.columns)), pivot_data.columns)
        plt.yticks(range(len(pivot_data.index)), pivot_data.index)
        
        # Add text annotations
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                plt.text(j, i, f'{pivot_data.iloc[i, j]:.2f}', 
                        ha='center', va='center', color='black', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('./reports/optimization_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Heatmap saved to ./reports/optimization_heatmap.png")
        
    except Exception as e:
        print(f"Error creating heatmap: {e}")
    
    # Walk-forward analysis
    print("\nRunning walk-forward analysis...")
    
    best_params = {
        'short_period': int(best_by_sharpe['short_period']),
        'long_period': int(best_by_sharpe['long_period']),
        'rsi_lower': int(best_by_sharpe['rsi_lower']),
        'rsi_upper': int(best_by_sharpe['rsi_upper'])
    }
    
    wf_results = walk_forward_analysis(processed_data, best_params)
    
    if len(wf_results) > 0:
        print("\nWalk-Forward Analysis Results:")
        print(f"Average Return: {wf_results['total_return'].mean():.2f}%")
        print(f"Average Sharpe: {wf_results['sharpe_ratio'].mean():.3f}")
        print(f"Return Std Dev: {wf_results['total_return'].std():.2f}%")
        print(f"Positive Periods: {(wf_results['total_return'] > 0).sum()}/{len(wf_results)}")
        
        # Plot walk-forward results
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(wf_results['start_date'], wf_results['total_return'], marker='o')
        plt.title('Walk-Forward Returns by Period')
        plt.ylabel('Return (%)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(wf_results['start_date'], wf_results['sharpe_ratio'], marker='o', color='orange')
        plt.title('Walk-Forward Sharpe Ratio by Period')
        plt.ylabel('Sharpe Ratio')
        plt.xlabel('Period Start Date')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./reports/walk_forward_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Walk-forward chart saved to ./reports/walk_forward_analysis.png")
    
    # Save detailed results
    optimization_results.to_csv('./reports/optimization_results.csv', index=False)
    if len(wf_results) > 0:
        wf_results.to_csv('./reports/walk_forward_results.csv', index=False)
    
    print("\nOptimization example completed!")
    print("Results saved to ./reports/")

if __name__ == "__main__":
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    
    main()