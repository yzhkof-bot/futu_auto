#!/usr/bin/env python3
"""
Complete backtesting example demonstrating the full trend backtesting framework.
This example shows how to use all components together for comprehensive strategy analysis.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import framework components
from src.data.data_fetcher import DataFetcher
from src.data.data_processor import DataProcessor
from src.strategies.ma_strategy import MovingAverageStrategy
from src.strategies.breakout_strategy import BreakoutStrategy
from src.strategies.trend_following_strategy import TrendFollowingStrategy
from src.backtest.engine import BacktestEngine
from src.analytics.analyzer import StrategyAnalyzer
from src.visualization.charts import ChartGenerator
from src.visualization.reports import ReportGenerator

def main():
    """Main example execution."""
    
    print("="*60)
    print("TREND BACKTESTING FRAMEWORK - COMPLETE EXAMPLE")
    print("="*60)
    
    # Configuration
    symbol = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    initial_capital = 100000
    
    print(f"\nTesting Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,}")
    
    # Step 1: Data Fetching and Processing
    print("\n" + "="*40)
    print("STEP 1: DATA PREPARATION")
    print("="*40)
    
    # Initialize data components
    data_fetcher = DataFetcher(cache_dir=".cache")
    data_processor = DataProcessor()
    
    # Fetch market data
    print(f"Fetching data for {symbol}...")
    try:
        raw_data = data_fetcher.fetch_stock_data(symbol, start_date, end_date)
        print(f"✓ Successfully fetched {len(raw_data)} days of data")
    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        return
    
    # Process data with technical indicators
    print("Processing technical indicators...")
    processed_data = data_processor.add_technical_indicators(raw_data)
    print(f"✓ Added technical indicators. Data shape: {processed_data.shape}")
    
    # Step 2: Strategy Testing
    print("\n" + "="*40)
    print("STEP 2: STRATEGY BACKTESTING")
    print("="*40)
    
    # Initialize backtesting engine
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=0.001,  # 0.1%
        slippage_rate=0.0005,   # 0.05%
        use_kelly_sizing=True,
        kelly_scaling=0.25      # Quarter Kelly
    )
    
    # Define strategies to test
    strategies = {
        "Moving Average": MovingAverageStrategy(
            short_period=10,
            long_period=30,
            ma_type='SMA',
            rsi_lower=30,
            rsi_upper=70
        ),
        "Breakout": BreakoutStrategy(
            breakout_period=20,
            min_volume_ratio=1.5,
            rsi_momentum_threshold=50
        ),
        "Trend Following": TrendFollowingStrategy(
            ema_fast=10,
            ema_medium=20,
            ema_slow=50,
            rsi_lower=30,
            rsi_upper=70
        )
    }
    
    # Run backtests
    results = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"\nTesting {strategy_name} Strategy...")
        try:
            backtest_result = engine.run_backtest(processed_data, strategy, symbol)
            results[strategy_name] = backtest_result
            
            final_value = backtest_result['final_value']
            total_return = backtest_result['total_return_pct']
            total_trades = backtest_result['total_trades']
            
            print(f"  ✓ Final Value: ${final_value:,.2f}")
            print(f"  ✓ Total Return: {total_return:.2f}%")
            print(f"  ✓ Number of Trades: {total_trades}")
            
        except Exception as e:
            print(f"  ✗ Error testing {strategy_name}: {e}")
            continue
    
    if not results:
        print("No successful backtests to analyze.")
        return
    
    # Step 3: Performance Analysis
    print("\n" + "="*40)
    print("STEP 3: PERFORMANCE ANALYSIS")
    print("="*40)
    
    analyzer = StrategyAnalyzer()
    
    # Get benchmark data for comparison
    print("Fetching benchmark data (SPY)...")
    try:
        benchmark_data = data_fetcher.fetch_stock_data("SPY", start_date, end_date)
        benchmark_prices = benchmark_data['Close']
        print("✓ Benchmark data loaded")
    except Exception as e:
        print(f"⚠ Could not load benchmark: {e}")
        benchmark_prices = None
    
    # Analyze each strategy
    analysis_results = {}
    
    for strategy_name, backtest_result in results.items():
        print(f"\nAnalyzing {strategy_name}...")
        
        try:
            analysis = analyzer.analyze_strategy(backtest_result, benchmark_prices)
            analysis_results[strategy_name] = analysis
            
            # Print key metrics
            performance = analysis.get('performance_metrics', {})
            summary = analysis.get('summary', {})
            
            print(f"  Overall Rating: {summary.get('overall_rating', 'N/A')}")
            print(f"  Sharpe Ratio: {performance.get('Sharpe Ratio', 0):.3f}")
            print(f"  Max Drawdown: {performance.get('Max Drawdown (%)', 0):.2f}%")
            print(f"  Win Rate: {performance.get('Win Rate (%)', 0):.1f}%")
            
        except Exception as e:
            print(f"  ✗ Error analyzing {strategy_name}: {e}")
    
    # Step 4: Visualization and Reporting
    print("\n" + "="*40)
    print("STEP 4: VISUALIZATION & REPORTING")
    print("="*40)
    
    # Initialize visualization components
    chart_generator = ChartGenerator(style='professional')
    report_generator = ReportGenerator(output_dir="./reports")
    
    # Create charts for best performing strategy
    if results:
        # Find best strategy by Sharpe ratio
        best_strategy = None
        best_sharpe = -999
        
        for name, analysis in analysis_results.items():
            sharpe = analysis.get('performance_metrics', {}).get('Sharpe Ratio', -999)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_strategy = name
        
        if best_strategy:
            print(f"\nGenerating charts for best strategy: {best_strategy}")
            
            best_result = results[best_strategy]
            equity_curve = best_result.get('equity_curve', pd.Series())
            trades = best_result.get('trades', [])
            
            try:
                # Equity curve chart
                fig1 = chart_generator.create_equity_curve(
                    equity_curve, 
                    benchmark_prices,
                    title=f"{best_strategy} - Equity Curve",
                    save_path=f"./reports/{best_strategy}_equity_curve.png"
                )
                print("  ✓ Equity curve chart saved")
                
                # Drawdown chart
                fig2 = chart_generator.create_drawdown_chart(
                    equity_curve,
                    title=f"{best_strategy} - Drawdown Analysis",
                    save_path=f"./reports/{best_strategy}_drawdown.png"
                )
                print("  ✓ Drawdown chart saved")
                
                # Trade analysis chart
                if trades:
                    fig3 = chart_generator.create_trade_analysis_chart(
                        trades,
                        title=f"{best_strategy} - Trade Analysis",
                        save_path=f"./reports/{best_strategy}_trades.png"
                    )
                    print("  ✓ Trade analysis chart saved")
                
                # Returns distribution
                if len(equity_curve) > 1:
                    returns = equity_curve.pct_change().dropna()
                    fig4 = chart_generator.create_returns_distribution(
                        returns,
                        title=f"{best_strategy} - Returns Distribution",
                        save_path=f"./reports/{best_strategy}_returns.png"
                    )
                    print("  ✓ Returns distribution chart saved")
                
            except Exception as e:
                print(f"  ⚠ Error creating charts: {e}")
    
    # Generate comprehensive reports
    print("\nGenerating reports...")
    
    for strategy_name in results.keys():
        if strategy_name in analysis_results:
            try:
                # HTML report
                html_path = report_generator.generate_html_report(
                    analysis_results[strategy_name],
                    results[strategy_name],
                    strategy_name
                )
                print(f"  ✓ HTML report saved: {html_path}")
                
                # CSV export
                csv_path = report_generator.generate_csv_export(
                    results[strategy_name],
                    strategy_name
                )
                print(f"  ✓ CSV export saved: {csv_path}")
                
                # Summary report
                summary_path = report_generator.generate_summary_report(
                    analysis_results[strategy_name],
                    strategy_name
                )
                print(f"  ✓ Summary report saved: {summary_path}")
                
            except Exception as e:
                print(f"  ⚠ Error generating reports for {strategy_name}: {e}")
    
    # Step 5: Strategy Comparison
    print("\n" + "="*40)
    print("STEP 5: STRATEGY COMPARISON")
    print("="*40)
    
    if len(results) > 1:
        try:
            # Create comparison chart
            comparison_results = {}
            for name, analysis in analysis_results.items():
                comparison_results[name] = analysis.get('performance_metrics', {})
            
            fig_comparison = chart_generator.create_performance_comparison(
                comparison_results,
                title="Strategy Performance Comparison",
                save_path="./reports/strategy_comparison.png"
            )
            print("✓ Strategy comparison chart saved")
            
        except Exception as e:
            print(f"⚠ Error creating comparison chart: {e}")
    
    # Final Summary
    print("\n" + "="*60)
    print("BACKTESTING COMPLETE - SUMMARY")
    print("="*60)
    
    print(f"Tested {len(strategies)} strategies on {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Data points: {len(processed_data)}")
    
    if analysis_results:
        print("\nStrategy Rankings (by Sharpe Ratio):")
        rankings = []
        for name, analysis in analysis_results.items():
            sharpe = analysis.get('performance_metrics', {}).get('Sharpe Ratio', 0)
            total_return = analysis.get('performance_metrics', {}).get('Total Return (%)', 0)
            rankings.append((name, sharpe, total_return))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, sharpe, ret) in enumerate(rankings, 1):
            print(f"  {i}. {name}: Sharpe={sharpe:.3f}, Return={ret:.2f}%")
    
    print(f"\nReports and charts saved to: ./reports/")
    print("Backtesting framework demonstration complete!")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("reports", exist_ok=True)
    os.makedirs(".cache", exist_ok=True)
    
    main()