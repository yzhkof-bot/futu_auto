"""
Performance optimization utilities for the backtesting framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
import multiprocessing as mp
from functools import partial
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

class PerformanceOptimizer:
    """
    Performance optimization utilities for faster backtesting.
    """
    
    def __init__(self, n_jobs: int = -1):
        """
        Initialize performance optimizer.
        
        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
    
    def parallel_backtest(self, 
                         backtest_func: Callable,
                         parameter_sets: List[Dict],
                         data: pd.DataFrame,
                         progress_callback: Optional[Callable] = None) -> List[Dict]:
        """
        Run multiple backtests in parallel.
        
        Args:
            backtest_func: Function to run backtest
            parameter_sets: List of parameter dictionaries
            data: Market data
            progress_callback: Optional progress callback function
            
        Returns:
            List of backtest results
        """
        
        print(f"Running {len(parameter_sets)} backtests using {self.n_jobs} cores...")
        
        # Create partial function with fixed data
        partial_func = partial(self._run_single_backtest, backtest_func, data)
        
        results = []
        completed = 0
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all jobs
            future_to_params = {
                executor.submit(partial_func, params): params 
                for params in parameter_sets
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_params):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, len(parameter_sets))
                    elif completed % 10 == 0:
                        print(f"Completed: {completed}/{len(parameter_sets)}")
                        
                except Exception as e:
                    print(f"Error in backtest: {e}")
                    continue
        
        return results
    
    def _run_single_backtest(self, backtest_func: Callable, data: pd.DataFrame, params: Dict) -> Dict:
        """Run a single backtest with error handling."""
        
        try:
            result = backtest_func(data, params)
            result['parameters'] = params
            return result
        except Exception as e:
            return {
                'parameters': params,
                'error': str(e),
                'total_return_pct': 0,
                'sharpe_ratio': 0,
                'max_drawdown_pct': -100
            }
    
    def optimize_data_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize data processing for faster backtesting.
        
        Args:
            data: Raw market data
            
        Returns:
            Optimized data
        """
        
        # Convert to more efficient data types
        optimized_data = data.copy()
        
        # Convert float64 to float32 where possible (saves memory)
        float_cols = optimized_data.select_dtypes(include=['float64']).columns
        for col in float_cols:
            if optimized_data[col].max() < np.finfo(np.float32).max:
                optimized_data[col] = optimized_data[col].astype(np.float32)
        
        # Ensure datetime index
        if not isinstance(optimized_data.index, pd.DatetimeIndex):
            optimized_data.index = pd.to_datetime(optimized_data.index)
        
        # Sort by date for faster access
        optimized_data = optimized_data.sort_index()
        
        # Pre-calculate commonly used indicators to avoid repeated calculation
        if 'Returns' not in optimized_data.columns:
            optimized_data['Returns'] = optimized_data['Close'].pct_change()
        
        return optimized_data
    
    def vectorized_signal_generation(self, 
                                   data: pd.DataFrame,
                                   short_ma: int,
                                   long_ma: int,
                                   rsi_period: int = 14) -> pd.DataFrame:
        """
        Vectorized signal generation for faster processing.
        
        Args:
            data: Market data
            short_ma: Short moving average period
            long_ma: Long moving average period
            rsi_period: RSI period
            
        Returns:
            DataFrame with signals
        """
        
        signals = pd.DataFrame(index=data.index)
        
        # Vectorized moving averages
        signals['SMA_Short'] = data['Close'].rolling(short_ma).mean()
        signals['SMA_Long'] = data['Close'].rolling(long_ma).mean()
        
        # Vectorized RSI calculation
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        rs = gain / loss
        signals['RSI'] = 100 - (100 / (1 + rs))
        
        # Vectorized signal generation
        signals['MA_Signal'] = np.where(
            signals['SMA_Short'] > signals['SMA_Long'], 1, -1
        )
        
        signals['RSI_Signal'] = np.where(
            (signals['RSI'] > 30) & (signals['RSI'] < 70), 1, 0
        )
        
        # Combined signal
        signals['Signal'] = np.where(
            (signals['MA_Signal'] == 1) & (signals['RSI_Signal'] == 1), 1,
            np.where(signals['MA_Signal'] == -1, -1, 0)
        )
        
        return signals
    
    def fast_backtest_engine(self, 
                           data: pd.DataFrame,
                           signals: pd.DataFrame,
                           initial_capital: float = 100000,
                           commission: float = 0.001) -> Dict:
        """
        Fast vectorized backtest engine.
        
        Args:
            data: Market data
            signals: Trading signals
            initial_capital: Starting capital
            commission: Commission rate
            
        Returns:
            Backtest results
        """
        
        # Align data and signals
        aligned_data = data.reindex(signals.index).fillna(method='ffill')
        
        # Vectorized position calculation
        positions = signals['Signal'].shift(1).fillna(0)  # Lag signals by 1 day
        
        # Calculate returns
        price_returns = aligned_data['Close'].pct_change()
        strategy_returns = positions * price_returns
        
        # Apply transaction costs
        position_changes = positions.diff().abs()
        transaction_costs = position_changes * commission
        net_returns = strategy_returns - transaction_costs
        
        # Calculate equity curve
        equity_curve = (1 + net_returns).cumprod() * initial_capital
        
        # Performance metrics
        total_return = (equity_curve.iloc[-1] / initial_capital - 1) * 100
        
        if len(net_returns.dropna()) > 0:
            sharpe_ratio = net_returns.mean() / net_returns.std() * np.sqrt(252)
            
            # Drawdown calculation
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak
            max_drawdown = drawdown.min() * 100
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Trade statistics
        trades = position_changes[position_changes > 0]
        num_trades = len(trades)
        
        if num_trades > 0:
            trade_returns = []
            current_pos = 0
            entry_price = 0
            
            for i, pos in enumerate(positions):
                if pos != current_pos:  # Position change
                    if current_pos != 0:  # Closing position
                        exit_price = aligned_data['Close'].iloc[i]
                        trade_return = (exit_price - entry_price) / entry_price * current_pos
                        trade_returns.append(trade_return)
                    
                    if pos != 0:  # Opening new position
                        entry_price = aligned_data['Close'].iloc[i]
                    
                    current_pos = pos
            
            if trade_returns:
                win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns) * 100
            else:
                win_rate = 0
        else:
            win_rate = 0
        
        return {
            'equity_curve': equity_curve,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'num_trades': num_trades,
            'win_rate_pct': win_rate,
            'final_value': equity_curve.iloc[-1]
        }
    
    def memory_efficient_processing(self, 
                                  data: pd.DataFrame,
                                  chunk_size: int = 10000) -> pd.DataFrame:
        """
        Process large datasets in chunks to save memory.
        
        Args:
            data: Large dataset
            chunk_size: Size of each chunk
            
        Returns:
            Processed data
        """
        
        if len(data) <= chunk_size:
            return data
        
        processed_chunks = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i+chunk_size].copy()
            
            # Process chunk (add indicators, etc.)
            chunk = self._process_chunk(chunk)
            
            processed_chunks.append(chunk)
        
        # Combine chunks
        result = pd.concat(processed_chunks, axis=0)
        
        # Handle overlapping indicators that need full history
        result = self._fix_overlapping_indicators(result, data)
        
        return result
    
    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk of data."""
        
        # Add basic indicators that don't require full history
        chunk['Returns'] = chunk['Close'].pct_change()
        chunk['Volume_MA'] = chunk['Volume'].rolling(20, min_periods=1).mean()
        
        return chunk
    
    def _fix_overlapping_indicators(self, 
                                  processed_data: pd.DataFrame,
                                  original_data: pd.DataFrame) -> pd.DataFrame:
        """Fix indicators that need full history."""
        
        # Recalculate indicators that need full history
        processed_data['SMA_50'] = original_data['Close'].rolling(50).mean()
        processed_data['SMA_200'] = original_data['Close'].rolling(200).mean()
        
        return processed_data
    
    def benchmark_performance(self, 
                            backtest_func: Callable,
                            data: pd.DataFrame,
                            params: Dict,
                            iterations: int = 5) -> Dict:
        """
        Benchmark backtest performance.
        
        Args:
            backtest_func: Backtest function to benchmark
            data: Market data
            params: Strategy parameters
            iterations: Number of iterations to run
            
        Returns:
            Performance statistics
        """
        
        print(f"Benchmarking performance over {iterations} iterations...")
        
        times = []
        
        for i in range(iterations):
            start_time = time.time()
            
            try:
                result = backtest_func(data, params)
                end_time = time.time()
                times.append(end_time - start_time)
                
                if i == 0:
                    sample_result = result
                    
            except Exception as e:
                print(f"Error in iteration {i}: {e}")
                continue
        
        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            print(f"Average time: {avg_time:.3f}s")
            print(f"Std deviation: {std_time:.3f}s")
            print(f"Min time: {min_time:.3f}s")
            print(f"Max time: {max_time:.3f}s")
            
            return {
                'avg_time': avg_time,
                'std_time': std_time,
                'min_time': min_time,
                'max_time': max_time,
                'sample_result': sample_result if 'sample_result' in locals() else None
            }
        else:
            return {'error': 'No successful iterations'}
    
    def get_optimization_suggestions(self, data_size: int, num_parameters: int) -> Dict:
        """
        Get optimization suggestions based on data size and parameter count.
        
        Args:
            data_size: Number of data points
            num_parameters: Number of parameters to optimize
            
        Returns:
            Optimization suggestions
        """
        
        suggestions = {
            'use_parallel': num_parameters > 10,
            'chunk_processing': data_size > 50000,
            'vectorized_signals': True,
            'memory_optimization': data_size > 100000
        }
        
        # Estimate processing time
        base_time = data_size * 0.0001  # Base time per data point
        param_multiplier = num_parameters * 0.1  # Additional time per parameter
        estimated_time = base_time * param_multiplier
        
        suggestions['estimated_time_minutes'] = estimated_time / 60
        
        # Recommend chunk size
        if data_size > 50000:
            suggestions['recommended_chunk_size'] = min(10000, data_size // 10)
        
        # Recommend number of cores
        if num_parameters > 20:
            suggestions['recommended_cores'] = min(self.n_jobs, num_parameters // 4)
        
        return suggestions