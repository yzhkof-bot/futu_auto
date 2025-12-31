"""
Performance metrics calculation module.
Comprehensive financial metrics for strategy evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import scipy.stats as stats
from datetime import datetime

class PerformanceMetrics:
    """
    Comprehensive performance metrics calculator for trading strategies.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_all_metrics(self, 
                            equity_curve: pd.Series,
                            benchmark: Optional[pd.Series] = None,
                            trades: Optional[List] = None) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            equity_curve: Portfolio value over time
            benchmark: Benchmark returns for comparison
            trades: List of trade objects
            
        Returns:
            Dictionary with all performance metrics
        """
        
        if len(equity_curve) == 0:
            return {}
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Basic metrics
        basic_metrics = self._calculate_basic_metrics(equity_curve, returns)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(returns, equity_curve)
        
        # Trade-based metrics
        trade_metrics = self._calculate_trade_metrics(trades) if trades else {}
        
        # Benchmark comparison
        benchmark_metrics = self._calculate_benchmark_metrics(
            returns, benchmark
        ) if benchmark is not None else {}
        
        # Advanced metrics
        advanced_metrics = self._calculate_advanced_metrics(returns, equity_curve)
        
        # Combine all metrics
        all_metrics = {
            **basic_metrics,
            **risk_metrics,
            **trade_metrics,
            **benchmark_metrics,
            **advanced_metrics
        }
        
        return all_metrics
    
    def _calculate_basic_metrics(self, 
                               equity_curve: pd.Series, 
                               returns: pd.Series) -> Dict:
        """Calculate basic performance metrics."""
        
        initial_value = equity_curve.iloc[0]
        final_value = equity_curve.iloc[-1]
        
        # Total return
        total_return = (final_value / initial_value - 1) * 100
        
        # Annualized return
        years = len(returns) / 252  # Assuming daily data
        if years > 0:
            annualized_return = (final_value / initial_value) ** (1 / years) - 1
        else:
            annualized_return = 0
        
        # CAGR (Compound Annual Growth Rate)
        cagr = annualized_return * 100
        
        return {
            'Total Return (%)': round(total_return, 2),
            'Annualized Return (%)': round(cagr, 2),
            'Initial Value': round(initial_value, 2),
            'Final Value': round(final_value, 2),
            'Trading Days': len(returns),
            'Years': round(years, 2)
        }
    
    def _calculate_risk_metrics(self, 
                              returns: pd.Series, 
                              equity_curve: pd.Series) -> Dict:
        """Calculate risk-related metrics."""
        
        # Volatility
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Sharpe ratio
        excess_returns = returns - self.risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / daily_vol * np.sqrt(252) if daily_vol > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (returns.mean() * 252 - self.risk_free_rate) / downside_deviation
        else:
            sortino_ratio = float('inf')
        
        # Drawdown metrics
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        drawdown_periods = self._calculate_drawdown_periods(drawdown)
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        
        # Calmar ratio
        calmar_ratio = (returns.mean() * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        
        # Expected Shortfall (Conditional VaR)
        es_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        es_99 = returns[returns <= np.percentile(returns, 1)].mean() * 100
        
        return {
            'Volatility (%)': round(annual_vol * 100, 2),
            'Sharpe Ratio': round(sharpe_ratio, 3),
            'Sortino Ratio': round(sortino_ratio, 3),
            'Calmar Ratio': round(calmar_ratio, 3),
            'Max Drawdown (%)': round(max_drawdown * 100, 2),
            'Max Drawdown Duration (days)': max_drawdown_duration,
            'Avg Drawdown Duration (days)': round(avg_drawdown_duration, 1),
            'VaR 95% (%)': round(var_95, 2),
            'VaR 99% (%)': round(var_99, 2),
            'Expected Shortfall 95% (%)': round(es_95, 2),
            'Expected Shortfall 99% (%)': round(es_99, 2)
        }
    
    def _calculate_trade_metrics(self, trades: List) -> Dict:
        """Calculate trade-based metrics."""
        
        if not trades:
            return {}
        
        # Basic trade statistics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / total_trades * 100
        
        # P&L statistics
        total_pnl = sum(t.pnl for t in trades)
        avg_trade_pnl = total_pnl / total_trades
        
        if winning_trades:
            avg_win = np.mean([t.pnl for t in winning_trades])
            avg_win_pct = np.mean([t.pnl_pct for t in winning_trades]) * 100
            max_win = max(t.pnl for t in winning_trades)
            max_win_pct = max(t.pnl_pct for t in winning_trades) * 100
        else:
            avg_win = avg_win_pct = max_win = max_win_pct = 0
        
        if losing_trades:
            avg_loss = np.mean([t.pnl for t in losing_trades])
            avg_loss_pct = np.mean([t.pnl_pct for t in losing_trades]) * 100
            max_loss = min(t.pnl for t in losing_trades)
            max_loss_pct = min(t.pnl_pct for t in losing_trades) * 100
        else:
            avg_loss = avg_loss_pct = max_loss = max_loss_pct = 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        expectancy = (win_rate / 100 * avg_win) + ((100 - win_rate) / 100 * avg_loss)
        
        # Holding period statistics
        holding_periods = [t.holding_period for t in trades]
        avg_holding_period = np.mean(holding_periods)
        median_holding_period = np.median(holding_periods)
        
        # Consecutive wins/losses
        consecutive_wins = self._calculate_consecutive_runs(trades, 'wins')
        consecutive_losses = self._calculate_consecutive_runs(trades, 'losses')
        
        return {
            'Total Trades': total_trades,
            'Winning Trades': len(winning_trades),
            'Losing Trades': len(losing_trades),
            'Win Rate (%)': round(win_rate, 2),
            'Profit Factor': round(profit_factor, 2),
            'Expectancy': round(expectancy, 2),
            'Avg Trade P&L': round(avg_trade_pnl, 2),
            'Avg Win': round(avg_win, 2),
            'Avg Win (%)': round(avg_win_pct, 2),
            'Avg Loss': round(avg_loss, 2),
            'Avg Loss (%)': round(avg_loss_pct, 2),
            'Max Win': round(max_win, 2),
            'Max Win (%)': round(max_win_pct, 2),
            'Max Loss': round(max_loss, 2),
            'Max Loss (%)': round(max_loss_pct, 2),
            'Avg Holding Period (days)': round(avg_holding_period, 1),
            'Median Holding Period (days)': median_holding_period,
            'Max Consecutive Wins': consecutive_wins,
            'Max Consecutive Losses': consecutive_losses
        }
    
    def _calculate_benchmark_metrics(self, 
                                   returns: pd.Series, 
                                   benchmark: pd.Series) -> Dict:
        """Calculate benchmark comparison metrics."""
        
        # Align returns and benchmark
        aligned_returns, aligned_benchmark = returns.align(benchmark, join='inner')
        
        if len(aligned_returns) == 0:
            return {}
        
        # Beta calculation
        covariance = np.cov(aligned_returns, aligned_benchmark)[0][1]
        benchmark_variance = np.var(aligned_benchmark)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha calculation (Jensen's alpha)
        alpha = aligned_returns.mean() - (self.risk_free_rate / 252 + beta * (aligned_benchmark.mean() - self.risk_free_rate / 252))
        alpha_annualized = alpha * 252
        
        # Correlation
        correlation = aligned_returns.corr(aligned_benchmark)
        
        # Information ratio
        excess_returns = aligned_returns - aligned_benchmark
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        # Up/Down capture ratios
        up_periods = aligned_benchmark > 0
        down_periods = aligned_benchmark < 0
        
        if up_periods.sum() > 0:
            up_capture = (aligned_returns[up_periods].mean() / aligned_benchmark[up_periods].mean()) * 100
        else:
            up_capture = 0
        
        if down_periods.sum() > 0:
            down_capture = (aligned_returns[down_periods].mean() / aligned_benchmark[down_periods].mean()) * 100
        else:
            down_capture = 0
        
        return {
            'Beta': round(beta, 3),
            'Alpha (annualized)': round(alpha_annualized * 100, 2),
            'Correlation': round(correlation, 3),
            'Information Ratio': round(information_ratio, 3),
            'Tracking Error (%)': round(tracking_error * 100, 2),
            'Up Capture (%)': round(up_capture, 2),
            'Down Capture (%)': round(down_capture, 2)
        }
    
    def _calculate_advanced_metrics(self, 
                                  returns: pd.Series, 
                                  equity_curve: pd.Series) -> Dict:
        """Calculate advanced performance metrics."""
        
        # Skewness and Kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Tail ratio
        tail_ratio = abs(np.percentile(returns, 95)) / abs(np.percentile(returns, 5))
        
        # Recovery factor
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = abs(drawdown.min())
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # Ulcer Index
        ulcer_index = np.sqrt(np.mean(drawdown ** 2)) * 100
        
        # Pain Index
        pain_index = np.mean(abs(drawdown)) * 100
        
        # Sterling Ratio
        sterling_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            'Skewness': round(skewness, 3),
            'Kurtosis': round(kurtosis, 3),
            'Tail Ratio': round(tail_ratio, 3),
            'Recovery Factor': round(recovery_factor, 2),
            'Ulcer Index': round(ulcer_index, 2),
            'Pain Index': round(pain_index, 2),
            'Sterling Ratio': round(sterling_ratio, 2)
        }
    
    def _calculate_drawdown_periods(self, drawdown: pd.Series) -> List[int]:
        """Calculate drawdown periods in days."""
        
        periods = []
        in_drawdown = False
        start_date = None
        
        for date, dd in drawdown.items():
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_date = date
            elif dd >= 0 and in_drawdown:
                # End of drawdown
                in_drawdown = False
                if start_date is not None:
                    period_length = (date - start_date).days
                    periods.append(period_length)
        
        # Handle case where drawdown continues to end
        if in_drawdown and start_date is not None:
            period_length = (drawdown.index[-1] - start_date).days
            periods.append(period_length)
        
        return periods
    
    def _calculate_consecutive_runs(self, trades: List, run_type: str) -> int:
        """Calculate maximum consecutive wins or losses."""
        
        if not trades:
            return 0
        
        max_run = 0
        current_run = 0
        
        for trade in trades:
            if run_type == 'wins' and trade.pnl > 0:
                current_run += 1
                max_run = max(max_run, current_run)
            elif run_type == 'losses' and trade.pnl < 0:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        return max_run
    
    def calculate_rolling_metrics(self, 
                                returns: pd.Series, 
                                window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            returns: Return series
            window: Rolling window size in days
            
        Returns:
            DataFrame with rolling metrics
        """
        
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        # Rolling Sharpe ratio
        rolling_excess = returns - self.risk_free_rate / 252
        rolling_metrics['Sharpe_Ratio'] = (
            rolling_excess.rolling(window).mean() / 
            returns.rolling(window).std() * np.sqrt(252)
        )
        
        # Rolling volatility
        rolling_metrics['Volatility'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Rolling maximum drawdown
        equity_curve = (1 + returns).cumprod()
        rolling_peak = equity_curve.rolling(window, min_periods=1).max()
        rolling_drawdown = (equity_curve - rolling_peak) / rolling_peak
        rolling_metrics['Max_Drawdown'] = rolling_drawdown.rolling(window).min()
        
        return rolling_metrics