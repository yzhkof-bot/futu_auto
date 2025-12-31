"""
Comprehensive strategy analyzer with advanced analytics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from ..backtest.metrics import PerformanceMetrics

class StrategyAnalyzer:
    """
    Advanced strategy analyzer for comprehensive performance evaluation.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.metrics_calculator = PerformanceMetrics()
    
    def analyze_strategy(self, 
                        backtest_results: Dict,
                        benchmark_data: Optional[pd.Series] = None) -> Dict:
        """
        Perform comprehensive strategy analysis.
        
        Args:
            backtest_results: Results from backtesting engine
            benchmark_data: Benchmark price data for comparison
            
        Returns:
            Dictionary with comprehensive analysis
        """
        
        equity_curve = backtest_results.get('equity_curve', pd.Series())
        trades = backtest_results.get('trades', [])
        
        if len(equity_curve) == 0:
            return {'error': 'No equity curve data available'}
        
        # Performance metrics
        performance_metrics = self.metrics_calculator.calculate_all_metrics(
            equity_curve, benchmark_data, trades
        )
        
        # Time-based analysis
        time_analysis = self._analyze_time_patterns(equity_curve, trades)
        
        # Risk analysis
        risk_analysis = self._analyze_risk_patterns(equity_curve, trades)
        
        # Trade analysis
        trade_analysis = self._analyze_trade_patterns(trades)
        
        # Market regime analysis
        regime_analysis = self._analyze_market_regimes(equity_curve, benchmark_data)
        
        return {
            'performance_metrics': performance_metrics,
            'time_analysis': time_analysis,
            'risk_analysis': risk_analysis,
            'trade_analysis': trade_analysis,
            'regime_analysis': regime_analysis,
            'summary': self._create_summary(performance_metrics, trade_analysis)
        }
    
    def _analyze_time_patterns(self, 
                             equity_curve: pd.Series, 
                             trades: List) -> Dict:
        """Analyze time-based performance patterns."""
        
        if len(equity_curve) == 0:
            return {}
        
        returns = equity_curve.pct_change().dropna()
        
        # Monthly performance
        monthly_returns = equity_curve.resample('M').last().pct_change().dropna()
        
        # Yearly performance
        yearly_returns = equity_curve.resample('Y').last().pct_change().dropna()
        
        # Day of week analysis
        if len(returns) > 0:
            returns_by_dow = returns.groupby(returns.index.dayofweek).agg(['mean', 'std', 'count']) * 100
            
            dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            returns_by_dow.index = [dow_names[i] for i in returns_by_dow.index if i < len(dow_names)]
        else:
            returns_by_dow = pd.DataFrame()
        
        # Month of year analysis
        if len(monthly_returns) > 0:
            returns_by_month = monthly_returns.groupby(monthly_returns.index.month).agg(['mean', 'std', 'count']) * 100
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            returns_by_month.index = [month_names[i-1] for i in returns_by_month.index if 1 <= i <= 12]
        else:
            returns_by_month = pd.DataFrame()
        
        # Best and worst periods
        best_month = monthly_returns.max() * 100 if len(monthly_returns) > 0 else 0
        worst_month = monthly_returns.min() * 100 if len(monthly_returns) > 0 else 0
        best_year = yearly_returns.max() * 100 if len(yearly_returns) > 0 else 0
        worst_year = yearly_returns.min() * 100 if len(yearly_returns) > 0 else 0
        
        return {
            'monthly_returns': monthly_returns * 100,
            'yearly_returns': yearly_returns * 100,
            'returns_by_day_of_week': returns_by_dow,
            'returns_by_month': returns_by_month,
            'best_month_pct': round(best_month, 2),
            'worst_month_pct': round(worst_month, 2),
            'best_year_pct': round(best_year, 2),
            'worst_year_pct': round(worst_year, 2),
            'positive_months': len(monthly_returns[monthly_returns > 0]) if len(monthly_returns) > 0 else 0,
            'negative_months': len(monthly_returns[monthly_returns < 0]) if len(monthly_returns) > 0 else 0
        }
    
    def _analyze_risk_patterns(self, 
                             equity_curve: pd.Series, 
                             trades: List) -> Dict:
        """Analyze risk patterns and characteristics."""
        
        if len(equity_curve) == 0:
            return {}
        
        returns = equity_curve.pct_change().dropna()
        
        # Drawdown analysis
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        
        # Identify drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        start_value = None
        
        for date, dd in drawdown.items():
            if dd < -0.01 and not in_drawdown:  # Start of significant drawdown (>1%)
                in_drawdown = True
                start_date = date
                start_value = equity_curve[date]
            elif dd >= 0 and in_drawdown:  # Recovery
                in_drawdown = False
                if start_date is not None:
                    recovery_date = date
                    recovery_value = equity_curve[date]
                    max_dd = drawdown[start_date:recovery_date].min()
                    
                    drawdown_periods.append({
                        'start_date': start_date,
                        'recovery_date': recovery_date,
                        'duration_days': (recovery_date - start_date).days,
                        'max_drawdown_pct': max_dd * 100,
                        'start_value': start_value,
                        'recovery_value': recovery_value
                    })
        
        # Risk concentration analysis
        if trades:
            trade_returns = [t.pnl_pct for t in trades]
            
            # Tail risk analysis
            var_95 = np.percentile(trade_returns, 5) * 100
            var_99 = np.percentile(trade_returns, 1) * 100
            
            # Risk of ruin estimation (simplified)
            negative_trades = [r for r in trade_returns if r < 0]
            if negative_trades:
                avg_loss_pct = np.mean(negative_trades) * 100
                max_loss_pct = min(negative_trades) * 100
                loss_std = np.std(negative_trades) * 100
            else:
                avg_loss_pct = max_loss_pct = loss_std = 0
        else:
            var_95 = var_99 = avg_loss_pct = max_loss_pct = loss_std = 0
        
        return {
            'drawdown_periods': drawdown_periods,
            'num_drawdown_periods': len(drawdown_periods),
            'avg_drawdown_duration': np.mean([dd['duration_days'] for dd in drawdown_periods]) if drawdown_periods else 0,
            'max_drawdown_duration': max([dd['duration_days'] for dd in drawdown_periods]) if drawdown_periods else 0,
            'var_95_pct': round(var_95, 2),
            'var_99_pct': round(var_99, 2),
            'avg_loss_pct': round(avg_loss_pct, 2),
            'max_loss_pct': round(max_loss_pct, 2),
            'loss_volatility_pct': round(loss_std, 2)
        }
    
    def _analyze_trade_patterns(self, trades: List) -> Dict:
        """Analyze trading patterns and characteristics."""
        
        if not trades:
            return {}
        
        # Trade timing analysis
        entry_hours = [t.entry_time.hour for t in trades if hasattr(t.entry_time, 'hour')]
        exit_hours = [t.exit_time.hour for t in trades if hasattr(t.exit_time, 'hour')]
        
        # Trade size analysis
        trade_sizes = [abs(t.quantity * t.entry_price) for t in trades]
        
        # Holding period analysis
        holding_periods = [t.holding_period for t in trades]
        
        # Win/loss streaks
        win_loss_sequence = [1 if t.pnl > 0 else -1 for t in trades]
        
        # Calculate streaks
        max_win_streak = max_loss_streak = current_streak = 0
        current_type = 0
        
        for result in win_loss_sequence:
            if result == current_type:
                current_streak += 1
            else:
                if current_type == 1:
                    max_win_streak = max(max_win_streak, current_streak)
                elif current_type == -1:
                    max_loss_streak = max(max_loss_streak, current_streak)
                current_streak = 1
                current_type = result
        
        # Final streak update
        if current_type == 1:
            max_win_streak = max(max_win_streak, current_streak)
        elif current_type == -1:
            max_loss_streak = max(max_loss_streak, current_streak)
        
        # Trade distribution by P&L
        pnl_ranges = {
            'large_wins': len([t for t in trades if t.pnl_pct > 0.05]),  # >5%
            'small_wins': len([t for t in trades if 0 < t.pnl_pct <= 0.05]),  # 0-5%
            'small_losses': len([t for t in trades if -0.05 <= t.pnl_pct < 0]),  # 0 to -5%
            'large_losses': len([t for t in trades if t.pnl_pct < -0.05])  # <-5%
        }
        
        return {
            'avg_trade_size': np.mean(trade_sizes) if trade_sizes else 0,
            'median_trade_size': np.median(trade_sizes) if trade_sizes else 0,
            'avg_holding_period': np.mean(holding_periods) if holding_periods else 0,
            'median_holding_period': np.median(holding_periods) if holding_periods else 0,
            'max_holding_period': max(holding_periods) if holding_periods else 0,
            'min_holding_period': min(holding_periods) if holding_periods else 0,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'pnl_distribution': pnl_ranges,
            'trades_per_month': len(trades) / (max(holding_periods) / 30) if holding_periods and max(holding_periods) > 0 else 0
        }
    
    def _analyze_market_regimes(self, 
                              equity_curve: pd.Series, 
                              benchmark_data: Optional[pd.Series]) -> Dict:
        """Analyze performance across different market regimes."""
        
        if benchmark_data is None or len(equity_curve) == 0:
            return {}
        
        # Align data
        strategy_returns = equity_curve.pct_change().dropna()
        benchmark_returns = benchmark_data.pct_change().dropna()
        
        aligned_strategy, aligned_benchmark = strategy_returns.align(benchmark_returns, join='inner')
        
        if len(aligned_strategy) == 0:
            return {}
        
        # Define market regimes based on benchmark performance
        # Bull market: benchmark returns > 75th percentile
        # Bear market: benchmark returns < 25th percentile
        # Sideways: between 25th and 75th percentile
        
        bull_threshold = aligned_benchmark.quantile(0.75)
        bear_threshold = aligned_benchmark.quantile(0.25)
        
        bull_periods = aligned_benchmark > bull_threshold
        bear_periods = aligned_benchmark < bear_threshold
        sideways_periods = (aligned_benchmark >= bear_threshold) & (aligned_benchmark <= bull_threshold)
        
        # Calculate performance in each regime
        regimes = {
            'bull_market': {
                'periods': bull_periods.sum(),
                'strategy_return': aligned_strategy[bull_periods].mean() * 252 * 100,
                'benchmark_return': aligned_benchmark[bull_periods].mean() * 252 * 100,
                'strategy_volatility': aligned_strategy[bull_periods].std() * np.sqrt(252) * 100,
                'correlation': aligned_strategy[bull_periods].corr(aligned_benchmark[bull_periods])
            },
            'bear_market': {
                'periods': bear_periods.sum(),
                'strategy_return': aligned_strategy[bear_periods].mean() * 252 * 100,
                'benchmark_return': aligned_benchmark[bear_periods].mean() * 252 * 100,
                'strategy_volatility': aligned_strategy[bear_periods].std() * np.sqrt(252) * 100,
                'correlation': aligned_strategy[bear_periods].corr(aligned_benchmark[bear_periods])
            },
            'sideways_market': {
                'periods': sideways_periods.sum(),
                'strategy_return': aligned_strategy[sideways_periods].mean() * 252 * 100,
                'benchmark_return': aligned_benchmark[sideways_periods].mean() * 252 * 100,
                'strategy_volatility': aligned_strategy[sideways_periods].std() * np.sqrt(252) * 100,
                'correlation': aligned_strategy[sideways_periods].corr(aligned_benchmark[sideways_periods])
            }
        }
        
        # Clean up NaN values
        for regime in regimes.values():
            for key, value in regime.items():
                if pd.isna(value):
                    regime[key] = 0
        
        return regimes
    
    def _create_summary(self, 
                       performance_metrics: Dict, 
                       trade_analysis: Dict) -> Dict:
        """Create executive summary of strategy performance."""
        
        # Extract key metrics
        total_return = performance_metrics.get('Total Return (%)', 0)
        sharpe_ratio = performance_metrics.get('Sharpe Ratio', 0)
        max_drawdown = performance_metrics.get('Max Drawdown (%)', 0)
        win_rate = performance_metrics.get('Win Rate (%)', 0)
        profit_factor = performance_metrics.get('Profit Factor', 0)
        
        # Performance rating
        score = 0
        if total_return > 10: score += 1
        if sharpe_ratio > 1.0: score += 1
        if max_drawdown > -15: score += 1
        if win_rate > 50: score += 1
        if profit_factor > 1.5: score += 1
        
        ratings = ['Poor', 'Below Average', 'Average', 'Good', 'Excellent']
        rating = ratings[min(score, 4)]
        
        # Key strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if sharpe_ratio > 1.5:
            strengths.append("Strong risk-adjusted returns")
        elif sharpe_ratio < 0.5:
            weaknesses.append("Poor risk-adjusted returns")
        
        if max_drawdown > -10:
            strengths.append("Low maximum drawdown")
        elif max_drawdown < -25:
            weaknesses.append("High maximum drawdown")
        
        if win_rate > 60:
            strengths.append("High win rate")
        elif win_rate < 40:
            weaknesses.append("Low win rate")
        
        if profit_factor > 2.0:
            strengths.append("Excellent profit factor")
        elif profit_factor < 1.2:
            weaknesses.append("Poor profit factor")
        
        return {
            'overall_rating': rating,
            'performance_score': f"{score}/5",
            'key_strengths': strengths,
            'key_weaknesses': weaknesses,
            'recommendation': self._get_recommendation(score, performance_metrics)
        }
    
    def _get_recommendation(self, score: int, metrics: Dict) -> str:
        """Get strategy recommendation based on performance."""
        
        if score >= 4:
            return "Excellent strategy. Consider live trading with appropriate position sizing."
        elif score >= 3:
            return "Good strategy. Consider further optimization or paper trading."
        elif score >= 2:
            return "Average strategy. Needs improvement before live trading."
        else:
            return "Poor strategy. Significant improvements needed."
    
    def generate_performance_report(self, analysis_results: Dict) -> str:
        """Generate a formatted performance report."""
        
        performance = analysis_results.get('performance_metrics', {})
        summary = analysis_results.get('summary', {})
        
        report = f"""
STRATEGY PERFORMANCE REPORT
{'='*50}

EXECUTIVE SUMMARY
Overall Rating: {summary.get('overall_rating', 'N/A')}
Performance Score: {summary.get('performance_score', 'N/A')}

KEY METRICS
Total Return: {performance.get('Total Return (%)', 'N/A')}%
Annualized Return: {performance.get('Annualized Return (%)', 'N/A')}%
Sharpe Ratio: {performance.get('Sharpe Ratio', 'N/A')}
Max Drawdown: {performance.get('Max Drawdown (%)', 'N/A')}%
Win Rate: {performance.get('Win Rate (%)', 'N/A')}%
Profit Factor: {performance.get('Profit Factor', 'N/A')}

STRENGTHS
{chr(10).join(['• ' + s for s in summary.get('key_strengths', [])])}

WEAKNESSES
{chr(10).join(['• ' + w for w in summary.get('key_weaknesses', [])])}

RECOMMENDATION
{summary.get('recommendation', 'N/A')}
"""
        
        return report