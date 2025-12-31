"""
Risk analysis module for comprehensive risk assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import scipy.stats as stats

class RiskAnalyzer:
    """
    Comprehensive risk analyzer for trading strategies.
    """
    
    def __init__(self):
        """Initialize risk analyzer."""
        pass
    
    def analyze_risk(self, 
                    equity_curve: pd.Series,
                    trades: List,
                    benchmark: Optional[pd.Series] = None) -> Dict:
        """
        Perform comprehensive risk analysis.
        
        Args:
            equity_curve: Portfolio value over time
            trades: List of trade objects
            benchmark: Benchmark for comparison
            
        Returns:
            Dictionary with risk analysis results
        """
        
        if len(equity_curve) == 0:
            return {}
        
        returns = equity_curve.pct_change().dropna()
        
        # Volatility analysis
        volatility_analysis = self._analyze_volatility(returns)
        
        # Drawdown analysis
        drawdown_analysis = self._analyze_drawdowns(equity_curve)
        
        # Value at Risk analysis
        var_analysis = self._analyze_var(returns)
        
        # Tail risk analysis
        tail_risk_analysis = self._analyze_tail_risk(returns, trades)
        
        # Correlation analysis
        correlation_analysis = self._analyze_correlations(returns, benchmark)
        
        return {
            'volatility_analysis': volatility_analysis,
            'drawdown_analysis': drawdown_analysis,
            'var_analysis': var_analysis,
            'tail_risk_analysis': tail_risk_analysis,
            'correlation_analysis': correlation_analysis
        }
    
    def _analyze_volatility(self, returns: pd.Series) -> Dict:
        """Analyze volatility patterns."""
        
        # Basic volatility metrics
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Rolling volatility
        rolling_vol_30 = returns.rolling(30).std() * np.sqrt(252)
        rolling_vol_90 = returns.rolling(90).std() * np.sqrt(252)
        
        # Volatility clustering (GARCH-like analysis)
        squared_returns = returns ** 2
        vol_autocorr = squared_returns.autocorr(lag=1)
        
        # Volatility regime changes
        vol_changes = rolling_vol_30.pct_change().abs()
        high_vol_changes = (vol_changes > vol_changes.quantile(0.95)).sum()
        
        return {
            'daily_volatility': round(daily_vol * 100, 3),
            'annual_volatility': round(annual_vol * 100, 2),
            'volatility_clustering': round(vol_autocorr, 3),
            'high_volatility_events': high_vol_changes,
            'max_30d_volatility': round(rolling_vol_30.max() * 100, 2),
            'min_30d_volatility': round(rolling_vol_30.min() * 100, 2),
            'volatility_stability': round(rolling_vol_30.std() * 100, 2)
        }
    
    def _analyze_drawdowns(self, equity_curve: pd.Series) -> Dict:
        """Comprehensive drawdown analysis."""
        
        # Calculate drawdowns
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        
        # Drawdown statistics
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
        
        # Drawdown duration analysis
        drawdown_periods = self._get_drawdown_periods(drawdown)
        
        if drawdown_periods:
            max_duration = max(dd['duration'] for dd in drawdown_periods)
            avg_duration = np.mean([dd['duration'] for dd in drawdown_periods])
            recovery_times = [dd['recovery_time'] for dd in drawdown_periods if dd['recovery_time'] is not None]
            avg_recovery = np.mean(recovery_times) if recovery_times else 0
        else:
            max_duration = avg_duration = avg_recovery = 0
        
        # Underwater curve analysis
        underwater_pct = (drawdown < -0.05).sum() / len(drawdown) * 100  # % time >5% underwater
        
        return {
            'max_drawdown_pct': round(max_drawdown * 100, 2),
            'avg_drawdown_pct': round(avg_drawdown * 100, 2),
            'num_drawdown_periods': len(drawdown_periods),
            'max_drawdown_duration_days': max_duration,
            'avg_drawdown_duration_days': round(avg_duration, 1),
            'avg_recovery_time_days': round(avg_recovery, 1),
            'time_underwater_pct': round(underwater_pct, 1),
            'drawdown_periods': drawdown_periods[:5]  # Top 5 worst drawdowns
        }
    
    def _analyze_var(self, returns: pd.Series) -> Dict:
        """Value at Risk analysis."""
        
        # Historical VaR
        var_95_hist = np.percentile(returns, 5)
        var_99_hist = np.percentile(returns, 1)
        
        # Parametric VaR (assuming normal distribution)
        mean_return = returns.mean()
        std_return = returns.std()
        var_95_param = mean_return - 1.645 * std_return
        var_99_param = mean_return - 2.326 * std_return
        
        # Expected Shortfall (Conditional VaR)
        es_95 = returns[returns <= var_95_hist].mean()
        es_99 = returns[returns <= var_99_hist].mean()
        
        # Modified VaR (Cornish-Fisher expansion)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Cornish-Fisher quantiles
        z_95 = 1.645
        z_99 = 2.326
        
        cf_95 = z_95 + (z_95**2 - 1) * skewness / 6 + (z_95**3 - 3*z_95) * kurtosis / 24
        cf_99 = z_99 + (z_99**2 - 1) * skewness / 6 + (z_99**3 - 3*z_99) * kurtosis / 24
        
        var_95_modified = mean_return - cf_95 * std_return
        var_99_modified = mean_return - cf_99 * std_return
        
        return {
            'var_95_historical_pct': round(var_95_hist * 100, 2),
            'var_99_historical_pct': round(var_99_hist * 100, 2),
            'var_95_parametric_pct': round(var_95_param * 100, 2),
            'var_99_parametric_pct': round(var_99_param * 100, 2),
            'var_95_modified_pct': round(var_95_modified * 100, 2),
            'var_99_modified_pct': round(var_99_modified * 100, 2),
            'expected_shortfall_95_pct': round(es_95 * 100, 2),
            'expected_shortfall_99_pct': round(es_99 * 100, 2)
        }
    
    def _analyze_tail_risk(self, returns: pd.Series, trades: List) -> Dict:
        """Analyze tail risk characteristics."""
        
        # Distribution characteristics
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Tail ratios
        left_tail = returns.quantile(0.05)
        right_tail = returns.quantile(0.95)
        tail_ratio = abs(right_tail / left_tail) if left_tail != 0 else 0
        
        # Extreme returns
        extreme_positive = (returns > returns.quantile(0.99)).sum()
        extreme_negative = (returns < returns.quantile(0.01)).sum()
        
        # Black swan events (>3 sigma moves)
        sigma = returns.std()
        black_swan_positive = (returns > 3 * sigma).sum()
        black_swan_negative = (returns < -3 * sigma).sum()
        
        # Trade-based tail risk
        if trades:
            trade_returns = [t.pnl_pct for t in trades]
            worst_trade = min(trade_returns) * 100
            best_trade = max(trade_returns) * 100
            
            # Consecutive loss analysis
            consecutive_losses = self._analyze_consecutive_losses(trades)
        else:
            worst_trade = best_trade = 0
            consecutive_losses = {}
        
        return {
            'skewness': round(skewness, 3),
            'kurtosis': round(kurtosis, 3),
            'tail_ratio': round(tail_ratio, 2),
            'extreme_positive_days': extreme_positive,
            'extreme_negative_days': extreme_negative,
            'black_swan_positive': black_swan_positive,
            'black_swan_negative': black_swan_negative,
            'worst_trade_pct': round(worst_trade, 2),
            'best_trade_pct': round(best_trade, 2),
            'consecutive_losses': consecutive_losses
        }
    
    def _analyze_correlations(self, 
                            returns: pd.Series, 
                            benchmark: Optional[pd.Series]) -> Dict:
        """Analyze correlation patterns."""
        
        if benchmark is None:
            return {}
        
        benchmark_returns = benchmark.pct_change().dropna()
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) == 0:
            return {}
        
        # Overall correlation
        correlation = aligned_returns.corr(aligned_benchmark)
        
        # Rolling correlation
        rolling_corr_30 = aligned_returns.rolling(30).corr(aligned_benchmark)
        rolling_corr_90 = aligned_returns.rolling(90).corr(aligned_benchmark)
        
        # Correlation in different market conditions
        up_market = aligned_benchmark > 0
        down_market = aligned_benchmark < 0
        
        corr_up = aligned_returns[up_market].corr(aligned_benchmark[up_market])
        corr_down = aligned_returns[down_market].corr(aligned_benchmark[down_market])
        
        # Correlation stability
        corr_stability = rolling_corr_30.std()
        
        return {
            'overall_correlation': round(correlation, 3),
            'correlation_up_market': round(corr_up, 3),
            'correlation_down_market': round(corr_down, 3),
            'correlation_stability': round(corr_stability, 3),
            'max_30d_correlation': round(rolling_corr_30.max(), 3),
            'min_30d_correlation': round(rolling_corr_30.min(), 3)
        }
    
    def _get_drawdown_periods(self, drawdown: pd.Series) -> List[Dict]:
        """Extract drawdown periods with details."""
        
        periods = []
        in_drawdown = False
        start_date = None
        peak_date = None
        
        for date, dd in drawdown.items():
            if dd < -0.01 and not in_drawdown:  # Start of drawdown (>1%)
                in_drawdown = True
                start_date = date
                peak_date = date
                min_dd = dd
            elif in_drawdown:
                if dd < min_dd:
                    min_dd = dd
                    peak_date = date
                
                if dd >= 0:  # Recovery
                    in_drawdown = False
                    duration = (peak_date - start_date).days
                    recovery_time = (date - peak_date).days
                    
                    periods.append({
                        'start_date': start_date,
                        'peak_date': peak_date,
                        'end_date': date,
                        'duration': duration,
                        'recovery_time': recovery_time,
                        'max_drawdown': min_dd * 100
                    })
        
        # Sort by severity
        periods.sort(key=lambda x: x['max_drawdown'])
        
        return periods
    
    def _analyze_consecutive_losses(self, trades: List) -> Dict:
        """Analyze consecutive loss patterns."""
        
        if not trades:
            return {}
        
        loss_streaks = []
        current_streak = 0
        
        for trade in trades:
            if trade.pnl < 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    loss_streaks.append(current_streak)
                current_streak = 0
        
        # Don't forget the last streak
        if current_streak > 0:
            loss_streaks.append(current_streak)
        
        if loss_streaks:
            max_consecutive = max(loss_streaks)
            avg_consecutive = np.mean(loss_streaks)
            
            # Probability of consecutive losses
            total_losses = len([t for t in trades if t.pnl < 0])
            loss_probability = total_losses / len(trades)
            
            # Expected consecutive losses
            prob_3_losses = loss_probability ** 3
            prob_5_losses = loss_probability ** 5
        else:
            max_consecutive = avg_consecutive = 0
            prob_3_losses = prob_5_losses = 0
        
        return {
            'max_consecutive_losses': max_consecutive,
            'avg_consecutive_losses': round(avg_consecutive, 1),
            'probability_3_consecutive_pct': round(prob_3_losses * 100, 2),
            'probability_5_consecutive_pct': round(prob_5_losses * 100, 2)
        }