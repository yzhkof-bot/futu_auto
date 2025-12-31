"""
Chart generation module for strategy visualization.
Enhanced version with professional styling and interactive features.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ChartGenerator:
    """
    Professional chart generator for trading strategy analysis.
    """
    
    def __init__(self, style: str = 'professional', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize chart generator.
        
        Args:
            style: Chart style ('professional', 'minimal', 'dark')
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'danger': '#C73E1D',
            'neutral': '#6C757D'
        }
        
        # Set matplotlib style
        if style == 'dark':
            plt.style.use('dark_background')
        else:
            plt.style.use('seaborn-v0_8-whitegrid')
    
    def create_equity_curve(self, 
                          equity_curve: pd.Series,
                          benchmark: Optional[pd.Series] = None,
                          title: str = "Portfolio Equity Curve",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create equity curve chart.
        
        Args:
            equity_curve: Portfolio value over time
            benchmark: Benchmark for comparison
            title: Chart title
            save_path: Path to save chart
            
        Returns:
            Matplotlib figure
        """
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot equity curve
        ax.plot(equity_curve.index, equity_curve.values, 
               color=self.colors['primary'], linewidth=2, label='Strategy')
        
        # Plot benchmark if provided
        if benchmark is not None:
            # Normalize benchmark to same starting value
            normalized_benchmark = benchmark / benchmark.iloc[0] * equity_curve.iloc[0]
            ax.plot(benchmark.index, normalized_benchmark.values,
                   color=self.colors['secondary'], linewidth=2, 
                   label='Benchmark', alpha=0.7)
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # Add performance metrics as text
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        ax.text(0.02, 0.98, f'Total Return: {total_return:.1f}%', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_drawdown_chart(self, 
                            equity_curve: pd.Series,
                            title: str = "Drawdown Analysis",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create drawdown chart.
        
        Args:
            equity_curve: Portfolio value over time
            title: Chart title
            save_path: Path to save chart
            
        Returns:
            Matplotlib figure
        """
        
        # Calculate drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak * 100
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                      gridspec_kw={'height_ratios': [2, 1]})
        
        # Equity curve
        ax1.plot(equity_curve.index, equity_curve.values, 
                color=self.colors['primary'], linewidth=2)
        ax1.fill_between(equity_curve.index, equity_curve.values, peak.values,
                        alpha=0.3, color=self.colors['danger'])
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2.fill_between(drawdown.index, drawdown.values, 0,
                        color=self.colors['danger'], alpha=0.7)
        ax2.plot(drawdown.index, drawdown.values, 
                color=self.colors['danger'], linewidth=1)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        plt.xticks(rotation=45)
        
        # Add max drawdown info
        max_dd = drawdown.min()
        ax2.text(0.02, 0.02, f'Max Drawdown: {max_dd:.1f}%', 
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_returns_distribution(self, 
                                  returns: pd.Series,
                                  title: str = "Returns Distribution",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create returns distribution chart.
        
        Args:
            returns: Return series
            title: Chart title
            save_path: Path to save chart
            
        Returns:
            Matplotlib figure
        """
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        # Histogram
        ax1.hist(returns * 100, bins=50, alpha=0.7, color=self.colors['primary'], 
                edgecolor='black', linewidth=0.5)
        ax1.axvline(returns.mean() * 100, color=self.colors['danger'], 
                   linestyle='--', linewidth=2, label=f'Mean: {returns.mean()*100:.2f}%')
        ax1.set_title('Returns Histogram', fontweight='bold')
        ax1.set_xlabel('Daily Returns (%)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Distribution)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Box plot
        ax3.boxplot(returns * 100, vert=True, patch_artist=True,
                   boxprops=dict(facecolor=self.colors['primary'], alpha=0.7))
        ax3.set_title('Returns Box Plot', fontweight='bold')
        ax3.set_ylabel('Daily Returns (%)')
        ax3.grid(True, alpha=0.3)
        
        # Rolling volatility
        rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100
        ax4.plot(rolling_vol.index, rolling_vol.values, 
                color=self.colors['secondary'], linewidth=2)
        ax4.set_title('30-Day Rolling Volatility', fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Annualized Volatility (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_trade_analysis_chart(self, 
                                  trades: List,
                                  title: str = "Trade Analysis",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive trade analysis chart.
        
        Args:
            trades: List of trade objects
            title: Chart title
            save_path: Path to save chart
            
        Returns:
            Matplotlib figure
        """
        
        if not trades:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No trades to display', ha='center', va='center',
                   transform=ax.transAxes, fontsize=16)
            return fig
        
        # Extract trade data
        trade_returns = [t.pnl_pct * 100 for t in trades]
        trade_dates = [t.exit_time for t in trades]
        holding_periods = [t.holding_period for t in trades]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        # Trade returns over time
        colors = [self.colors['success'] if r > 0 else self.colors['danger'] for r in trade_returns]
        ax1.scatter(trade_dates, trade_returns, c=colors, alpha=0.7, s=50)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_title('Trade Returns Over Time', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Trade Return (%)')
        ax1.grid(True, alpha=0.3)
        
        # Trade returns histogram
        ax2.hist(trade_returns, bins=30, alpha=0.7, color=self.colors['primary'],
                edgecolor='black', linewidth=0.5)
        ax2.axvline(np.mean(trade_returns), color=self.colors['danger'], 
                   linestyle='--', linewidth=2, label=f'Mean: {np.mean(trade_returns):.2f}%')
        ax2.set_title('Trade Returns Distribution', fontweight='bold')
        ax2.set_xlabel('Trade Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Holding period analysis
        ax3.hist(holding_periods, bins=20, alpha=0.7, color=self.colors['secondary'],
                edgecolor='black', linewidth=0.5)
        ax3.axvline(np.mean(holding_periods), color=self.colors['danger'], 
                   linestyle='--', linewidth=2, label=f'Mean: {np.mean(holding_periods):.1f} days')
        ax3.set_title('Holding Period Distribution', fontweight='bold')
        ax3.set_xlabel('Holding Period (Days)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Cumulative P&L
        cumulative_pnl = np.cumsum([t.pnl for t in trades])
        ax4.plot(range(1, len(cumulative_pnl) + 1), cumulative_pnl, 
                color=self.colors['primary'], linewidth=2, marker='o', markersize=4)
        ax4.set_title('Cumulative P&L by Trade', fontweight='bold')
        ax4.set_xlabel('Trade Number')
        ax4.set_ylabel('Cumulative P&L ($)')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_strategy_signals_chart(self, 
                                    data: pd.DataFrame,
                                    signals: List,
                                    title: str = "Strategy Signals",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create strategy signals visualization.
        
        Args:
            data: Price data DataFrame
            signals: List of signal objects
            title: Chart title
            save_path: Path to save chart
            
        Returns:
            Matplotlib figure
        """
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize,
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Price chart with signals
        ax1.plot(data.index, data['Close'], color='black', linewidth=1, label='Price')
        
        # Add moving averages if available
        if 'SMA_20' in data.columns:
            ax1.plot(data.index, data['SMA_20'], color=self.colors['primary'], 
                    linewidth=1, alpha=0.7, label='SMA 20')
        if 'SMA_50' in data.columns:
            ax1.plot(data.index, data['SMA_50'], color=self.colors['secondary'], 
                    linewidth=1, alpha=0.7, label='SMA 50')
        
        # Plot signals
        if signals:
            buy_signals = [s for s in signals if s.signal_type.value == 1]
            sell_signals = [s for s in signals if s.signal_type.value == -1]
            
            if buy_signals:
                buy_dates = [s.timestamp for s in buy_signals]
                buy_prices = [s.price for s in buy_signals]
                ax1.scatter(buy_dates, buy_prices, color=self.colors['success'], 
                           marker='^', s=100, label='Buy Signal', zorder=5)
            
            if sell_signals:
                sell_dates = [s.timestamp for s in sell_signals]
                sell_prices = [s.price for s in sell_signals]
                ax1.scatter(sell_dates, sell_prices, color=self.colors['danger'], 
                           marker='v', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title(title, fontweight='bold')
        ax1.set_ylabel('Price ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # RSI subplot if available
        if 'RSI_14' in data.columns:
            ax2.plot(data.index, data['RSI_14'], color=self.colors['neutral'], linewidth=1)
            ax2.axhline(y=70, color=self.colors['danger'], linestyle='--', alpha=0.7)
            ax2.axhline(y=30, color=self.colors['success'], linestyle='--', alpha=0.7)
            ax2.fill_between(data.index, 30, 70, alpha=0.1, color=self.colors['neutral'])
            ax2.set_ylabel('RSI')
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Date')
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_performance_comparison(self, 
                                   results_dict: Dict[str, Dict],
                                   metrics: List[str] = None,
                                   title: str = "Strategy Comparison",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create performance comparison chart for multiple strategies.
        
        Args:
            results_dict: Dictionary of strategy results
            metrics: List of metrics to compare
            title: Chart title
            save_path: Path to save chart
            
        Returns:
            Matplotlib figure
        """
        
        if metrics is None:
            metrics = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)']
        
        # Prepare data
        comparison_data = {}
        for strategy_name, results in results_dict.items():
            comparison_data[strategy_name] = {
                metric: results.get(metric, 0) for metric in metrics
            }
        
        df = pd.DataFrame(comparison_data).T
        
        # Create subplots
        n_metrics = len(metrics)
        cols = 2
        rows = (n_metrics + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        if rows == 1:
            axes = [axes]
        axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            values = df[metric].values
            strategies = df.index.tolist()
            
            bars = ax.bar(strategies, values, color=self.colors['primary'], alpha=0.7)
            
            # Color bars based on performance
            if 'Return' in metric or 'Ratio' in metric:
                # Higher is better
                colors = [self.colors['success'] if v > 0 else self.colors['danger'] for v in values]
            elif 'Drawdown' in metric:
                # Lower (less negative) is better
                colors = [self.colors['success'] if v > -10 else self.colors['danger'] for v in values]
            else:
                colors = [self.colors['primary']] * len(values)
            
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_title(metric, fontweight='bold')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(self, 
                                   backtest_results: Dict,
                                   data: pd.DataFrame) -> go.Figure:
        """
        Create interactive dashboard using Plotly.
        
        Args:
            backtest_results: Backtest results dictionary
            data: Price data DataFrame
            
        Returns:
            Plotly figure
        """
        
        equity_curve = backtest_results.get('equity_curve', pd.Series())
        trades = backtest_results.get('trades', [])
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Equity Curve', 'Drawdown', 'Price & Signals', 
                          'Returns Distribution', 'Trade Analysis', 'Performance Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(x=equity_curve.index, y=equity_curve.values,
                      mode='lines', name='Portfolio Value',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak * 100
        
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values,
                      mode='lines', name='Drawdown (%)',
                      fill='tonexty', fillcolor='rgba(255,0,0,0.3)',
                      line=dict(color='red', width=1)),
            row=1, col=2
        )
        
        # Price and signals
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Close'],
                      mode='lines', name='Price',
                      line=dict(color='black', width=1)),
            row=2, col=1
        )
        
        # Add trade signals
        if trades:
            buy_trades = [t for t in trades if hasattr(t, 'entry_time')]
            if buy_trades:
                fig.add_trace(
                    go.Scatter(x=[t.entry_time for t in buy_trades],
                              y=[t.entry_price for t in buy_trades],
                              mode='markers', name='Entry Points',
                              marker=dict(color='green', size=8, symbol='triangle-up')),
                    row=2, col=1
                )
        
        # Returns distribution
        if len(equity_curve) > 1:
            returns = equity_curve.pct_change().dropna() * 100
            fig.add_trace(
                go.Histogram(x=returns.values, name='Returns Distribution',
                           marker_color='lightblue', opacity=0.7),
                row=2, col=2
            )
        
        # Trade analysis
        if trades:
            trade_returns = [t.pnl_pct * 100 for t in trades]
            fig.add_trace(
                go.Scatter(x=list(range(1, len(trade_returns) + 1)),
                          y=np.cumsum(trade_returns),
                          mode='lines+markers', name='Cumulative Trade Returns',
                          line=dict(color='purple', width=2)),
                row=3, col=1
            )
        
        # Performance metrics table
        metrics = backtest_results.get('performance_metrics', {})
        if metrics:
            metric_names = list(metrics.keys())[:10]  # Top 10 metrics
            metric_values = [str(metrics[name]) for name in metric_names]
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Value'],
                               fill_color='paleturquoise',
                               align='left'),
                    cells=dict(values=[metric_names, metric_values],
                              fill_color='lavender',
                              align='left')),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            showlegend=True,
            title_text="Interactive Strategy Dashboard",
            title_x=0.5
        )
        
        return fig