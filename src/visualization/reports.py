"""
Report generation module for comprehensive strategy reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import os
from pathlib import Path

class ReportGenerator:
    """
    Comprehensive report generator for trading strategies.
    """
    
    def __init__(self, output_dir: str = "./reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_html_report(self, 
                           analysis_results: Dict,
                           backtest_results: Dict,
                           strategy_name: str = "Strategy") -> str:
        """
        Generate comprehensive HTML report.
        
        Args:
            analysis_results: Results from strategy analyzer
            backtest_results: Results from backtesting engine
            strategy_name: Name of the strategy
            
        Returns:
            Path to generated HTML report
        """
        
        # Extract key data
        performance = analysis_results.get('performance_metrics', {})
        summary = analysis_results.get('summary', {})
        time_analysis = analysis_results.get('time_analysis', {})
        risk_analysis = analysis_results.get('risk_analysis', {})
        
        # Generate HTML content
        html_content = self._create_html_template(
            strategy_name, performance, summary, time_analysis, risk_analysis, backtest_results
        )
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{strategy_name}_report_{timestamp}.html"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(filepath)
    
    def generate_csv_export(self, 
                          backtest_results: Dict,
                          strategy_name: str = "Strategy") -> str:
        """
        Generate CSV export of key results.
        
        Args:
            backtest_results: Results from backtesting engine
            strategy_name: Name of the strategy
            
        Returns:
            Path to generated CSV file
        """
        
        # Create comprehensive data export
        export_data = {}
        
        # Equity curve
        equity_curve = backtest_results.get('equity_curve', pd.Series())
        if len(equity_curve) > 0:
            export_data['Date'] = equity_curve.index
            export_data['Portfolio_Value'] = equity_curve.values
            
            # Calculate additional metrics
            returns = equity_curve.pct_change().dropna()
            export_data['Daily_Returns'] = returns.reindex(equity_curve.index, fill_value=0)
            
            # Drawdown
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak
            export_data['Drawdown'] = drawdown.values
        
        # Create DataFrame
        df = pd.DataFrame(export_data)
        
        # Add trade data if available
        trades = backtest_results.get('trades', [])
        if trades:
            trade_data = []
            for i, trade in enumerate(trades):
                trade_data.append({
                    'Trade_Number': i + 1,
                    'Entry_Date': trade.entry_time,
                    'Exit_Date': trade.exit_time,
                    'Side': trade.side,
                    'Entry_Price': trade.entry_price,
                    'Exit_Price': trade.exit_price,
                    'Quantity': trade.quantity,
                    'PnL': trade.pnl,
                    'PnL_Percent': trade.pnl_pct * 100,
                    'Holding_Period': trade.holding_period,
                    'Exit_Reason': trade.exit_reason
                })
            
            trades_df = pd.DataFrame(trade_data)
        
        # Save files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main results
        main_file = self.output_dir / f"{strategy_name}_results_{timestamp}.csv"
        df.to_csv(main_file, index=False)
        
        # Trades file
        if trades:
            trades_file = self.output_dir / f"{strategy_name}_trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
        
        return str(main_file)
    
    def _create_html_template(self, 
                            strategy_name: str,
                            performance: Dict,
                            summary: Dict,
                            time_analysis: Dict,
                            risk_analysis: Dict,
                            backtest_results: Dict) -> str:
        """Create HTML report template."""
        
        # Extract key metrics with safe defaults
        total_return = performance.get('Total Return (%)', 0)
        sharpe_ratio = performance.get('Sharpe Ratio', 0)
        max_drawdown = performance.get('Max Drawdown (%)', 0)
        win_rate = performance.get('Win Rate (%)', 0)
        total_trades = performance.get('Total Trades', 0)
        
        # Summary info
        rating = summary.get('overall_rating', 'N/A')
        score = summary.get('performance_score', 'N/A')
        strengths = summary.get('key_strengths', [])
        weaknesses = summary.get('key_weaknesses', [])
        recommendation = summary.get('recommendation', 'N/A')
        
        # Time analysis
        best_month = time_analysis.get('best_month_pct', 0)
        worst_month = time_analysis.get('worst_month_pct', 0)
        positive_months = time_analysis.get('positive_months', 0)
        
        # Risk analysis
        var_95 = risk_analysis.get('var_95_pct', 0)
        max_consecutive_losses = risk_analysis.get('max_consecutive_losses', 0)
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{strategy_name} - Strategy Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #2E86AB;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #2E86AB;
            margin: 0;
            font-size: 2.5em;
        }}
        .header .subtitle {{
            color: #666;
            font-size: 1.2em;
            margin-top: 10px;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #2E86AB, #A23B72);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .summary-box h2 {{
            margin: 0 0 15px 0;
            font-size: 1.8em;
        }}
        .summary-stats {{
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }}
        .stat-item {{
            text-align: center;
            margin: 10px;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            display: block;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #2E86AB;
            border-bottom: 2px solid #2E86AB;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #2E86AB;
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            color: #333;
            font-size: 1.1em;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2E86AB;
        }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        .neutral {{ color: #6c757d; }}
        .rating {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .rating.excellent {{ background: #28a745; color: white; }}
        .rating.good {{ background: #17a2b8; color: white; }}
        .rating.average {{ background: #ffc107; color: black; }}
        .rating.poor {{ background: #dc3545; color: white; }}
        .list-item {{
            background: #f8f9fa;
            margin: 5px 0;
            padding: 10px;
            border-radius: 5px;
            border-left: 3px solid #2E86AB;
        }}
        .strength {{ border-left-color: #28a745; }}
        .weakness {{ border-left-color: #dc3545; }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }}
        @media (max-width: 768px) {{
            .summary-stats {{
                flex-direction: column;
            }}
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{strategy_name}</h1>
            <div class="subtitle">Strategy Performance Report</div>
            <div class="subtitle">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>

        <div class="summary-box">
            <h2>Executive Summary</h2>
            <div class="summary-stats">
                <div class="stat-item">
                    <span class="stat-value">{total_return:.1f}%</span>
                    <span class="stat-label">Total Return</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">{sharpe_ratio:.2f}</span>
                    <span class="stat-label">Sharpe Ratio</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">{max_drawdown:.1f}%</span>
                    <span class="stat-label">Max Drawdown</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">{win_rate:.1f}%</span>
                    <span class="stat-label">Win Rate</span>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Overall Assessment</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Overall Rating</h3>
                    <div class="metric-value">
                        <span class="rating {rating.lower()}">{rating}</span>
                    </div>
                </div>
                <div class="metric-card">
                    <h3>Performance Score</h3>
                    <div class="metric-value">{score}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Key Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Total Return</h3>
                    <div class="metric-value {'positive' if total_return > 0 else 'negative'}">{total_return:.2f}%</div>
                </div>
                <div class="metric-card">
                    <h3>Annualized Return</h3>
                    <div class="metric-value">{performance.get('Annualized Return (%)', 0):.2f}%</div>
                </div>
                <div class="metric-card">
                    <h3>Volatility</h3>
                    <div class="metric-value">{performance.get('Volatility (%)', 0):.2f}%</div>
                </div>
                <div class="metric-card">
                    <h3>Sharpe Ratio</h3>
                    <div class="metric-value {'positive' if sharpe_ratio > 1 else 'neutral'}">{sharpe_ratio:.3f}</div>
                </div>
                <div class="metric-card">
                    <h3>Maximum Drawdown</h3>
                    <div class="metric-value negative">{max_drawdown:.2f}%</div>
                </div>
                <div class="metric-card">
                    <h3>Calmar Ratio</h3>
                    <div class="metric-value">{performance.get('Calmar Ratio', 0):.3f}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Trading Statistics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Total Trades</h3>
                    <div class="metric-value">{total_trades}</div>
                </div>
                <div class="metric-card">
                    <h3>Win Rate</h3>
                    <div class="metric-value {'positive' if win_rate > 50 else 'neutral'}">{win_rate:.1f}%</div>
                </div>
                <div class="metric-card">
                    <h3>Profit Factor</h3>
                    <div class="metric-value">{performance.get('Profit Factor', 0):.2f}</div>
                </div>
                <div class="metric-card">
                    <h3>Average Win</h3>
                    <div class="metric-value positive">{performance.get('Avg Win (%)', 0):.2f}%</div>
                </div>
                <div class="metric-card">
                    <h3>Average Loss</h3>
                    <div class="metric-value negative">{performance.get('Avg Loss (%)', 0):.2f}%</div>
                </div>
                <div class="metric-card">
                    <h3>Avg Holding Period</h3>
                    <div class="metric-value">{performance.get('Avg Holding Period (days)', 0):.1f} days</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Risk Analysis</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Value at Risk (95%)</h3>
                    <div class="metric-value negative">{var_95:.2f}%</div>
                </div>
                <div class="metric-card">
                    <h3>Max Consecutive Losses</h3>
                    <div class="metric-value">{max_consecutive_losses}</div>
                </div>
                <div class="metric-card">
                    <h3>Sortino Ratio</h3>
                    <div class="metric-value">{performance.get('Sortino Ratio', 0):.3f}</div>
                </div>
                <div class="metric-card">
                    <h3>Max Drawdown Duration</h3>
                    <div class="metric-value">{performance.get('Max Drawdown Duration (days)', 0)} days</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Time Analysis</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Best Month</h3>
                    <div class="metric-value positive">{best_month:.1f}%</div>
                </div>
                <div class="metric-card">
                    <h3>Worst Month</h3>
                    <div class="metric-value negative">{worst_month:.1f}%</div>
                </div>
                <div class="metric-card">
                    <h3>Positive Months</h3>
                    <div class="metric-value">{positive_months}</div>
                </div>
                <div class="metric-card">
                    <h3>Trading Period</h3>
                    <div class="metric-value">{performance.get('Years', 0):.1f} years</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Strategy Strengths</h2>
            {''.join([f'<div class="list-item strength">✓ {strength}</div>' for strength in strengths]) if strengths else '<div class="list-item">No specific strengths identified</div>'}
        </div>

        <div class="section">
            <h2>Areas for Improvement</h2>
            {''.join([f'<div class="list-item weakness">⚠ {weakness}</div>' for weakness in weaknesses]) if weaknesses else '<div class="list-item">No specific weaknesses identified</div>'}
        </div>

        <div class="section">
            <h2>Recommendation</h2>
            <div class="list-item">
                <strong>Recommendation:</strong> {recommendation}
            </div>
        </div>

        <div class="footer">
            <p>This report was generated by the Trend Backtesting Framework</p>
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html_template
    
    def generate_summary_report(self, 
                              analysis_results: Dict,
                              strategy_name: str = "Strategy") -> str:
        """
        Generate a concise text summary report.
        
        Args:
            analysis_results: Results from strategy analyzer
            strategy_name: Name of the strategy
            
        Returns:
            Path to generated text report
        """
        
        performance = analysis_results.get('performance_metrics', {})
        summary = analysis_results.get('summary', {})
        
        report_content = f"""
{strategy_name.upper()} - STRATEGY PERFORMANCE SUMMARY
{'='*60}

EXECUTIVE SUMMARY
Overall Rating: {summary.get('overall_rating', 'N/A')}
Performance Score: {summary.get('performance_score', 'N/A')}

KEY PERFORMANCE INDICATORS
Total Return: {performance.get('Total Return (%)', 0):.2f}%
Annualized Return: {performance.get('Annualized Return (%)', 0):.2f}%
Sharpe Ratio: {performance.get('Sharpe Ratio', 0):.3f}
Maximum Drawdown: {performance.get('Max Drawdown (%)', 0):.2f}%
Win Rate: {performance.get('Win Rate (%)', 0):.1f}%
Profit Factor: {performance.get('Profit Factor', 0):.2f}

TRADING STATISTICS
Total Trades: {performance.get('Total Trades', 0)}
Average Win: {performance.get('Avg Win (%)', 0):.2f}%
Average Loss: {performance.get('Avg Loss (%)', 0):.2f}%
Average Holding Period: {performance.get('Avg Holding Period (days)', 0):.1f} days

RISK METRICS
Volatility: {performance.get('Volatility (%)', 0):.2f}%
Sortino Ratio: {performance.get('Sortino Ratio', 0):.3f}
Calmar Ratio: {performance.get('Calmar Ratio', 0):.3f}
VaR (95%): {performance.get('VaR 95% (%)', 0):.2f}%

STRENGTHS
{chr(10).join(['• ' + s for s in summary.get('key_strengths', ['None identified'])])}

WEAKNESSES
{chr(10).join(['• ' + w for w in summary.get('key_weaknesses', ['None identified'])])}

RECOMMENDATION
{summary.get('recommendation', 'No recommendation available')}

Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{strategy_name}_summary_{timestamp}.txt"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return str(filepath)