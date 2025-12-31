#!/usr/bin/env python3
"""
多指标确认策略回测示例。
RSI + MACD + 布林带同时出现买入信号时买入，持有1个月后卖出。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_fetcher import DataFetcher
from src.data.data_processor import DataProcessor
from src.strategies.multi_indicator_strategy import MultiIndicatorStrategy
from src.backtest.engine import BacktestEngine
from src.analytics.analyzer import StrategyAnalyzer
from src.visualization.charts import ChartGenerator

def main():
    """运行多指标确认策略回测。"""
    
    print("=" * 60)
    print("多指标确认策略回测")
    print("买入条件: RSI超卖 + MACD金叉 + 触及布林带下轨")
    print("卖出条件: 持有20个交易日（约1个月）")
    print("=" * 60)
    
    # 配置
    symbol = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    initial_capital = 100000
    
    print(f"\n标的: {symbol}")
    print(f"时间段: {start_date} 至 {end_date}")
    print(f"初始资金: ${initial_capital:,}")
    
    # 1. 获取数据
    print("\n[1/4] 获取数据...")
    data_fetcher = DataFetcher(cache_dir=".cache")
    data = data_fetcher.fetch_stock_data(symbol, start_date, end_date)
    
    data_processor = DataProcessor()
    processed_data = data_processor.add_technical_indicators(data)
    print(f"✓ 加载 {len(processed_data)} 天数据")
    
    # 2. 创建策略
    print("\n[2/4] 创建多指标确认策略...")
    strategy = MultiIndicatorStrategy(
        rsi_period=14,
        rsi_oversold=30,          # RSI低于30视为超卖
        rsi_overbought=70,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        bb_period=20,
        bb_std=2.0,
        holding_days=20,          # 持有20个交易日（约1个月）
        stop_loss_pct=0.08,       # 8%止损
        take_profit_pct=0.15      # 15%止盈
    )
    print("✓ 策略参数:")
    print(f"  - RSI超卖阈值: < 30")
    print(f"  - MACD: 12/26/9")
    print(f"  - 布林带: 20日, 2倍标准差")
    print(f"  - 持有天数: 20天")
    print(f"  - 止损: 8%, 止盈: 15%")
    
    # 3. 运行回测
    print("\n[3/4] 运行回测...")
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=0.001,
        slippage_rate=0.0005,
        use_kelly_sizing=False,   # 使用固定仓位
        max_position_size=0.95    # 最大95%仓位
    )
    
    results = engine.run_backtest(processed_data, strategy, symbol)
    
    # 4. 显示结果
    print("\n[4/4] 回测结果")
    print("=" * 40)
    print(f"最终价值: ${results['final_value']:,.2f}")
    print(f"总收益: {results['total_return_pct']:.2f}%")
    print(f"年化收益: {results['annualized_return_pct']:.2f}%")
    print(f"夏普比率: {results['sharpe_ratio']:.3f}")
    print(f"最大回撤: {results['max_drawdown_pct']:.2f}%")
    print(f"总交易次数: {results['total_trades']}")
    print(f"胜率: {results['win_rate_pct']:.1f}%")
    print(f"盈利因子: {results['profit_factor']:.2f}")
    print(f"平均持仓天数: {results['avg_holding_period_days']:.1f}天")
    
    # 显示交易详情
    if results['trades']:
        print("\n交易记录:")
        print("-" * 80)
        for i, trade in enumerate(results['trades'][:10], 1):  # 只显示前10笔
            pnl_sign = "+" if trade.pnl > 0 else ""
            print(f"{i}. {trade.entry_time.strftime('%Y-%m-%d')} -> {trade.exit_time.strftime('%Y-%m-%d')} | "
                  f"入场: ${trade.entry_price:.2f} | 出场: ${trade.exit_price:.2f} | "
                  f"盈亏: {pnl_sign}{trade.pnl_pct*100:.2f}% | {trade.exit_reason}")
        
        if len(results['trades']) > 10:
            print(f"... 共 {len(results['trades'])} 笔交易")
    
    # 生成图表
    print("\n生成图表...")
    os.makedirs("reports", exist_ok=True)
    
    try:
        chart_generator = ChartGenerator(style='professional')
        
        # 权益曲线
        if len(results['equity_curve']) > 0:
            chart_generator.create_equity_curve(
                results['equity_curve'],
                title=f"多指标策略 - {symbol} 权益曲线",
                save_path="./reports/multi_indicator_equity.png"
            )
            print("✓ 权益曲线已保存")
        
        # 回撤图
        if len(results['equity_curve']) > 0:
            chart_generator.create_drawdown_chart(
                results['equity_curve'],
                title=f"多指标策略 - {symbol} 回撤分析",
                save_path="./reports/multi_indicator_drawdown.png"
            )
            print("✓ 回撤图已保存")
        
    except Exception as e:
        print(f"⚠ 图表生成失败: {e}")
    
    print("\n" + "=" * 60)
    print("回测完成！报告已保存至 ./reports/")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    os.makedirs(".cache", exist_ok=True)
    main()
