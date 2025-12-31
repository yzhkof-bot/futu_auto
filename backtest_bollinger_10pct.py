"""
布林带下轨策略回测
买入条件：价格低于布林带下轨2%
卖出条件：持有2个月（约40个交易日）
测试股票：纳斯达克100随机股票
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import sys
import os
import matplotlib.pyplot as plt

# 添加src到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.data_fetcher import DataFetcher

# 纳斯达克100成分股（部分常见股票）
NASDAQ_100_SYMBOLS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 
    'AVGO', 'COST', 'ASML', 'PEP', 'CSCO', 'AZN', 'ADBE', 'NFLX',
    'AMD', 'TMUS', 'TXN', 'QCOM', 'INTC', 'INTU', 'CMCSA', 'AMGN',
    'HON', 'AMAT', 'ISRG', 'BKNG', 'VRTX', 'SBUX', 'GILD', 'ADI',
    'MDLZ', 'ADP', 'REGN', 'LRCX', 'PANW', 'MU', 'SNPS', 'KLAC',
    'CDNS', 'PYPL', 'MELI', 'ORLY', 'MAR', 'CTAS', 'ABNB', 'MRVL',
    'FTNT', 'CSX', 'MNST', 'WDAY', 'PCAR', 'DXCM', 'NXPI', 'CPRT',
    'ROST', 'PAYX', 'ODFL', 'KDP', 'MCHP', 'KHC', 'IDXX', 'AEP',
    'LULU', 'FAST', 'VRSK', 'EXC', 'GEHC', 'CTSH', 'EA', 'ON',
    'CSGP', 'BKR', 'FANG', 'XEL', 'ANSS', 'DDOG', 'ZS', 'TEAM',
    'ILMN', 'WBD', 'DLTR', 'BIIB', 'WBA', 'SIRI', 'LCID', 'RIVN'
]


def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """计算布林带指标"""
    df = data.copy()
    df['BB_Middle'] = df['Close'].rolling(window=period).mean()
    rolling_std = df['Close'].rolling(window=period).std()
    df['BB_Upper'] = df['BB_Middle'] + (rolling_std * std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (rolling_std * std_dev)
    return df


def run_single_stock_backtest(
    data: pd.DataFrame, 
    symbol: str,
    below_threshold: float = 0.10,  # 低于下轨10%
    holding_days: int = 20,  # 持有约1个月
    initial_capital: float = 100000
) -> Dict:
    """
    对单只股票运行回测
    
    Args:
        data: 股票数据
        symbol: 股票代码
        below_threshold: 低于布林带下轨的阈值（10% = 0.10）
        holding_days: 持有天数
        initial_capital: 初始资金
    
    Returns:
        回测结果字典
    """
    # 计算布林带
    df = calculate_bollinger_bands(data)
    
    trades = []
    position = None  # 当前持仓信息
    
    for i in range(20, len(df)):  # 从第20天开始（需要布林带数据）
        current_date = df.index[i]
        current_price = df.iloc[i]['Close']
        bb_lower = df.iloc[i]['BB_Lower']
        
        if pd.isna(bb_lower):
            continue
        
        # 检查是否需要卖出（持有期满）
        if position is not None:
            days_held = (current_date - position['entry_date']).days
            if days_held >= holding_days:
                # 卖出
                exit_price = current_price
                pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                
                trades.append({
                    'symbol': symbol,
                    'entry_date': position['entry_date'],
                    'exit_date': current_date,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'days_held': days_held,
                    'bb_lower_at_entry': position['bb_lower'],
                    'below_pct_at_entry': position['below_pct']
                })
                position = None
        
        # 检查买入条件（没有持仓时）
        if position is None:
            # 计算价格低于下轨的百分比
            below_pct = (bb_lower - current_price) / bb_lower
            
            if below_pct >= below_threshold:
                # 买入
                position = {
                    'entry_date': current_date,
                    'entry_price': current_price,
                    'bb_lower': bb_lower,
                    'below_pct': below_pct
                }
    
    # 如果还有未平仓的持仓，在最后一天平仓
    if position is not None:
        current_date = df.index[-1]
        exit_price = df.iloc[-1]['Close']
        days_held = (current_date - position['entry_date']).days
        pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        
        trades.append({
            'symbol': symbol,
            'entry_date': position['entry_date'],
            'exit_date': current_date,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'days_held': days_held,
            'bb_lower_at_entry': position['bb_lower'],
            'below_pct_at_entry': position['below_pct'],
            'note': '未满持有期强制平仓'
        })
    
    # 计算统计数据
    if not trades:
        return {
            'symbol': symbol,
            'total_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'total_return': 0,
            'trades': []
        }
    
    trades_df = pd.DataFrame(trades)
    winning_trades = trades_df[trades_df['pnl_pct'] > 0]
    losing_trades = trades_df[trades_df['pnl_pct'] <= 0]
    
    return {
        'symbol': symbol,
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / len(trades) * 100,
        'avg_return': trades_df['pnl_pct'].mean() * 100,
        'median_return': trades_df['pnl_pct'].median() * 100,
        'max_return': trades_df['pnl_pct'].max() * 100,
        'min_return': trades_df['pnl_pct'].min() * 100,
        'total_return': (1 + trades_df['pnl_pct']).prod() - 1,
        'avg_days_held': trades_df['days_held'].mean(),
        'trades': trades
    }


def main():
    """主函数"""
    print("=" * 60)
    print("布林带下轨策略回测")
    print("买入条件：价格低于布林带下轨2%")
    print("卖出条件：持有2个月（约40个交易日）")
    print("=" * 60)
    
    # 参数设置
    num_stocks = 50  # 随机选择的股票数量（增加样本）
    start_date = "2015-01-01"  # 更长的回测期
    end_date = "2024-12-30"
    below_threshold = 0.02  # 低于下轨2%
    holding_days = 40  # 持有约2个月
    
    # 随机选择股票
    random.seed(42)  # 固定随机种子以便复现
    selected_symbols = random.sample(NASDAQ_100_SYMBOLS, min(num_stocks, len(NASDAQ_100_SYMBOLS)))
    
    print(f"\n随机选择 {len(selected_symbols)} 只纳斯达克100股票进行回测:")
    print(f"股票列表: {selected_symbols}")
    print(f"回测期间: {start_date} 至 {end_date}")
    print(f"买入阈值: 低于布林带下轨 {below_threshold*100:.0f}%")
    print(f"持有天数: {holding_days} 天")
    print("-" * 60)
    
    # 获取数据
    fetcher = DataFetcher()
    all_results = []
    all_trades = []
    
    for symbol in selected_symbols:
        try:
            print(f"\n处理 {symbol}...")
            data = fetcher.fetch_stock_data(symbol, start_date, end_date)
            
            if len(data) < 100:
                print(f"  {symbol} 数据不足，跳过")
                continue
            
            result = run_single_stock_backtest(
                data, symbol, 
                below_threshold=below_threshold, 
                holding_days=holding_days
            )
            
            all_results.append(result)
            all_trades.extend(result['trades'])
            
            if result['total_trades'] > 0:
                print(f"  交易次数: {result['total_trades']}, "
                      f"胜率: {result['win_rate']:.1f}%, "
                      f"平均收益: {result['avg_return']:.2f}%")
            else:
                print(f"  无交易信号")
                
        except Exception as e:
            print(f"  {symbol} 处理失败: {e}")
    
    # 汇总统计
    print("\n" + "=" * 60)
    print("汇总统计")
    print("=" * 60)
    
    if not all_trades:
        print("没有产生任何交易")
        return
    
    trades_df = pd.DataFrame(all_trades)
    
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
    losing_trades = len(trades_df[trades_df['pnl_pct'] <= 0])
    
    print(f"\n总交易次数: {total_trades}")
    print(f"盈利交易: {winning_trades}")
    print(f"亏损交易: {losing_trades}")
    print(f"总体胜率: {winning_trades/total_trades*100:.2f}%")
    print(f"\n收益统计:")
    print(f"  平均收益率: {trades_df['pnl_pct'].mean()*100:.2f}%")
    print(f"  中位数收益率: {trades_df['pnl_pct'].median()*100:.2f}%")
    print(f"  最大收益率: {trades_df['pnl_pct'].max()*100:.2f}%")
    print(f"  最小收益率: {trades_df['pnl_pct'].min()*100:.2f}%")
    print(f"  收益率标准差: {trades_df['pnl_pct'].std()*100:.2f}%")
    print(f"\n平均持有天数: {trades_df['days_held'].mean():.1f} 天")
    
    # 按股票统计
    print("\n" + "-" * 60)
    print("各股票统计:")
    print("-" * 60)
    
    stock_stats = trades_df.groupby('symbol').agg({
        'pnl_pct': ['count', 'mean', lambda x: (x > 0).sum() / len(x) * 100],
    }).round(2)
    stock_stats.columns = ['交易次数', '平均收益%', '胜率%']
    stock_stats['平均收益%'] = stock_stats['平均收益%'] * 100
    stock_stats = stock_stats.sort_values('交易次数', ascending=False)
    print(stock_stats.to_string())
    
    # 按年份统计
    print("\n" + "-" * 60)
    print("按年份统计:")
    print("-" * 60)
    
    trades_df['year'] = pd.to_datetime(trades_df['entry_date']).dt.year
    yearly_stats = trades_df.groupby('year').agg({
        'pnl_pct': ['count', 'mean', lambda x: (x > 0).sum() / len(x) * 100],
    }).round(2)
    yearly_stats.columns = ['交易次数', '平均收益%', '胜率%']
    yearly_stats['平均收益%'] = yearly_stats['平均收益%'] * 100
    print(yearly_stats.to_string())
    
    # 显示部分交易详情
    print("\n" + "-" * 60)
    print("最近10笔交易详情:")
    print("-" * 60)
    
    recent_trades = trades_df.tail(10)[['symbol', 'entry_date', 'exit_date', 'entry_price', 
                                         'exit_price', 'pnl_pct', 'days_held', 'below_pct_at_entry']]
    recent_trades['pnl_pct'] = recent_trades['pnl_pct'].apply(lambda x: f"{x*100:.2f}%")
    recent_trades['below_pct_at_entry'] = recent_trades['below_pct_at_entry'].apply(lambda x: f"{x*100:.2f}%")
    recent_trades['entry_price'] = recent_trades['entry_price'].apply(lambda x: f"${x:.2f}")
    recent_trades['exit_price'] = recent_trades['exit_price'].apply(lambda x: f"${x:.2f}")
    print(recent_trades.to_string(index=False))
    
    # 保存结果
    output_file = 'reports/bollinger_2pct_backtest_results.csv'
    os.makedirs('reports', exist_ok=True)
    trades_df.to_csv(output_file, index=False)
    print(f"\n详细结果已保存到: {output_file}")
    
    # 绘制图表
    plot_backtest_results(trades_df, below_threshold, holding_days)


def plot_backtest_results(trades_df: pd.DataFrame, below_threshold: float, holding_days: int):
    """绘制回测结果图表"""
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'布林带下轨策略回测结果\n(低于下轨{below_threshold*100:.0f}%, 持有{holding_days}天)', 
                 fontsize=14, fontweight='bold')
    
    # 1. 收益分布直方图
    ax1 = axes[0, 0]
    returns = trades_df['pnl_pct'] * 100
    colors = ['green' if r > 0 else 'red' for r in returns]
    ax1.hist(returns, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='盈亏平衡')
    ax1.axvline(x=returns.mean(), color='orange', linestyle='-', linewidth=2, 
                label=f'平均收益: {returns.mean():.2f}%')
    ax1.set_xlabel('收益率 (%)')
    ax1.set_ylabel('交易次数')
    ax1.set_title('收益分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 按年份胜率柱状图
    ax2 = axes[0, 1]
    trades_df['year'] = pd.to_datetime(trades_df['entry_date']).dt.year
    yearly_stats = trades_df.groupby('year').agg({
        'pnl_pct': ['count', lambda x: (x > 0).sum() / len(x) * 100, 'mean']
    })
    yearly_stats.columns = ['交易次数', '胜率', '平均收益']
    
    x = range(len(yearly_stats))
    bars = ax2.bar(x, yearly_stats['胜率'], color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=1, label='50%基准线')
    ax2.set_xticks(x)
    ax2.set_xticklabels(yearly_stats.index, rotation=45)
    ax2.set_xlabel('年份')
    ax2.set_ylabel('胜率 (%)')
    ax2.set_title('按年份胜率')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上标注数值
    for bar, count in zip(bars, yearly_stats['交易次数']):
        height = bar.get_height()
        ax2.annotate(f'{height:.0f}%\n(n={int(count)})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # 3. 按年份累计收益（更真实的展示）
    ax3 = axes[1, 0]
    trades_df_sorted = trades_df.sort_values('entry_date')
    trades_df_sorted['year'] = pd.to_datetime(trades_df_sorted['entry_date']).dt.year
    
    # 按年份计算平均收益（假设每年等权投资所有信号）
    yearly_returns = trades_df_sorted.groupby('year')['pnl_pct'].mean() * 100
    cumulative_yearly = yearly_returns.cumsum()
    
    x = range(len(cumulative_yearly))
    ax3.bar(x, yearly_returns.values, color='steelblue', alpha=0.6, label='年度平均收益')
    ax3.plot(x, cumulative_yearly.values, color='orange', linewidth=2, marker='o', label='累计收益')
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax3.set_xticks(x)
    ax3.set_xticklabels(yearly_returns.index, rotation=45)
    ax3.set_xlabel('年份')
    ax3.set_ylabel('收益率 (%)')
    ax3.set_title('年度平均收益与累计收益')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 各股票表现散点图
    ax4 = axes[1, 1]
    stock_stats = trades_df.groupby('symbol').agg({
        'pnl_pct': ['count', 'mean', lambda x: (x > 0).sum() / len(x) * 100]
    })
    stock_stats.columns = ['交易次数', '平均收益', '胜率']
    stock_stats['平均收益'] = stock_stats['平均收益'] * 100
    
    # 只显示交易次数>=2的股票
    stock_stats_filtered = stock_stats[stock_stats['交易次数'] >= 2]
    
    scatter = ax4.scatter(stock_stats_filtered['胜率'], 
                         stock_stats_filtered['平均收益'],
                         s=stock_stats_filtered['交易次数'] * 20,
                         c=stock_stats_filtered['平均收益'],
                         cmap='RdYlGn', alpha=0.7, edgecolors='black')
    
    # 标注股票名称
    for idx, row in stock_stats_filtered.iterrows():
        ax4.annotate(idx, (row['胜率'], row['平均收益']), fontsize=8, alpha=0.8)
    
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax4.axvline(x=50, color='gray', linestyle='--', linewidth=1)
    ax4.set_xlabel('胜率 (%)')
    ax4.set_ylabel('平均收益率 (%)')
    ax4.set_title('各股票表现 (气泡大小=交易次数)')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='平均收益率(%)')
    
    plt.tight_layout()
    
    # 保存图表
    chart_file = 'reports/bollinger_2pct_backtest_chart.png'
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {chart_file}")
    plt.show()


if __name__ == "__main__":
    main()
