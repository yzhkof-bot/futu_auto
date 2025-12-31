"""
W底形态策略回测
买入条件：识别到W底形态
卖出条件：持有2个月（约40个交易日）
测试股票：纳斯达克100随机股票
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sys
import os
import matplotlib.pyplot as plt

# 添加src到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.data_fetcher import DataFetcher

# 纳斯达克100成分股
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


def find_local_minima(prices: pd.Series, window: int = 5) -> List[int]:
    """
    找出局部最小值点
    
    Args:
        prices: 价格序列
        window: 窗口大小，用于判断局部最小
    
    Returns:
        局部最小值的索引列表
    """
    minima = []
    for i in range(window, len(prices) - window):
        # 检查当前点是否是窗口内的最小值
        window_prices = prices.iloc[i-window:i+window+1]
        if prices.iloc[i] == window_prices.min():
            minima.append(i)
    return minima


def find_local_maxima(prices: pd.Series, window: int = 5) -> List[int]:
    """
    找出局部最大值点
    """
    maxima = []
    for i in range(window, len(prices) - window):
        window_prices = prices.iloc[i-window:i+window+1]
        if prices.iloc[i] == window_prices.max():
            maxima.append(i)
    return maxima


def detect_w_bottom(data: pd.DataFrame, 
                    min_pattern_days: int = 10,
                    max_pattern_days: int = 60,
                    tolerance: float = 0.03,
                    neckline_break_pct: float = 0.01) -> List[Dict]:
    """
    检测W底形态（双底形态）
    
    W底形态特征：
    1. 第一个底部（左底）
    2. 反弹形成中间高点（颈线）
    3. 第二个底部（右底），与左底价格接近
    4. 突破颈线确认形态
    
    Args:
        data: 包含OHLCV的DataFrame
        min_pattern_days: 形态最小持续天数
        max_pattern_days: 形态最大持续天数
        tolerance: 两个底部价格的容差（百分比）
        neckline_break_pct: 突破颈线的百分比确认
    
    Returns:
        W底信号列表
    """
    signals = []
    prices = data['Close']
    
    # 找出所有局部最小值和最大值
    minima = find_local_minima(prices, window=5)
    maxima = find_local_maxima(prices, window=5)
    
    # 遍历寻找W底形态
    for i, first_bottom_idx in enumerate(minima):
        first_bottom_price = prices.iloc[first_bottom_idx]
        
        # 寻找第一个底之后的中间高点
        middle_highs = [m for m in maxima if first_bottom_idx < m < first_bottom_idx + max_pattern_days]
        
        for middle_high_idx in middle_highs:
            middle_high_price = prices.iloc[middle_high_idx]
            
            # 中间高点必须比第一个底高出至少3%
            if middle_high_price < first_bottom_price * 1.03:
                continue
            
            # 寻找中间高点之后的第二个底
            second_bottoms = [m for m in minima 
                            if middle_high_idx < m < middle_high_idx + max_pattern_days // 2
                            and m > first_bottom_idx + min_pattern_days]
            
            for second_bottom_idx in second_bottoms:
                second_bottom_price = prices.iloc[second_bottom_idx]
                
                # 检查两个底部价格是否接近（在容差范围内）
                price_diff_pct = abs(first_bottom_price - second_bottom_price) / first_bottom_price
                if price_diff_pct > tolerance:
                    continue
                
                # 第二个底必须低于中间高点
                if second_bottom_price >= middle_high_price * 0.97:
                    continue
                
                # 寻找突破颈线的确认点
                neckline = middle_high_price
                
                # 在第二个底之后寻找突破点
                for confirm_idx in range(second_bottom_idx + 1, min(second_bottom_idx + 20, len(prices))):
                    confirm_price = prices.iloc[confirm_idx]
                    
                    # 确认突破颈线
                    if confirm_price > neckline * (1 + neckline_break_pct):
                        # 检查是否已经有相近的信号（避免重复）
                        is_duplicate = False
                        for existing_signal in signals:
                            if abs(existing_signal['confirm_idx'] - confirm_idx) < 10:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            pattern_days = confirm_idx - first_bottom_idx
                            
                            signals.append({
                                'first_bottom_idx': first_bottom_idx,
                                'first_bottom_price': first_bottom_price,
                                'first_bottom_date': data.index[first_bottom_idx],
                                'middle_high_idx': middle_high_idx,
                                'middle_high_price': middle_high_price,
                                'middle_high_date': data.index[middle_high_idx],
                                'second_bottom_idx': second_bottom_idx,
                                'second_bottom_price': second_bottom_price,
                                'second_bottom_date': data.index[second_bottom_idx],
                                'confirm_idx': confirm_idx,
                                'confirm_price': confirm_price,
                                'confirm_date': data.index[confirm_idx],
                                'neckline': neckline,
                                'pattern_days': pattern_days,
                                'bottom_diff_pct': price_diff_pct * 100
                            })
                        break
    
    return signals


def run_single_stock_backtest(
    data: pd.DataFrame, 
    symbol: str,
    holding_days: int = 40,
) -> Dict:
    """
    对单只股票运行W底策略回测
    """
    # 检测W底形态
    w_signals = detect_w_bottom(data)
    
    trades = []
    last_exit_idx = -1
    
    for signal in w_signals:
        entry_idx = signal['confirm_idx']
        
        # 避免重叠交易
        if entry_idx <= last_exit_idx:
            continue
        
        # 确保有足够的数据进行持有
        exit_idx = entry_idx + holding_days
        if exit_idx >= len(data):
            exit_idx = len(data) - 1
        
        entry_price = signal['confirm_price']
        exit_price = data.iloc[exit_idx]['Close']
        entry_date = signal['confirm_date']
        exit_date = data.index[exit_idx]
        
        pnl_pct = (exit_price - entry_price) / entry_price
        days_held = (exit_date - entry_date).days
        
        trades.append({
            'symbol': symbol,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'days_held': days_held,
            'neckline': signal['neckline'],
            'first_bottom_price': signal['first_bottom_price'],
            'second_bottom_price': signal['second_bottom_price'],
            'pattern_days': signal['pattern_days'],
            'bottom_diff_pct': signal['bottom_diff_pct']
        })
        
        last_exit_idx = exit_idx
    
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
    
    return {
        'symbol': symbol,
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(trades) - len(winning_trades),
        'win_rate': len(winning_trades) / len(trades) * 100,
        'avg_return': trades_df['pnl_pct'].mean() * 100,
        'median_return': trades_df['pnl_pct'].median() * 100,
        'max_return': trades_df['pnl_pct'].max() * 100,
        'min_return': trades_df['pnl_pct'].min() * 100,
        'avg_days_held': trades_df['days_held'].mean(),
        'trades': trades
    }


def plot_backtest_results(trades_df: pd.DataFrame, holding_days: int):
    """绘制回测结果图表"""
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'W底形态策略回测结果\n(持有{holding_days}天)', 
                 fontsize=14, fontweight='bold')
    
    # 1. 收益分布直方图
    ax1 = axes[0, 0]
    returns = trades_df['pnl_pct'] * 100
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
    
    stock_stats_filtered = stock_stats[stock_stats['交易次数'] >= 2]
    
    if len(stock_stats_filtered) > 0:
        scatter = ax4.scatter(stock_stats_filtered['胜率'], 
                             stock_stats_filtered['平均收益'],
                             s=stock_stats_filtered['交易次数'] * 20,
                             c=stock_stats_filtered['平均收益'],
                             cmap='RdYlGn', alpha=0.7, edgecolors='black')
        
        for idx, row in stock_stats_filtered.iterrows():
            ax4.annotate(idx, (row['胜率'], row['平均收益']), fontsize=8, alpha=0.8)
        
        plt.colorbar(scatter, ax=ax4, label='平均收益率(%)')
    
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax4.axvline(x=50, color='gray', linestyle='--', linewidth=1)
    ax4.set_xlabel('胜率 (%)')
    ax4.set_ylabel('平均收益率 (%)')
    ax4.set_title('各股票表现 (气泡大小=交易次数)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    chart_file = 'reports/w_bottom_backtest_chart.png'
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {chart_file}")
    plt.show()


def main():
    """主函数"""
    print("=" * 60)
    print("W底形态策略回测")
    print("买入条件：识别到W底形态并突破颈线")
    print("卖出条件：持有2个月（约40个交易日）")
    print("=" * 60)
    
    # 参数设置
    num_stocks = 50
    start_date = "2015-01-01"
    end_date = "2024-12-30"
    holding_days = 40
    
    # 随机选择股票
    random.seed(42)
    selected_symbols = random.sample(NASDAQ_100_SYMBOLS, min(num_stocks, len(NASDAQ_100_SYMBOLS)))
    
    print(f"\n随机选择 {len(selected_symbols)} 只纳斯达克100股票进行回测:")
    print(f"股票列表: {selected_symbols}")
    print(f"回测期间: {start_date} 至 {end_date}")
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
            
            result = run_single_stock_backtest(data, symbol, holding_days=holding_days)
            
            all_results.append(result)
            all_trades.extend(result['trades'])
            
            if result['total_trades'] > 0:
                print(f"  W底信号: {result['total_trades']}, "
                      f"胜率: {result['win_rate']:.1f}%, "
                      f"平均收益: {result['avg_return']:.2f}%")
            else:
                print(f"  无W底信号")
                
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
    print(f"平均形态持续天数: {trades_df['pattern_days'].mean():.1f} 天")
    
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
    print(stock_stats.head(20).to_string())
    
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
                                         'exit_price', 'pnl_pct', 'days_held', 'pattern_days']]
    recent_trades = recent_trades.copy()
    recent_trades['pnl_pct'] = recent_trades['pnl_pct'].apply(lambda x: f"{x*100:.2f}%")
    recent_trades['entry_price'] = recent_trades['entry_price'].apply(lambda x: f"${x:.2f}")
    recent_trades['exit_price'] = recent_trades['exit_price'].apply(lambda x: f"${x:.2f}")
    print(recent_trades.to_string(index=False))
    
    # 保存结果
    output_file = 'reports/w_bottom_backtest_results.csv'
    os.makedirs('reports', exist_ok=True)
    trades_df.to_csv(output_file, index=False)
    print(f"\n详细结果已保存到: {output_file}")
    
    # 绘制图表
    plot_backtest_results(trades_df, holding_days)


if __name__ == "__main__":
    main()
