"""
多周期KDJ金叉策略回测
买入条件：
1. 日线KDJ金叉（K线上穿D线）
2. 周线KDJ金叉
3. 日均线呈上涨趋势（MA5 > MA10 > MA20）
卖出条件：持有3个月（约60个交易日）
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


def calculate_kdj(data: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """
    计算KDJ指标
    
    Args:
        data: OHLCV数据
        n: RSV周期（默认9）
        m1: K值平滑周期（默认3）
        m2: D值平滑周期（默认3）
    
    Returns:
        包含K, D, J值的DataFrame
    """
    df = data.copy()
    
    # 计算RSV
    low_min = df['Low'].rolling(window=n, min_periods=1).min()
    high_max = df['High'].rolling(window=n, min_periods=1).max()
    
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    rsv = rsv.fillna(50)  # 处理除零情况
    
    # 计算K值（RSV的m1日移动平均）
    df['K'] = rsv.ewm(alpha=1/m1, adjust=False).mean()
    
    # 计算D值（K的m2日移动平均）
    df['D'] = df['K'].ewm(alpha=1/m2, adjust=False).mean()
    
    # 计算J值
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    return df


def calculate_weekly_kdj(daily_data: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """
    计算周线KDJ指标
    
    Args:
        daily_data: 日线OHLCV数据
        
    Returns:
        周线KDJ数据，索引为周五日期
    """
    df = daily_data.copy()
    
    # 转换为周线数据
    weekly = df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    # 计算周线KDJ
    weekly = calculate_kdj(weekly, n, m1, m2)
    weekly = weekly.rename(columns={'K': 'K_weekly', 'D': 'D_weekly', 'J': 'J_weekly'})
    
    return weekly[['K_weekly', 'D_weekly', 'J_weekly']]


def calculate_ma(data: pd.DataFrame, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """计算移动平均线"""
    df = data.copy()
    for period in periods:
        df[f'MA{period}'] = df['Close'].rolling(window=period).mean()
    return df


def detect_kdj_golden_cross(k: pd.Series, d: pd.Series) -> pd.Series:
    """
    检测KDJ金叉信号
    金叉：K线从下向上穿过D线
    
    Returns:
        布尔Series，True表示当天出现金叉
    """
    # 前一天K<D，当天K>=D
    cross = (k.shift(1) < d.shift(1)) & (k >= d)
    return cross


def check_ma_uptrend(data: pd.DataFrame) -> pd.Series:
    """
    检查均线是否呈上涨趋势
    条件：MA5 > MA10 > MA20 且 MA5在上涨
    
    Returns:
        布尔Series
    """
    ma5_rising = data['MA5'] > data['MA5'].shift(1)
    ma_aligned = (data['MA5'] > data['MA10']) & (data['MA10'] > data['MA20'])
    
    return ma5_rising & ma_aligned


def run_single_stock_backtest(
    data: pd.DataFrame, 
    symbol: str,
    holding_days: int = 60,
) -> Dict:
    """
    对单只股票运行KDJ金叉策略回测
    """
    df = data.copy()
    
    # 计算日线KDJ
    df = calculate_kdj(df)
    
    # 计算均线
    df = calculate_ma(df, [5, 10, 20])
    
    # 计算周线KDJ
    weekly_kdj = calculate_weekly_kdj(data)
    
    # 将周线KDJ映射到日线（使用前向填充）
    df['K_weekly'] = np.nan
    df['D_weekly'] = np.nan
    
    for week_date in weekly_kdj.index:
        # 找到该周对应的日线数据
        mask = (df.index <= week_date) & (df.index > week_date - pd.Timedelta(days=7))
        df.loc[mask, 'K_weekly'] = weekly_kdj.loc[week_date, 'K_weekly']
        df.loc[mask, 'D_weekly'] = weekly_kdj.loc[week_date, 'D_weekly']
    
    # 前向填充周线数据
    df['K_weekly'] = df['K_weekly'].ffill()
    df['D_weekly'] = df['D_weekly'].ffill()
    
    # 检测日线KDJ金叉
    df['daily_golden_cross'] = detect_kdj_golden_cross(df['K'], df['D'])
    
    # 检测周线KDJ金叉（周线K上穿D）
    df['weekly_golden_cross'] = detect_kdj_golden_cross(df['K_weekly'], df['D_weekly'])
    
    # 周线金叉信号持续一周有效
    df['weekly_golden_cross_valid'] = df['weekly_golden_cross'].rolling(window=5, min_periods=1).max().astype(bool)
    
    # 检查均线上涨趋势
    df['ma_uptrend'] = check_ma_uptrend(df)
    
    # 综合买入信号：日线金叉 + 周线金叉有效 + 均线上涨趋势
    df['buy_signal'] = df['daily_golden_cross'] & df['weekly_golden_cross_valid'] & df['ma_uptrend']
    
    # 回测交易
    trades = []
    last_exit_idx = -1
    
    signal_dates = df[df['buy_signal']].index.tolist()
    
    for signal_date in signal_dates:
        entry_idx = df.index.get_loc(signal_date)
        
        # 避免重叠交易
        if entry_idx <= last_exit_idx:
            continue
        
        # 确保有足够的数据进行持有
        exit_idx = entry_idx + holding_days
        if exit_idx >= len(df):
            exit_idx = len(df) - 1
        
        entry_price = df.iloc[entry_idx]['Close']
        exit_price = df.iloc[exit_idx]['Close']
        entry_date = df.index[entry_idx]
        exit_date = df.index[exit_idx]
        
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
            'K_daily': df.iloc[entry_idx]['K'],
            'D_daily': df.iloc[entry_idx]['D'],
            'K_weekly': df.iloc[entry_idx]['K_weekly'],
            'D_weekly': df.iloc[entry_idx]['D_weekly'],
            'MA5': df.iloc[entry_idx]['MA5'],
            'MA10': df.iloc[entry_idx]['MA10'],
            'MA20': df.iloc[entry_idx]['MA20'],
        })
        
        last_exit_idx = exit_idx
    
    # 计算统计数据
    if not trades:
        return {
            'symbol': symbol,
            'total_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
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
    fig.suptitle(f'日线+周线KDJ金叉+均线趋势策略回测结果\n(持有{holding_days}天)', 
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
            ax4.annotate(idx, (row['胜率'], row['平均收益']), fontsize=7, alpha=0.8)
        
        plt.colorbar(scatter, ax=ax4, label='平均收益率(%)')
    
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax4.axvline(x=50, color='gray', linestyle='--', linewidth=1)
    ax4.set_xlabel('胜率 (%)')
    ax4.set_ylabel('平均收益率 (%)')
    ax4.set_title('各股票表现 (气泡大小=交易次数)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    chart_file = 'reports/kdj_cross_backtest_chart.png'
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {chart_file}")
    plt.show()


def main():
    """主函数"""
    print("=" * 70)
    print("日线+周线KDJ金叉+均线趋势策略回测")
    print("=" * 70)
    print("买入条件：")
    print("  1. 日线KDJ金叉（K线上穿D线）")
    print("  2. 周线KDJ金叉（近5天内出现）")
    print("  3. 日均线上涨趋势（MA5 > MA10 > MA20 且 MA5上涨）")
    print("卖出条件：持有3个月（约60个交易日）")
    print("=" * 70)
    
    # 参数设置
    num_stocks = 50
    start_date = "2015-01-01"
    end_date = "2024-12-30"
    holding_days = 60  # 3个月约60个交易日
    
    # 随机选择股票
    random.seed(42)
    selected_symbols = random.sample(NASDAQ_100_SYMBOLS, min(num_stocks, len(NASDAQ_100_SYMBOLS)))
    
    print(f"\n随机选择 {len(selected_symbols)} 只纳斯达克100股票进行回测:")
    print(f"股票列表: {selected_symbols}")
    print(f"回测期间: {start_date} 至 {end_date}")
    print(f"持有天数: {holding_days} 天（约3个月）")
    print("-" * 70)
    
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
                print(f"  信号次数: {result['total_trades']}, "
                      f"胜率: {result['win_rate']:.1f}%, "
                      f"平均收益: {result['avg_return']:.2f}%")
            else:
                print(f"  无买入信号")
                
        except Exception as e:
            print(f"  {symbol} 处理失败: {e}")
    
    # 汇总统计
    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)
    
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
    print("\n" + "-" * 70)
    print("各股票统计（前20）:")
    print("-" * 70)
    
    stock_stats = trades_df.groupby('symbol').agg({
        'pnl_pct': ['count', 'mean', lambda x: (x > 0).sum() / len(x) * 100],
    }).round(2)
    stock_stats.columns = ['交易次数', '平均收益%', '胜率%']
    stock_stats['平均收益%'] = stock_stats['平均收益%'] * 100
    stock_stats = stock_stats.sort_values('交易次数', ascending=False)
    print(stock_stats.head(20).to_string())
    
    # 按年份统计
    print("\n" + "-" * 70)
    print("按年份统计:")
    print("-" * 70)
    
    trades_df['year'] = pd.to_datetime(trades_df['entry_date']).dt.year
    yearly_stats = trades_df.groupby('year').agg({
        'pnl_pct': ['count', 'mean', lambda x: (x > 0).sum() / len(x) * 100],
    }).round(2)
    yearly_stats.columns = ['交易次数', '平均收益%', '胜率%']
    yearly_stats['平均收益%'] = yearly_stats['平均收益%'] * 100
    print(yearly_stats.to_string())
    
    # 显示部分交易详情
    print("\n" + "-" * 70)
    print("最近10笔交易详情:")
    print("-" * 70)
    
    recent_trades = trades_df.tail(10)[['symbol', 'entry_date', 'exit_date', 'entry_price', 
                                         'exit_price', 'pnl_pct', 'days_held']]
    recent_trades = recent_trades.copy()
    recent_trades['pnl_pct'] = recent_trades['pnl_pct'].apply(lambda x: f"{x*100:.2f}%")
    recent_trades['entry_price'] = recent_trades['entry_price'].apply(lambda x: f"${x:.2f}")
    recent_trades['exit_price'] = recent_trades['exit_price'].apply(lambda x: f"${x:.2f}")
    print(recent_trades.to_string(index=False))
    
    # 保存结果
    output_file = 'reports/kdj_cross_backtest_results.csv'
    os.makedirs('reports', exist_ok=True)
    trades_df.to_csv(output_file, index=False)
    print(f"\n详细结果已保存到: {output_file}")
    
    # 绘制图表
    plot_backtest_results(trades_df, holding_days)


if __name__ == "__main__":
    main()
