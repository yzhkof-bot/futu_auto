"""
Top 5 卖出因子策略回测

基于卖点分析结果，选取出现频率最高的5个因子：
1. MACD正值 (90.06%) - 多头趋势中见顶
2. 布林带上轨突破 (73.11%) - 价格接近布林带上轨
3. 价格接近50日高点 (60.03%) - 价格位置 > 90%
4. KDJ超买 (48.41%) - K值 > 80
5. RSI超买 (47.46%) - RSI > 70

功能：显示所有卖点并生成图表
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.strategies.top5_sell_strategy import Top5SellStrategy

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 纳斯达克100成分股
NASDAQ100 = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'COST',
    'NFLX', 'TMUS', 'ASML', 'CSCO', 'ADBE', 'AMD', 'PEP', 'LIN', 'INTC', 'INTU',
    'TXN', 'CMCSA', 'QCOM', 'AMGN', 'AMAT', 'HON', 'ISRG', 'BKNG', 'SBUX', 'VRTX',
    'GILD', 'ADP', 'MDLZ', 'ADI', 'REGN', 'PANW', 'SNPS', 'LRCX', 'KLAC', 'CDNS',
    'MU', 'MELI', 'PYPL', 'MAR', 'ORLY', 'MNST', 'CTAS', 'NXPI', 'MCHP', 'FTNT',
    'ABNB', 'PCAR', 'KDP', 'AEP', 'PAYX', 'KHC', 'ODFL', 'CPRT', 'CHTR', 'ROST',
    'IDXX', 'DXCM', 'FAST', 'AZN', 'MRNA', 'EA', 'CTSH', 'EXC', 'VRSK', 'CSGP',
    'XEL', 'BKR', 'GEHC', 'FANG', 'TTWO', 'BIIB', 'ON', 'DLTR', 'WBD',
    'CDW', 'ZS', 'ILMN', 'MDB', 'DDOG', 'GFS', 'LCID', 'SIRI',
    'CEG', 'CRWD', 'DASH', 'SMCI', 'ARM', 'COIN', 'TTD', 'PDD', 'LULU', 'WDAY'
]


def analyze_sell_points(symbol: str, min_factors: int = 5, 
                       start_date: str = None, end_date: str = None,
                       years: int = 15):
    """
    分析股票卖点
    
    Args:
        symbol: 股票代码
        min_factors: 最少满足因子数（默认5，即全部因子同时满足）
        start_date: 开始日期
        end_date: 结束日期
        years: 分析年数（当start_date为None时使用）
    """
    print("=" * 70)
    print(f"Top 5 卖出因子策略分析: {symbol}")
    print("=" * 70)
    print(f"因子要求: >= {min_factors} 个")
    print("=" * 70)
    
    # 日期范围
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
    
    print(f"分析区间: {start_date} ~ {end_date}")
    
    # 获取数据并计算因子
    strategy = Top5SellStrategy(min_factors=min_factors)
    df = strategy.fetch_and_calculate(symbol, start_date, end_date)
    
    if df.empty:
        print("数据获取失败")
        return None
    
    print(f"数据天数: {len(df)}")
    
    # 获取卖点
    sell_points = strategy.find_sell_points(symbol, start_date, end_date, data=df)
    
    print("-" * 70)
    print(f"卖出因子 (按出现频率排序):")
    for name, desc in strategy.FACTOR_DESCRIPTIONS.items():
        freq = strategy.FACTOR_FREQUENCIES.get(name, 0)
        print(f"  {freq:.1%} - {desc}")
    print("-" * 70)
    
    # 输出卖点
    if not sell_points:
        print("\n无卖点信号")
        return None
    
    print(f"\n卖点记录 ({len(sell_points)} 个):")
    print("-" * 70)
    print(f"{'日期':<12} {'价格':>10} {'因子数':>6} {'满足因子':<40}")
    print("-" * 70)
    
    for sp in sell_points:
        date_str = str(sp['date'])[:10]
        factors_str = ', '.join(sp['factors'])
        print(f"{date_str:<12} ${sp['price']:>9.2f} {sp['factor_count']:>6} {factors_str:<40}")
    
    # 统计
    print("\n" + "=" * 70)
    print("统计信息")
    print("=" * 70)
    print(f"总卖点数: {len(sell_points)}")
    
    # 按因子数统计
    factor_counts = {}
    for sp in sell_points:
        count = sp['factor_count']
        factor_counts[count] = factor_counts.get(count, 0) + 1
    
    print(f"\n按因子数分布:")
    for count in sorted(factor_counts.keys(), reverse=True):
        num = factor_counts[count]
        print(f"  {count} 个因子: {num} 次 ({num/len(sell_points)*100:.1f}%)")
    
    # 按年份统计
    year_counts = {}
    for sp in sell_points:
        year = sp['date'].year
        year_counts[year] = year_counts.get(year, 0) + 1
    
    print(f"\n按年份分布:")
    for year in sorted(year_counts.keys()):
        num = year_counts[year]
        print(f"  {year}: {num} 次")
    
    return sell_points, df


def plot_sell_points(symbol: str, df: pd.DataFrame, sell_points: list, 
                    min_factors: int = 5, save_path: str = None):
    """
    绘制卖点图表
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), 
                             gridspec_kw={'height_ratios': [3, 1, 1, 1]},
                             sharex=True)
    
    fig.suptitle(f'{symbol} Top 5 卖出因子策略 (满足 >= {min_factors} 个因子) - 共 {len(sell_points)} 个卖点', 
                 fontsize=14, fontweight='bold')
    
    # 图1: 价格图
    ax1 = axes[0]
    close_data = df['Close'].dropna()
    ax1.plot(close_data.index, close_data.values, label='收盘价', color='steelblue', linewidth=1)
    
    if 'SMA_20' in df.columns:
        sma20 = df['SMA_20'].dropna()
        ax1.plot(sma20.index, sma20.values, label='MA20', color='orange', linewidth=0.8, alpha=0.7)
    
    if 'SMA_50' in df.columns:
        sma50 = df['SMA_50'].dropna()
        ax1.plot(sma50.index, sma50.values, label='MA50', color='green', linewidth=0.8, alpha=0.7)
    
    # 绘制布林带
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        bb_upper = df['BB_Upper'].dropna()
        bb_lower = df['BB_Lower'].dropna()
        ax1.fill_between(bb_upper.index, bb_lower.values, bb_upper.values, 
                        alpha=0.1, color='gray', label='布林带')
    
    # 标注卖点信号
    if sell_points:
        sell_dates = [sp['date'] for sp in sell_points]
        sell_prices = [sp['price'] for sp in sell_points]
        ax1.scatter(sell_dates, sell_prices, marker='v', color='red', 
                    s=100, label=f'卖出信号 ({len(sell_points)})', zorder=5)
    
    ax1.set_ylabel('价格 ($)')
    ax1.set_title('价格走势与卖出信号', loc='left')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 图2: RSI
    ax2 = axes[1]
    if 'RSI' in df.columns:
        rsi = df['RSI'].dropna()
        ax2.plot(rsi.index, rsi.values, label='RSI(14)', color='purple', linewidth=1)
        ax2.axhline(70, color='red', linestyle='--', linewidth=0.8, label='超买线(70)')
        ax2.axhline(30, color='green', linestyle='--', linewidth=0.8)
        ax2.fill_between(rsi.index, 70, rsi.values, where=(rsi.values > 70), 
                        alpha=0.3, color='red')
        
        # 标记卖点
        for sp in sell_points:
            if sp['date'] in df.index:
                ax2.axvline(sp['date'], color='red', alpha=0.3, linewidth=0.5)
    
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('RSI指标 (超买 > 70)', loc='left')
    
    # 图3: KDJ
    ax3 = axes[2]
    if 'KDJ_K' in df.columns:
        kdj_k = df['KDJ_K'].dropna()
        kdj_d = df['KDJ_D'].dropna()
        ax3.plot(kdj_k.index, kdj_k.values, label='K', color='blue', linewidth=1)
        ax3.plot(kdj_d.index, kdj_d.values, label='D', color='orange', linewidth=1)
        ax3.axhline(80, color='red', linestyle='--', linewidth=0.8, label='超买线(80)')
        ax3.axhline(20, color='green', linestyle='--', linewidth=0.8)
        ax3.fill_between(kdj_k.index, 80, kdj_k.values, where=(kdj_k.values > 80), 
                        alpha=0.3, color='red')
        
        # 标记卖点
        for sp in sell_points:
            if sp['date'] in df.index:
                ax3.axvline(sp['date'], color='red', alpha=0.3, linewidth=0.5)
    
    ax3.set_ylabel('KDJ')
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('KDJ指标 (超买 K > 80)', loc='left')
    
    # 图4: MACD
    ax4 = axes[3]
    if 'MACD' in df.columns:
        macd = df['MACD'].dropna()
        macd_signal = df['MACD_Signal'].dropna()
        ax4.plot(macd.index, macd.values, label='MACD', color='blue', linewidth=1)
        ax4.plot(macd_signal.index, macd_signal.values, label='Signal', color='orange', linewidth=1)
        
        # 柱状图
        macd_hist = df['MACD_Histogram'].dropna()
        colors = ['green' if x >= 0 else 'red' for x in macd_hist.values]
        ax4.bar(macd_hist.index, macd_hist.values, color=colors, alpha=0.5, width=1)
        ax4.axhline(0, color='black', linewidth=0.5)
        
        # 标记卖点
        for sp in sell_points:
            if sp['date'] in df.index:
                ax4.axvline(sp['date'], color='red', alpha=0.3, linewidth=0.5)
    
    ax4.set_ylabel('MACD')
    ax4.set_xlabel('日期')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('MACD指标 (卖点时通常 MACD > 0)', loc='left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存至 {save_path}")
    
    plt.show()
    return fig


if __name__ == '__main__':
    import sys
    
    # 支持命令行指定股票: python backtest_sell_strategy.py TSLA
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
        print(f"\n指定股票: {symbol}\n")
    else:
        symbol = random.choice(NASDAQ100)
        print(f"\n随机选择股票: {symbol}\n")
    
    result = analyze_sell_points(
        symbol=symbol,
        min_factors=5,
        years=15
    )
    
    if result:
        sell_points, df = result
        plot_sell_points(symbol, df, sell_points, min_factors=5,
                        save_path=f'reports/{symbol.lower()}_sell_points.png')