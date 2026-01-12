"""
Top 10 因子卖出策略回测

基于前后两周最高点分析的Top 10高频因子组合策略
"""

import sys
sys.path.append('src')

from strategies.top10_sell_strategy import find_sell_points, Top10SellStrategy
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_sell_points(symbol: str, min_factors: int = 7, 
                       start_date: str = None, end_date: str = None,
                       years: int = 15):
    """
    分析股票卖点
    
    Args:
        symbol: 股票代码
        min_factors: 最少满足因子数（默认7，范围1-10）
        start_date: 开始日期
        end_date: 结束日期
        years: 分析年数（当start_date为None时使用）
    """
    
    # 设置日期范围
    if start_date is None:
        end_date = datetime.now() if end_date is None else datetime.strptime(end_date, '%Y-%m-%d')
        start_date = (end_date - timedelta(days=years * 365)).strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
    
    print(f"\n🔍 分析 {symbol} - Top 10因子卖出策略")
    print(f"📅 时间范围: {start_date} ~ {end_date}")
    print(f"⚙️  最少因子数: {min_factors}/10")
    print("=" * 50)
    
    try:
        # 下载数据
        data = yf.download(symbol, start=start_date, end=end_date, 
                          progress=False, auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if data.empty or len(data) < 100:
            print(f"❌ 数据不足，跳过分析")
            return None
        
        # 初始化策略
        strategy = Top10SellStrategy(min_factors=min_factors)
        
        # 计算指标和因子
        df = strategy.calculate_indicators(data.copy())
        df = strategy.calculate_factors(df)
        
        # 识别卖点
        sell_points = strategy.identify_sell_points(df)
        
        if not sell_points:
            print(f"❌ 未找到满足条件的卖点")
            return None
        
        # 统计分析
        print(f"📊 总卖点数: {len(sell_points)}")
        
        # 按年份统计
        yearly_stats = {}
        for point in sell_points:
            year = point['date'].year
            yearly_stats[year] = yearly_stats.get(year, 0) + 1
        
        print(f"\n📈 按年份分布:")
        for year in sorted(yearly_stats.keys()):
            print(f"  {year}: {yearly_stats[year]} 个")
        
        # 因子统计
        factor_names = [
            'Price_above_SMA10', 'Price_above_SMA20', 'Price_above_SMA5',
            'Aroon_Up_80', 'Aroon_Up_90', 'MACD_Histogram_positive',
            'Price_above_SMA50', 'MACD_positive', 'Williams_overbought', 'Stoch_overbought'
        ]
        
        factor_counts = {name: 0 for name in factor_names}
        for point in sell_points:
            for i, name in enumerate(factor_names, 1):
                if point['factors'][f'F{i}_{name}']:
                    factor_counts[name] += 1
        
        print(f"\n🎯 因子出现频率:")
        total_points = len(sell_points)
        sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
        for name, count in sorted_factors:
            print(f"  {name}: {count}/{total_points} = {count/total_points*100:.1f}%")
        
        # 最近卖点
        print(f"\n🔴 最近卖点:")
        recent_points = sell_points[-10:] if len(sell_points) >= 10 else sell_points
        for point in recent_points:
            print(f"  {point['date'].strftime('%Y-%m-%d')}: ${point['price']:.2f} "
                  f"({point['factor_count']}/10因子)")
        
        return sell_points, df
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return None


def plot_sell_points(symbol: str, df: pd.DataFrame, sell_points: list, 
                    min_factors: int = 7, save_path: str = None):
    """
    绘制卖点图表
    """
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 上图：价格和均线
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], label='Close Price', linewidth=1.5, color='black')
    ax1.plot(df.index, df['SMA_5'], label='SMA 5', alpha=0.7, color='orange')
    ax1.plot(df.index, df['SMA_10'], label='SMA 10', alpha=0.7, color='blue')
    ax1.plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.7, color='green')
    ax1.plot(df.index, df['SMA_50'], label='SMA 50', alpha=0.7, color='red')
    
    # 标记卖点
    if sell_points:
        sell_dates = [p['date'] for p in sell_points]
        sell_prices = [p['price'] for p in sell_points]
        ax1.scatter(sell_dates, sell_prices, color='red', s=60, 
                   label=f'Sell Points ({len(sell_points)})', zorder=5, marker='v')
    
    ax1.set_title(f'{symbol} - Top 10 Factor Sell Strategy (min_factors={min_factors}/10)')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 中图：因子计数
    ax2 = axes[1]
    ax2.plot(df.index, df['factor_count'], label='Factor Count', color='blue', linewidth=1)
    ax2.axhline(y=min_factors, color='red', linestyle='--', 
               label=f'Threshold ({min_factors})', linewidth=2)
    ax2.fill_between(df.index, df['factor_count'], min_factors, 
                    where=(df['factor_count'] >= min_factors), 
                    color='red', alpha=0.3, label='Sell Zone')
    
    ax2.set_ylabel('Factor Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 10)
    
    # 下图：主要技术指标
    ax3 = axes[2]
    
    # Aroon Up
    aroon_up_col = [c for c in df.columns if 'AROONU' in c][0]
    ax3_twin = ax3.twinx()
    
    ax3.plot(df.index, df[aroon_up_col], label='Aroon Up', color='green', alpha=0.7)
    ax3.axhline(y=80, color='green', linestyle=':', alpha=0.5)
    ax3.axhline(y=90, color='green', linestyle=':', alpha=0.5)
    
    # Williams %R
    willr_col = [c for c in df.columns if 'WILLR' in c][0]
    ax3_twin.plot(df.index, df[willr_col], label='Williams %R', color='purple', alpha=0.7)
    ax3_twin.axhline(y=-20, color='purple', linestyle=':', alpha=0.5)
    
    ax3.set_ylabel('Aroon Up')
    ax3_twin.set_ylabel('Williams %R')
    ax3.set_xlabel('Date')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 图表已保存: {save_path}")
    
    plt.show()


def backtest_strategy(symbol: str, min_factors: int = 7, years: int = 15):
    """
    简单回测策略效果
    """
    print(f"\n🔄 回测 {symbol} - Top 10因子策略")
    
    result = analyze_sell_points(symbol, min_factors, years=years)
    if not result:
        return
    
    sell_points, df = result
    
    # 计算卖点后的收益
    returns_1d = []
    returns_5d = []
    returns_10d = []
    
    for point in sell_points:
        sell_date = point['date']
        sell_price = point['price']
        
        # 找到卖点后的价格
        future_data = df[df.index > sell_date]
        
        if len(future_data) >= 1:
            price_1d = future_data.iloc[0]['Close']
            returns_1d.append((price_1d - sell_price) / sell_price * 100)
        
        if len(future_data) >= 5:
            price_5d = future_data.iloc[4]['Close']
            returns_5d.append((price_5d - sell_price) / sell_price * 100)
        
        if len(future_data) >= 10:
            price_10d = future_data.iloc[9]['Close']
            returns_10d.append((price_10d - sell_price) / sell_price * 100)
    
    # 统计结果
    print(f"\n📊 卖点后收益统计:")
    
    if returns_1d:
        avg_1d = np.mean(returns_1d)
        win_rate_1d = sum(1 for r in returns_1d if r < 0) / len(returns_1d) * 100
        print(f"  1天后: 平均收益 {avg_1d:.2f}%, 胜率 {win_rate_1d:.1f}% ({len(returns_1d)}个样本)")
    
    if returns_5d:
        avg_5d = np.mean(returns_5d)
        win_rate_5d = sum(1 for r in returns_5d if r < 0) / len(returns_5d) * 100
        print(f"  5天后: 平均收益 {avg_5d:.2f}%, 胜率 {win_rate_5d:.1f}% ({len(returns_5d)}个样本)")
    
    if returns_10d:
        avg_10d = np.mean(returns_10d)
        win_rate_10d = sum(1 for r in returns_10d if r < 0) / len(returns_10d) * 100
        print(f"  10天后: 平均收益 {avg_10d:.2f}%, 胜率 {win_rate_10d:.1f}% ({len(returns_10d)}个样本)")


if __name__ == '__main__':
    # 获取股票代码
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    min_factors = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    
    # 分析卖点
    result = analyze_sell_points(
        symbol=symbol,
        min_factors=min_factors,
        years=15
    )
    
    if result:
        sell_points, df = result
        plot_sell_points(symbol, df, sell_points, min_factors=min_factors,
                        save_path=f'reports/{symbol.lower()}_top10_sell_points.png')
        
        # 简单回测
        backtest_strategy(symbol, min_factors, years=15)