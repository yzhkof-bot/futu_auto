"""
TSLA 多信号组合策略回测
买入条件：至少3个信号同时出现
1. 均线空头排列 + RSI超卖(<30)
2. RSI超卖(<30) + KDJ超卖(K,D<20)
3. 接近60日低点 + KDJ超卖
4. MACD零下金叉

卖出条件：持有20个交易日（约1个月）
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
import warnings
warnings.filterwarnings('ignore')


def load_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """加载股票数据"""
    cache_dir = '.cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{symbol}_{start_date}_{end_date}.csv')
    
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if len(df) > 0:
            return df
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        if len(df) > 0:
            df.to_csv(cache_file)
        return df
    except Exception as e:
        print(f"获取数据失败: {e}")
        return pd.DataFrame()


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标"""
    if len(df) < 60:
        return df
    
    # 均线
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    # 均线空头排列
    df['ma_trend_down'] = ((df['MA5'] < df['MA10']) & (df['MA10'] < df['MA20'])).astype(int)
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_oversold'] = (df['RSI'] < 30).astype(int)
    
    # KDJ
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    df['KDJ_oversold'] = ((df['K'] < 20) & (df['D'] < 20)).astype(int)
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    df['MACD_below_zero'] = (df['MACD'] < 0).astype(int)
    df['MACD_golden_cross'] = ((df['MACD'] > df['MACD_signal']) & 
                               (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))).astype(int)
    
    # 布林带
    df['BB_mid'] = df['Close'].rolling(20).mean()
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
    
    # 价格位置
    df['low_60d'] = df['Low'].rolling(60).min()
    df['high_60d'] = df['High'].rolling(60).max()
    df['price_position_60d'] = (df['Close'] - df['low_60d']) / (df['high_60d'] - df['low_60d'])
    df['near_60d_low'] = (df['price_position_60d'] < 0.1).astype(int)
    
    return df


def count_signals(row) -> Tuple[int, List[str]]:
    """统计满足的信号数量"""
    signals = []
    
    # 信号1: 均线空头 + RSI超卖
    if row['ma_trend_down'] == 1 and row['RSI_oversold'] == 1:
        signals.append('均线空头+RSI超卖')
    
    # 信号2: RSI超卖 + KDJ超卖
    if row['RSI_oversold'] == 1 and row['KDJ_oversold'] == 1:
        signals.append('RSI+KDJ超卖')
    
    # 信号3: 接近60日低点 + KDJ超卖
    if row['near_60d_low'] == 1 and row['KDJ_oversold'] == 1:
        signals.append('60日低点+KDJ超卖')
    
    # 信号4: MACD零下金叉
    if row['MACD_below_zero'] == 1 and row['MACD_golden_cross'] == 1:
        signals.append('MACD零下金叉')
    
    return len(signals), signals


def backtest_strategy(df: pd.DataFrame, holding_days: int = 20, min_signals: int = 3) -> List[Dict]:
    """回测策略"""
    trades = []
    last_exit_idx = -1
    
    for i in range(60, len(df) - holding_days):
        if i <= last_exit_idx:
            continue
        
        row = df.iloc[i]
        signal_count, triggered_signals = count_signals(row)
        
        if signal_count < min_signals:
            continue
        
        # 记录交易
        entry_idx = i
        exit_idx = min(i + holding_days, len(df) - 1)
        
        entry_price = df.iloc[entry_idx]['Close']
        exit_price = df.iloc[exit_idx]['Close']
        entry_date = df.index[entry_idx]
        exit_date = df.index[exit_idx]
        
        pnl_pct = (exit_price - entry_price) / entry_price
        
        trades.append({
            'entry_idx': entry_idx,
            'exit_idx': exit_idx,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'signal_count': signal_count,
            'signals': triggered_signals,
            'RSI': row['RSI'],
            'K': row['K'],
        })
        
        last_exit_idx = exit_idx
    
    return trades


def plot_strategy(df: pd.DataFrame, trades: List[Dict], symbol: str, 
                  start_date: str, end_date: str):
    """绘制策略图表"""
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), 
                             gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    fig.suptitle(f'{symbol} 多信号组合策略回测 ({start_date} ~ {end_date})\n'
                 f'买入条件: ≥3个信号同时出现 | 持有: 20个交易日', 
                 fontsize=14, fontweight='bold')
    
    # 1. 主图：K线 + 均线 + 布林带 + 买卖点
    ax1 = axes[0]
    
    # 绘制收盘价
    ax1.plot(df.index, df['Close'], color='black', linewidth=1, label='收盘价')
    
    # 绘制均线
    ax1.plot(df.index, df['MA20'], color='blue', linewidth=0.8, alpha=0.7, label='MA20')
    ax1.plot(df.index, df['MA60'], color='purple', linewidth=0.8, alpha=0.7, label='MA60')
    
    # 绘制布林带
    ax1.fill_between(df.index, df['BB_lower'], df['BB_upper'], 
                     color='gray', alpha=0.1, label='布林带')
    ax1.plot(df.index, df['BB_upper'], color='gray', linewidth=0.5, linestyle='--')
    ax1.plot(df.index, df['BB_lower'], color='gray', linewidth=0.5, linestyle='--')
    
    # 标注买卖点
    for trade in trades:
        entry_date = trade['entry_date']
        exit_date = trade['exit_date']
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        pnl = trade['pnl_pct']
        
        # 买入点（绿色向上三角）
        ax1.scatter(entry_date, entry_price, marker='^', s=150, 
                   color='green', zorder=5, edgecolors='black', linewidths=1)
        
        # 卖出点（红色向下三角）
        color = 'red' if pnl < 0 else 'darkgreen'
        ax1.scatter(exit_date, exit_price, marker='v', s=150, 
                   color=color, zorder=5, edgecolors='black', linewidths=1)
        
        # 连接买卖点
        ax1.plot([entry_date, exit_date], [entry_price, exit_price], 
                color=color, linestyle='--', linewidth=1, alpha=0.5)
        
        # 标注收益
        mid_date = entry_date + (exit_date - entry_date) / 2
        mid_price = (entry_price + exit_price) / 2
        ax1.annotate(f'{pnl*100:+.1f}%', (mid_date, mid_price * 1.02),
                    fontsize=8, ha='center', color=color, fontweight='bold')
    
    ax1.set_ylabel('价格 ($)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'价格走势与买卖点 (共{len(trades)}笔交易)')
    
    # 2. RSI
    ax2 = axes[1]
    ax2.plot(df.index, df['RSI'], color='purple', linewidth=1)
    ax2.axhline(y=30, color='green', linestyle='--', linewidth=1, label='超卖(30)')
    ax2.axhline(y=70, color='red', linestyle='--', linewidth=1, label='超买(70)')
    ax2.fill_between(df.index, 0, 30, color='green', alpha=0.1)
    ax2.fill_between(df.index, 70, 100, color='red', alpha=0.1)
    
    # 标注买点时的RSI
    for trade in trades:
        ax2.scatter(trade['entry_date'], trade['RSI'], marker='^', s=80, 
                   color='green', zorder=5, edgecolors='black')
    
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. KDJ
    ax3 = axes[2]
    ax3.plot(df.index, df['K'], color='blue', linewidth=1, label='K')
    ax3.plot(df.index, df['D'], color='orange', linewidth=1, label='D')
    ax3.axhline(y=20, color='green', linestyle='--', linewidth=1)
    ax3.axhline(y=80, color='red', linestyle='--', linewidth=1)
    ax3.fill_between(df.index, 0, 20, color='green', alpha=0.1)
    
    # 标注买点时的K值
    for trade in trades:
        ax3.scatter(trade['entry_date'], trade['K'], marker='^', s=80, 
                   color='green', zorder=5, edgecolors='black')
    
    ax3.set_ylabel('KDJ')
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. MACD
    ax4 = axes[3]
    ax4.plot(df.index, df['MACD'], color='blue', linewidth=1, label='MACD')
    ax4.plot(df.index, df['MACD_signal'], color='orange', linewidth=1, label='Signal')
    
    # MACD柱状图
    colors = ['green' if v >= 0 else 'red' for v in df['MACD_hist']]
    ax4.bar(df.index, df['MACD_hist'], color=colors, alpha=0.5, width=1)
    ax4.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    # 标注买点
    for trade in trades:
        entry_idx = trade['entry_idx']
        ax4.scatter(trade['entry_date'], df.iloc[entry_idx]['MACD'], 
                   marker='^', s=80, color='green', zorder=5, edgecolors='black')
    
    ax4.set_ylabel('MACD')
    ax4.set_xlabel('日期')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 格式化x轴日期
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 保存图表
    chart_file = f'reports/{symbol}_strategy_backtest.png'
    os.makedirs('reports', exist_ok=True)
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {chart_file}")
    plt.close()


def main():
    """主函数"""
    print("=" * 70)
    print("TSLA 多信号组合策略回测")
    print("=" * 70)
    print("\n买入条件（至少3个信号同时出现）：")
    print("  1. 均线空头排列 + RSI超卖(<30)")
    print("  2. RSI超卖(<30) + KDJ超卖(K,D<20)")
    print("  3. 接近60日低点 + KDJ超卖")
    print("  4. MACD零下金叉")
    print("\n卖出条件：持有20个交易日")
    print("=" * 70)
    
    symbol = "TSLA"
    holding_days = 20
    min_signals = 3
    
    # 使用2021-2024区间，TSLA波动较大，信号更多
    start_year = 2021
    
    start_date = f"{start_year}-01-01"
    end_date = f"{start_year + 3}-12-31"
    
    print(f"\n随机选择回测区间: {start_date} ~ {end_date}")
    print("-" * 70)
    
    # 加载数据
    print(f"\n加载 {symbol} 数据...")
    df = load_stock_data(symbol, start_date, end_date)
    
    if len(df) < 100:
        print("数据不足，无法回测")
        return
    
    print(f"加载完成，共 {len(df)} 个交易日")
    
    # 计算指标
    df = calculate_indicators(df)
    
    # 回测
    trades = backtest_strategy(df, holding_days, min_signals)
    
    print(f"\n找到 {len(trades)} 个买入信号")
    
    if len(trades) == 0:
        print("该区间内没有符合条件的交易信号")
        print("尝试其他时间区间...")
        
        # 尝试其他区间
        for try_year in [2020, 2021, 2022, 2018, 2019]:
            start_date = f"{try_year}-01-01"
            end_date = f"{try_year + 3}-01-01"
            print(f"\n尝试区间: {start_date} ~ {end_date}")
            
            df = load_stock_data(symbol, start_date, end_date)
            if len(df) < 100:
                continue
            
            df = calculate_indicators(df)
            trades = backtest_strategy(df, holding_days, min_signals)
            
            if len(trades) > 0:
                print(f"找到 {len(trades)} 个信号")
                break
    
    if len(trades) == 0:
        print("所有区间都没有找到符合条件的信号")
        return
    
    # 统计结果
    trades_df = pd.DataFrame(trades)
    winning = len(trades_df[trades_df['pnl_pct'] > 0])
    
    print("\n" + "=" * 70)
    print("回测结果")
    print("=" * 70)
    print(f"\n回测区间: {start_date} ~ {end_date}")
    print(f"总交易次数: {len(trades)}")
    print(f"盈利交易: {winning}")
    print(f"亏损交易: {len(trades) - winning}")
    print(f"胜率: {winning/len(trades)*100:.2f}%")
    print(f"\n平均收益: {trades_df['pnl_pct'].mean()*100:.2f}%")
    print(f"中位数收益: {trades_df['pnl_pct'].median()*100:.2f}%")
    print(f"最大收益: {trades_df['pnl_pct'].max()*100:.2f}%")
    print(f"最小收益: {trades_df['pnl_pct'].min()*100:.2f}%")
    print(f"总收益: {trades_df['pnl_pct'].sum()*100:.2f}%")
    
    print("\n" + "-" * 70)
    print("交易详情:")
    print("-" * 70)
    
    for i, trade in enumerate(trades, 1):
        entry_date = trade['entry_date'].strftime('%Y-%m-%d')
        exit_date = trade['exit_date'].strftime('%Y-%m-%d')
        pnl = trade['pnl_pct'] * 100
        signals = ', '.join(trade['signals'])
        result = "✓" if pnl > 0 else "✗"
        print(f"{i}. {entry_date} → {exit_date} | "
              f"${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} | "
              f"{pnl:+.2f}% {result}")
        print(f"   信号: {signals}")
    
    # 绘制图表
    print("\n生成图表...")
    plot_strategy(df, trades, symbol, start_date, end_date)
    
    # 保存交易记录
    output_file = f'reports/{symbol}_trades.csv'
    trades_df.to_csv(output_file, index=False)
    print(f"交易记录已保存到: {output_file}")


if __name__ == "__main__":
    main()
