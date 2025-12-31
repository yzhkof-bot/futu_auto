"""
Top 5 因子策略回测

规则：
- 买入：出现买点信号，买入 $5000
- 止盈：+10%
- 止损：-10%
- 仓位：同时只能持有一笔 $5000
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.strategies.top5_factor_strategy import Top5FactorStrategy

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


def backtest(symbol: str, min_factors: int = 5, 
             position_size: float = 5000,
             take_profit: float = 0.10,
             stop_loss: float = 0.10,
             start_date: str = None,
             end_date: str = None):
    """
    回测 Top 5 因子策略
    
    Args:
        symbol: 股票代码
        min_factors: 最少满足因子数
        position_size: 每笔仓位金额
        take_profit: 止盈比例
        stop_loss: 止损比例
        start_date: 开始日期
        end_date: 结束日期
    """
    print("=" * 70)
    print(f"Top 5 因子策略回测: {symbol}")
    print("=" * 70)
    print(f"仓位: ${position_size:.0f} | 止盈: +{take_profit*100:.0f}% | 止损: -{stop_loss*100:.0f}%")
    print(f"因子要求: >= {min_factors} 个")
    print("=" * 70)
    
    # 日期范围
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    print(f"回测区间: {start_date} ~ {end_date}")
    
    # 获取数据并计算因子
    strategy = Top5FactorStrategy(min_factors=min_factors)
    df = strategy.fetch_and_calculate(symbol, start_date, end_date)
    
    if df.empty:
        print("数据获取失败")
        return None
    
    print(f"数据天数: {len(df)}")
    print("-" * 70)
    
    # 回测
    trades = []
    position = None  # 当前持仓 {'date', 'price', 'shares', 'cost'}
    
    for i in range(len(df)):
        date = df.index[i]
        row = df.iloc[i]
        close = row['Close']
        high = row['High']
        low = row['Low']
        
        # 如果有持仓，检查止盈止损
        if position is not None:
            entry_price = position['price']
            
            # 检查止盈（用最高价）
            if high >= entry_price * (1 + take_profit):
                sell_price = entry_price * (1 + take_profit)
                pnl = (sell_price - entry_price) * position['shares']
                pnl_pct = take_profit
                
                trades.append({
                    'buy_date': position['date'],
                    'buy_price': entry_price,
                    'sell_date': date,
                    'sell_price': sell_price,
                    'shares': position['shares'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_type': '止盈'
                })
                position = None
                continue
            
            # 检查止损（用最低价）
            if low <= entry_price * (1 - stop_loss):
                sell_price = entry_price * (1 - stop_loss)
                pnl = (sell_price - entry_price) * position['shares']
                pnl_pct = -stop_loss
                
                trades.append({
                    'buy_date': position['date'],
                    'buy_price': entry_price,
                    'sell_date': date,
                    'sell_price': sell_price,
                    'shares': position['shares'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_type': '止损'
                })
                position = None
                continue
        
        # 如果没有持仓且有买入信号
        if position is None and row.get('Buy_Signal', False):
            shares = position_size / close
            position = {
                'date': date,
                'price': close,
                'shares': shares,
                'cost': position_size
            }
    
    # 如果最后还有持仓，按收盘价平仓
    if position is not None:
        last_close = df['Close'].iloc[-1]
        pnl = (last_close - position['price']) * position['shares']
        pnl_pct = (last_close - position['price']) / position['price']
        
        trades.append({
            'buy_date': position['date'],
            'buy_price': position['price'],
            'sell_date': df.index[-1],
            'sell_price': last_close,
            'shares': position['shares'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_type': '持仓中'
        })
    
    # 输出交易记录
    if not trades:
        print("\n无交易记录")
        return None
    
    print(f"\n交易记录 ({len(trades)} 笔):")
    print("-" * 70)
    print(f"{'买入日期':<12} {'买入价':>8} {'卖出日期':<12} {'卖出价':>8} {'盈亏':>10} {'收益率':>8} {'类型'}")
    print("-" * 70)
    
    for t in trades:
        buy_date = str(t['buy_date'])[:10]
        sell_date = str(t['sell_date'])[:10]
        pnl_str = f"${t['pnl']:+.2f}"
        pnl_pct_str = f"{t['pnl_pct']*100:+.1f}%"
        print(f"{buy_date:<12} ${t['buy_price']:>7.2f} {sell_date:<12} ${t['sell_price']:>7.2f} "
              f"{pnl_str:>10} {pnl_pct_str:>8} {t['exit_type']}")
    
    # 统计
    total_pnl = sum(t['pnl'] for t in trades)
    win_trades = [t for t in trades if t['pnl'] > 0]
    lose_trades = [t for t in trades if t['pnl'] < 0]
    win_rate = len(win_trades) / len(trades) if trades else 0
    
    avg_win = np.mean([t['pnl'] for t in win_trades]) if win_trades else 0
    avg_lose = np.mean([t['pnl'] for t in lose_trades]) if lose_trades else 0
    
    # 计算持仓天数
    hold_days = []
    for t in trades:
        days = (pd.to_datetime(t['sell_date']) - pd.to_datetime(t['buy_date'])).days
        hold_days.append(days)
    avg_hold_days = np.mean(hold_days) if hold_days else 0
    
    print("\n" + "=" * 70)
    print("回测统计")
    print("=" * 70)
    print(f"总交易次数: {len(trades)}")
    print(f"盈利次数: {len(win_trades)} | 亏损次数: {len(lose_trades)}")
    print(f"胜率: {win_rate*100:.1f}%")
    print(f"总盈亏: ${total_pnl:+.2f}")
    print(f"平均盈利: ${avg_win:+.2f} | 平均亏损: ${avg_lose:+.2f}")
    print(f"盈亏比: {abs(avg_win/avg_lose):.2f}" if avg_lose != 0 else "盈亏比: N/A")
    print(f"平均持仓天数: {avg_hold_days:.1f} 天")
    
    # 按退出类型统计
    exit_types = {}
    for t in trades:
        exit_type = t['exit_type']
        if exit_type not in exit_types:
            exit_types[exit_type] = {'count': 0, 'pnl': 0}
        exit_types[exit_type]['count'] += 1
        exit_types[exit_type]['pnl'] += t['pnl']
    
    print("\n按退出类型统计:")
    for exit_type, stats in exit_types.items():
        print(f"  {exit_type}: {stats['count']} 笔, 盈亏 ${stats['pnl']:+.2f}")
    
    return trades, df


def plot_trades(symbol: str, df: pd.DataFrame, trades: list, 
                position_size: float = 5000, save_path: str = None):
    """
    绘制交易图表
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # 计算统计
    total_pnl = sum(t['pnl'] for t in trades)
    win_trades = len([t for t in trades if t['pnl'] > 0])
    win_rate = win_trades / len(trades) * 100 if trades else 0
    
    fig.suptitle(f'{symbol} Top 5 因子策略回测 (止盈10%/止损10%)', fontsize=14, fontweight='bold')
    
    # 图1: 价格图
    ax1 = axes[0]
    close_data = df['Close'].dropna()
    ax1.plot(close_data.index, close_data.values, label='收盘价', color='black', linewidth=1)
    
    if 'SMA_20' in df.columns:
        sma20 = df['SMA_20'].dropna()
        ax1.plot(sma20.index, sma20.values, label='MA20', color='blue', linewidth=0.8, alpha=0.7)
    
    # 标注买入信号点
    buy_signals = df[df['Buy_Signal'] == True]
    ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', 
                s=100, label='买入信号', zorder=5)
    
    # 标注交易（箭头+文字）
    for t in trades:
        color = 'green' if t['pnl'] > 0 else 'red'
        ax1.annotate(f"{t['exit_type']}\n{t['pnl_pct']*100:+.1f}%", 
                    xy=(t['buy_date'], t['buy_price']),
                    xytext=(0, 20), textcoords='offset points',
                    fontsize=8, color=color, ha='center',
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.5))
    
    ax1.set_ylabel('价格 ($)')
    ax1.set_title('价格走势与交易点')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 图2: 资金曲线
    ax2 = axes[1]
    capital_curve = [position_size]
    dates = [df.index[0]]
    
    for t in trades:
        dates.append(t['sell_date'])
        capital_curve.append(capital_curve[-1] + t['pnl'])
    
    ax2.plot(dates, capital_curve, color='steelblue', linewidth=2, marker='o')
    ax2.axhline(y=position_size, color='red', linestyle='--', linewidth=1, label='初始资金')
    ax2.fill_between(dates, position_size, capital_curve, alpha=0.3, 
                     color=['green' if c >= position_size else 'red' for c in capital_curve])
    ax2.set_ylabel('资金 ($)')
    ax2.set_xlabel('日期')
    ax2.set_title(f'资金曲线 (最终: ${capital_curve[-1]:,.2f}, 盈亏: ${total_pnl:+,.2f})')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存至 {save_path}")
    
    plt.show()
    return fig


if __name__ == '__main__':
    import sys
    
    # 支持命令行指定股票: python backtest_top5_strategy.py TSLA
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
        print(f"\n指定股票: {symbol}\n")
    else:
        symbol = random.choice(NASDAQ100)
        print(f"\n随机选择股票: {symbol}\n")
    
    result = backtest(
        symbol=symbol,
        min_factors=5,
        position_size=5000,
        take_profit=0.10,
        stop_loss=0.10
    )
    
    if result:
        trades, df = result
        plot_trades(symbol, df, trades, save_path=f'reports/{symbol.lower()}_backtest.png')
