"""
COIN 多信号策略回测 - 近5年
"""

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from src.strategies import get_strategy

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    ticker = 'ARM'
    
    # 下载近5年数据
    print(f"下载 {ticker} 数据...")
    data = yf.download(ticker, start='2020-01-01', end='2025-12-31', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    print(f"数据范围: {data.index[0].strftime('%Y-%m-%d')} ~ {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"数据条数: {len(data)}")
    
    # 回测
    strategy = get_strategy(min_signals=3, hold_days=20)
    trades, stats = strategy.backtest(data)
    df = strategy.run(data)
    
    print(f"\n{'='*50}")
    print(f"{ticker} 多信号策略回测结果")
    print(f"{'='*50}")
    print(f"总交易次数: {stats['total_trades']}")
    print(f"盈利交易: {stats['winning_trades']}")
    print(f"亏损交易: {stats['losing_trades']}")
    print(f"胜率: {stats['win_rate']:.2f}%")
    print(f"平均收益: {stats['avg_return']:.2f}%")
    print(f"中位数收益: {stats['median_return']:.2f}%")
    print(f"最大收益: {stats['max_return']:.2f}%")
    print(f"最小收益: {stats['min_return']:.2f}%")
    print(f"总收益: {stats['total_return']:.2f}%")
    
    if trades:
        print(f"\n交易详情:")
        for i, trade in enumerate(trades, 1):
            result = '✓' if trade.pnl_pct > 0 else '✗'
            print(f"{i}. {trade.entry_date.strftime('%Y-%m-%d')}: ${trade.entry_price:.2f} ({trade.pnl_pct*100:+.2f}%) {result}")
    
    # 绘图
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(f'{ticker} ARM Holdings - 多信号策略买点', fontsize=14, fontweight='bold')
    
    # 价格
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], label='收盘价', color='black', linewidth=1)
    ax1.plot(df.index, df['MA20'], label='MA20', color='blue', linewidth=0.8, alpha=0.7)
    ax1.plot(df.index, df['MA60'], label='MA60', color='orange', linewidth=0.8, alpha=0.7)
    buy_signals = df[df['Buy_Signal']]
    ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='买入信号', zorder=5)
    ax1.set_ylabel('价格 ($)')
    ax1.set_title('价格走势与买入信号')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # RSI
    ax2 = axes[1]
    ax2.plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=1)
    ax2.axhline(y=70, color='red', linestyle='--', linewidth=0.8)
    ax2.axhline(y=30, color='green', linestyle='--', linewidth=0.8)
    ax2.fill_between(df.index, 30, df['RSI'], where=df['RSI'] < 30, alpha=0.3, color='green')
    ax2.scatter(buy_signals.index, buy_signals['RSI'], marker='^', color='green', s=80, zorder=5)
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # KDJ
    ax3 = axes[2]
    ax3.plot(df.index, df['K'], label='K', color='blue', linewidth=1)
    ax3.plot(df.index, df['D'], label='D', color='orange', linewidth=1)
    ax3.axhline(y=80, color='red', linestyle='--', linewidth=0.8)
    ax3.axhline(y=20, color='green', linestyle='--', linewidth=0.8)
    ax3.fill_between(df.index, 20, df['K'], where=df['K'] < 20, alpha=0.3, color='green')
    ax3.scatter(buy_signals.index, buy_signals['K'], marker='^', color='green', s=80, zorder=5)
    ax3.set_ylabel('KDJ')
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # MACD
    ax4 = axes[3]
    ax4.plot(df.index, df['MACD'], label='MACD', color='blue', linewidth=1)
    ax4.plot(df.index, df['Signal_Line'], label='Signal', color='orange', linewidth=1)
    ax4.bar(df.index, df['MACD_Hist'], color=['green' if x >= 0 else 'red' for x in df['MACD_Hist']], alpha=0.5)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.scatter(buy_signals.index, buy_signals['MACD'], marker='^', color='green', s=80, zorder=5)
    ax4.set_ylabel('MACD')
    ax4.set_xlabel('日期')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/ARM_strategy_backtest.png', dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: reports/ARM_strategy_backtest.png")


if __name__ == '__main__':
    main()
