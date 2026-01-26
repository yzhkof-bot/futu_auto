"""
随机纳斯达克100股票回测
- 买入条件：≥3个信号
- 买入金额：$5000
- 止盈：+10%
- 止损：-10%
- 同时只能持有一个仓位
"""

import warnings
warnings.filterwarnings('ignore')

import random
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from src.strategies import get_strategy

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 纳斯达克100成分股
NASDAQ100 = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'COST',
    'NFLX', 'TMUS', 'ASML', 'CSCO', 'ADBE', 'AMD', 'PEP', 'LIN', 'INTC', 'INTU',
    'TXN', 'CMCSA', 'QCOM', 'AMGN', 'AMAT', 'HON', 'ISRG', 'BKNG', 'SBUX', 'VRTX',
    'LRCX', 'ADP', 'GILD', 'ADI', 'MU', 'MDLZ', 'REGN', 'PANW', 'KLAC', 'SNPS',
    'CDNS', 'MELI', 'MAR', 'PYPL', 'CSX', 'CRWD', 'ORLY', 'CTAS', 'NXPI', 'MNST',
    'MRVL', 'PCAR', 'WDAY', 'ADSK', 'ABNB', 'CPRT', 'ROP', 'AEP', 'FTNT', 'PAYX',
    'AZN', 'CHTR', 'ROST', 'KDP', 'ODFL', 'DXCM', 'KHC', 'FAST', 'TTD', 'MCHP',
    'GEHC', 'VRSK', 'EA', 'CTSH', 'EXC', 'LULU', 'CSGP', 'FANG', 'IDXX', 'BKR',
    'XEL', 'CCEP', 'ON', 'ANSS', 'TEAM', 'CDW', 'BIIB', 'ZS', 'GFS', 'DDOG',
    'ILMN', 'WBD', 'MDB', 'MRNA', 'DLTR', 'CEG', 'SMCI', 'ARM', 'DASH', 'COIN'
]


def backtest_with_stop(ticker: str, initial_capital: float = 5000, 
                       take_profit: float = 0.10, stop_loss: float = -0.10):
    """
    带止盈止损的回测
    """
    print(f"下载 {ticker} 数据...")
    data = yf.download(ticker, start='2020-01-01', progress=False)  # 下载到最新数据
    
    if data.empty:
        print(f"{ticker} 数据为空")
        return None
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    print(f"数据范围: {data.index[0].strftime('%Y-%m-%d')} ~ {data.index[-1].strftime('%Y-%m-%d')}")
    
    # 计算指标和信号
    strategy = get_strategy(min_signals=3)
    df = strategy.run(data)
    
    # 模拟交易
    trades = []
    capital = initial_capital
    position = None  # {'entry_date', 'entry_price', 'shares'}
    
    for i in range(len(df)):
        date = df.index[i]
        row = df.iloc[i]
        high = row['High']
        low = row['Low']
        close = row['Close']
        
        # 如果有持仓，检查止盈止损
        if position is not None:
            entry_price = position['entry_price']
            hold_days = (date - position['entry_date']).days
            
            # 检查是否触发止盈（用最高价）
            if high >= entry_price * (1 + take_profit):
                exit_price = entry_price * (1 + take_profit)
                pnl = position['shares'] * (exit_price - entry_price)
                capital += position['shares'] * exit_price
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'shares': position['shares'],
                    'pnl': pnl,
                    'pnl_pct': take_profit,
                    'exit_reason': '止盈',
                    'hold_days': hold_days
                })
                position = None
                continue
            
            # 检查是否触发止损（用最低价）
            if low <= entry_price * (1 + stop_loss):
                exit_price = entry_price * (1 + stop_loss)
                pnl = position['shares'] * (exit_price - entry_price)
                capital += position['shares'] * exit_price
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'shares': position['shares'],
                    'pnl': pnl,
                    'pnl_pct': stop_loss,
                    'exit_reason': '止损',
                    'hold_days': hold_days
                })
                position = None
                continue
            
            # 检查是否超过3个月（约90天）
            if hold_days >= 90:
                exit_price = close
                pnl = position['shares'] * (exit_price - entry_price)
                pnl_pct = (exit_price - entry_price) / entry_price
                capital += position['shares'] * exit_price
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'shares': position['shares'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': '超时(3个月)',
                    'hold_days': hold_days
                })
                position = None
                continue
        
        # 如果没有持仓且有买入信号
        if position is None and row['Buy_Signal']:
            shares = initial_capital / close
            position = {
                'entry_date': date,
                'entry_price': close,
                'shares': shares
            }
            capital -= initial_capital
    
    # 如果最后还有持仓，按收盘价平仓
    if position is not None:
        exit_price = df.iloc[-1]['Close']
        pnl = position['shares'] * (exit_price - position['entry_price'])
        pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        hold_days = (df.index[-1] - position['entry_date']).days
        capital += position['shares'] * exit_price
        trades.append({
            'entry_date': position['entry_date'],
            'exit_date': df.index[-1],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'shares': position['shares'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': '持有中',
            'hold_days': hold_days
        })
    
    return {
        'ticker': ticker,
        'trades': trades,
        'final_capital': capital,
        'df': df
    }


def main(ticker=None):
    # 如果没有指定股票，随机选择
    if ticker is None:
        ticker = random.choice(NASDAQ100)
        print(f"随机选择: {ticker}")
    else:
        print(f"指定股票: {ticker}")
    print("=" * 50)
    
    result = backtest_with_stop(ticker, initial_capital=5000, take_profit=0.10, stop_loss=-0.10)
    
    if result is None:
        return
    
    trades = result['trades']
    final_capital = result['final_capital']
    df = result['df']
    
    # 统计
    if trades:
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        take_profits = [t for t in trades if t['exit_reason'] == '止盈']
        stop_losses = [t for t in trades if t['exit_reason'] == '止损']
        timeouts = [t for t in trades if t['exit_reason'] == '超时(3个月)']
        
        total_pnl = sum(t['pnl'] for t in trades)
        
        print(f"\n{'='*50}")
        print(f"{ticker} 回测结果")
        print(f"{'='*50}")
        print(f"初始资金: $5,000")
        print(f"最终资金: ${final_capital:,.2f}")
        print(f"总盈亏: ${total_pnl:,.2f} ({total_pnl/5000*100:+.2f}%)")
        print(f"\n总交易次数: {len(trades)}")
        print(f"盈利交易: {len(wins)}")
        print(f"亏损交易: {len(losses)}")
        print(f"胜率: {len(wins)/len(trades)*100:.2f}%")
        print(f"\n止盈次数: {len(take_profits)}")
        print(f"止损次数: {len(stop_losses)}")
        print(f"超时(3个月)次数: {len(timeouts)}")
        
        print(f"\n交易详情:")
        for i, t in enumerate(trades, 1):
            print(f"{i}. {t['entry_date'].strftime('%Y-%m-%d')} -> {t['exit_date'].strftime('%Y-%m-%d')} ({t['hold_days']}天): "
                  f"${t['entry_price']:.2f} -> ${t['exit_price']:.2f} "
                  f"({t['pnl_pct']*100:+.2f}%) ${t['pnl']:+.2f} [{t['exit_reason']}]")
    else:
        print("\n无交易信号")
    
    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f'{ticker} 多信号策略回测 (止盈10%/止损10%)', fontsize=14, fontweight='bold')
    
    # 价格图
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], label='收盘价', color='black', linewidth=1)
    ax1.plot(df.index, df['MA20'], label='MA20', color='blue', linewidth=0.8, alpha=0.7)
    
    # 标注买入点
    buy_signals = df[df['Buy_Signal']]
    ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='买入信号', zorder=5)
    
    # 标注交易
    for t in trades:
        color = 'green' if t['pnl'] > 0 else 'red'
        ax1.annotate(f"{t['exit_reason']}\n{t['pnl_pct']*100:+.1f}%", 
                    xy=(t['entry_date'], t['entry_price']),
                    xytext=(0, 20), textcoords='offset points',
                    fontsize=8, color=color, ha='center',
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.5))
    
    ax1.set_ylabel('价格 ($)')
    ax1.set_title('价格走势与交易点')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 资金曲线
    ax2 = axes[1]
    capital_curve = [5000]
    dates = [df.index[0]]
    
    for t in trades:
        dates.append(t['exit_date'])
        capital_curve.append(capital_curve[-1] + t['pnl'])
    
    ax2.plot(dates, capital_curve, color='steelblue', linewidth=2, marker='o')
    ax2.axhline(y=5000, color='red', linestyle='--', linewidth=1, label='初始资金')
    ax2.fill_between(dates, 5000, capital_curve, alpha=0.3, 
                     color=['green' if c >= 5000 else 'red' for c in capital_curve])
    ax2.set_ylabel('资金 ($)')
    ax2.set_xlabel('日期')
    ax2.set_title('资金曲线')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'reports/{ticker}_stop_backtest.png', dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: reports/{ticker}_stop_backtest.png")


if __name__ == '__main__':
    import sys
    
    # 支持命令行指定股票: python backtest_random_nasdaq.py TSLA
    if len(sys.argv) > 1:
        main(ticker=sys.argv[1].upper())
    else:
        main()
