"""
GP 因子回测

使用遗传规划挖掘出的因子进行单股票回测
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# GP 挖掘出的最佳因子公式
# ============================================================

def ts_delay(series, d=1):
    """时序延迟"""
    return series.shift(d)

def ts_delta(series, d=1):
    """时序差分"""
    return series - series.shift(d)

def ts_mean(series, d=5):
    """滚动均值"""
    return series.rolling(window=d, min_periods=1).mean()

def ts_std(series, d=5):
    """滚动标准差"""
    return series.rolling(window=d, min_periods=2).std()

def ts_max(series, d=5):
    """滚动最大值"""
    return series.rolling(window=d, min_periods=1).max()

def ts_min(series, d=5):
    """滚动最小值"""
    return series.rolling(window=d, min_periods=1).min()


def compute_gp_factor_1(df):
    """
    因子 #4 (最佳): mul(neg(sub(delay5(min5(volume)), mean10(std10(return_5)))), mean5(neg(std10(min5(close)))))
    
    测试集: IC=0.0724, ICIR=0.4558, Sharpe=2.72
    """
    # 计算 return_5
    return_5 = df['Close'].pct_change(5)
    
    # 左半部分: neg(sub(delay5(min5(volume)), mean10(std10(return_5))))
    min5_volume = ts_min(df['Volume'], 5)
    delay5_min5_volume = ts_delay(min5_volume, 5)
    
    std10_return_5 = ts_std(return_5, 10)
    mean10_std10_return_5 = ts_mean(std10_return_5, 10)
    
    left = -(delay5_min5_volume - mean10_std10_return_5)
    
    # 右半部分: mean5(neg(std10(min5(close))))
    min5_close = ts_min(df['Close'], 5)
    std10_min5_close = ts_std(min5_close, 10)
    neg_std10_min5_close = -std10_min5_close
    right = ts_mean(neg_std10_min5_close, 5)
    
    # 组合
    factor = left * right
    return factor


def compute_gp_factor_2(df):
    """
    因子 #1: mul(max5(amount), min5(amplitude))
    
    测试集: IC=0.0608, ICIR=0.3373, Sharpe=2.29
    """
    # 计算 amount 和 amplitude
    amount = df['Close'] * df['Volume']
    amplitude = (df['High'] - df['Low']) / df['Close']
    
    max5_amount = ts_max(amount, 5)
    min5_amplitude = ts_min(amplitude, 5)
    
    factor = max5_amount * min5_amplitude
    return factor


def compute_gp_factor_3(df):
    """
    因子 #2: min5(log(amount))
    
    测试集: IC=0.0489, ICIR=0.3396, Sharpe=2.35
    """
    amount = df['Close'] * df['Volume']
    log_amount = np.log(amount + 1e-10)
    factor = ts_min(log_amount, 5)
    return factor


# 因子字典
GP_FACTORS = {
    'factor_1': {
        'name': 'GP最佳因子',
        'formula': 'mul(neg(sub(delay5(min5(volume)), mean10(std10(return_5)))), mean5(neg(std10(min5(close)))))',
        'compute': compute_gp_factor_1,
        'test_ic': 0.0724,
        'test_sharpe': 2.72,
    },
    'factor_2': {
        'name': '量价振幅因子',
        'formula': 'mul(max5(amount), min5(amplitude))',
        'compute': compute_gp_factor_2,
        'test_ic': 0.0608,
        'test_sharpe': 2.29,
    },
    'factor_3': {
        'name': '最小成交额因子',
        'formula': 'min5(log(amount))',
        'compute': compute_gp_factor_3,
        'test_ic': 0.0489,
        'test_sharpe': 2.35,
    },
}


def backtest_gp_factor(
    symbol: str,
    factor_key: str = 'factor_1',
    position_size: float = 10000,
    take_profit: float = 0.10,
    stop_loss: float = 0.10,
    percentile_threshold: float = 0.8,  # 因子值超过历史80%分位数时买入
    start_date: str = None,
    end_date: str = None,
    lookback_days: int = 60,  # 计算分位数的回看天数
):
    """
    GP 因子回测
    
    策略逻辑:
    - 当因子值超过过去 N 天的 X 分位数时，产生买入信号
    - 止盈止损退出
    
    Args:
        symbol: 股票代码
        factor_key: 使用哪个因子
        position_size: 每笔仓位金额
        take_profit: 止盈比例
        stop_loss: 止损比例
        percentile_threshold: 因子分位数阈值
        start_date: 开始日期
        end_date: 结束日期
        lookback_days: 计算分位数的回看天数
    """
    factor_info = GP_FACTORS.get(factor_key)
    if not factor_info:
        print(f"未知因子: {factor_key}")
        return None
    
    print("=" * 70)
    print(f"GP 因子回测: {symbol}")
    print("=" * 70)
    print(f"因子: {factor_info['name']}")
    print(f"公式: {factor_info['formula']}")
    print(f"测试集 IC: {factor_info['test_ic']:.4f}, Sharpe: {factor_info['test_sharpe']:.2f}")
    print("-" * 70)
    print(f"仓位: ${position_size:.0f} | 止盈: +{take_profit*100:.0f}% | 止损: -{stop_loss*100:.0f}%")
    print(f"买入条件: 因子值 > 过去{lookback_days}天的{percentile_threshold*100:.0f}%分位数")
    print("=" * 70)
    
    # 日期范围
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    # 需要额外的历史数据来计算因子
    fetch_start = (pd.to_datetime(start_date) - timedelta(days=100)).strftime('%Y-%m-%d')
    
    print(f"回测区间: {start_date} ~ {end_date}")
    
    # 获取数据
    print(f"正在获取 {symbol} 数据...")
    df = yf.download(symbol, start=fetch_start, end=end_date, progress=False)
    
    if df.empty:
        print("数据获取失败")
        return None
    
    # 处理多级索引
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print(f"获取数据: {len(df)} 天")
    
    # 计算因子
    df['Factor'] = factor_info['compute'](df)
    
    # 计算滚动分位数
    df['Factor_Percentile'] = df['Factor'].rolling(window=lookback_days, min_periods=20).apply(
        lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 1 else 0.5,
        raw=False
    )
    
    # 生成买入信号
    df['Buy_Signal'] = df['Factor_Percentile'] > percentile_threshold
    
    # 过滤到回测区间
    df = df[df.index >= start_date]
    
    print(f"回测天数: {len(df)}")
    print(f"买入信号数: {df['Buy_Signal'].sum()}")
    print("-" * 70)
    
    # 回测
    trades = []
    position = None
    
    for i in range(len(df)):
        date = df.index[i]
        row = df.iloc[i]
        close = row['Close']
        high = row['High']
        low = row['Low']
        
        # 如果有持仓，检查止盈止损
        if position is not None:
            entry_price = position['price']
            
            # 检查止盈
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
                    'exit_type': '止盈',
                    'factor_value': position['factor_value'],
                })
                position = None
                continue
            
            # 检查止损
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
                    'exit_type': '止损',
                    'factor_value': position['factor_value'],
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
                'cost': position_size,
                'factor_value': row['Factor'],
            }
    
    # 如果最后还有持仓
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
            'exit_type': '持仓中',
            'factor_value': position['factor_value'],
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
    
    # 计算收益率
    total_return = total_pnl / position_size * 100
    
    print("\n" + "=" * 70)
    print("回测统计")
    print("=" * 70)
    print(f"总交易次数: {len(trades)}")
    print(f"盈利次数: {len(win_trades)} | 亏损次数: {len(lose_trades)}")
    print(f"胜率: {win_rate*100:.1f}%")
    print(f"总盈亏: ${total_pnl:+.2f} ({total_return:+.1f}%)")
    print(f"平均盈利: ${avg_win:+.2f} | 平均亏损: ${avg_lose:+.2f}")
    if avg_lose != 0:
        print(f"盈亏比: {abs(avg_win/avg_lose):.2f}")
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


def plot_gp_backtest(symbol: str, df: pd.DataFrame, trades: list, 
                     factor_info: dict, position_size: float = 10000,
                     save_path: str = None):
    """绘制回测图表"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # 计算统计
    total_pnl = sum(t['pnl'] for t in trades)
    win_trades = len([t for t in trades if t['pnl'] > 0])
    win_rate = win_trades / len(trades) * 100 if trades else 0
    
    fig.suptitle(f'{symbol} GP因子回测: {factor_info["name"]}\n公式: {factor_info["formula"]}', 
                 fontsize=12, fontweight='bold')
    
    # 图1: 价格图
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], label='收盘价', color='black', linewidth=1)
    
    # 标注买入信号点
    buy_signals = df[df['Buy_Signal'] == True]
    ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', 
                s=80, label='买入信号', zorder=5, alpha=0.7)
    
    # 标注交易
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
    
    # 图2: 因子值
    ax2 = axes[1]
    ax2.plot(df.index, df['Factor'], label='因子值', color='purple', linewidth=1)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    
    # 标注买入点的因子值
    ax2.scatter(buy_signals.index, buy_signals['Factor'], marker='^', color='green', 
                s=80, zorder=5, alpha=0.7)
    
    ax2.set_ylabel('因子值')
    ax2.set_title('GP 因子值')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 图3: 资金曲线
    ax3 = axes[2]
    capital_curve = [position_size]
    dates = [df.index[0]]
    
    for t in trades:
        dates.append(t['sell_date'])
        capital_curve.append(capital_curve[-1] + t['pnl'])
    
    ax3.plot(dates, capital_curve, color='steelblue', linewidth=2, marker='o', markersize=4)
    ax3.axhline(y=position_size, color='red', linestyle='--', linewidth=1, label='初始资金')
    ax3.fill_between(dates, position_size, capital_curve, alpha=0.3, 
                     color=['green' if c >= position_size else 'red' for c in capital_curve])
    ax3.set_ylabel('资金 ($)')
    ax3.set_xlabel('日期')
    ax3.set_title(f'资金曲线 (最终: ${capital_curve[-1]:,.2f}, 盈亏: ${total_pnl:+,.2f}, 胜率: {win_rate:.1f}%)')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存至 {save_path}")
    
    plt.show()
    return fig


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='GP 因子回测')
    parser.add_argument('symbol', nargs='?', default='AAPL', help='股票代码')
    parser.add_argument('--factor', '-f', default='factor_1', 
                        choices=['factor_1', 'factor_2', 'factor_3'],
                        help='使用哪个因子')
    parser.add_argument('--position', '-p', type=float, default=10000, help='仓位金额')
    parser.add_argument('--take-profit', '-tp', type=float, default=0.10, help='止盈比例')
    parser.add_argument('--stop-loss', '-sl', type=float, default=0.10, help='止损比例')
    parser.add_argument('--threshold', '-t', type=float, default=0.8, help='因子分位数阈值')
    parser.add_argument('--lookback', '-l', type=int, default=60, help='回看天数')
    parser.add_argument('--start', '-s', default=None, help='开始日期')
    parser.add_argument('--end', '-e', default=None, help='结束日期')
    parser.add_argument('--plot', action='store_true', help='绘制图表')
    parser.add_argument('--save', action='store_true', help='保存图表')
    
    args = parser.parse_args()
    
    result = backtest_gp_factor(
        symbol=args.symbol.upper(),
        factor_key=args.factor,
        position_size=args.position,
        take_profit=args.take_profit,
        stop_loss=args.stop_loss,
        percentile_threshold=args.threshold,
        lookback_days=args.lookback,
        start_date=args.start,
        end_date=args.end,
    )
    
    if result:
        trades, df = result
        factor_info = GP_FACTORS[args.factor]
        
        save_path = None
        if args.save:
            save_path = f'reports/{args.symbol.lower()}_gp_backtest.png'
        
        if args.plot or args.save:
            plot_gp_backtest(args.symbol.upper(), df, trades, factor_info,
                           position_size=args.position, save_path=save_path)
