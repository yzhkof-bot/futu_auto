"""
GP 因子回测 V2 - 持有期策略

更专业的因子回测方式：
1. 因子值高时买入，固定持有 N 天后卖出
2. 因子值低时做空（可选）
3. 统计不同持有期的收益分布
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# GP 因子计算函数
# ============================================================

def ts_delay(series, d=1):
    return series.shift(d)

def ts_delta(series, d=1):
    return series - series.shift(d)

def ts_mean(series, d=5):
    return series.rolling(window=d, min_periods=1).mean()

def ts_std(series, d=5):
    return series.rolling(window=d, min_periods=2).std()

def ts_max(series, d=5):
    return series.rolling(window=d, min_periods=1).max()

def ts_min(series, d=5):
    return series.rolling(window=d, min_periods=1).min()


def compute_gp_factor_1(df):
    """因子 #4: mul(neg(sub(delay5(min5(volume)), mean10(std10(return_5)))), mean5(neg(std10(min5(close)))))"""
    return_5 = df['Close'].pct_change(5)
    min5_volume = ts_min(df['Volume'], 5)
    delay5_min5_volume = ts_delay(min5_volume, 5)
    std10_return_5 = ts_std(return_5, 10)
    mean10_std10_return_5 = ts_mean(std10_return_5, 10)
    left = -(delay5_min5_volume - mean10_std10_return_5)
    min5_close = ts_min(df['Close'], 5)
    std10_min5_close = ts_std(min5_close, 10)
    right = ts_mean(-std10_min5_close, 5)
    return left * right


def compute_gp_factor_2(df):
    """因子 #1: mul(max5(amount), min5(amplitude))"""
    amount = df['Close'] * df['Volume']
    amplitude = (df['High'] - df['Low']) / df['Close']
    return ts_max(amount, 5) * ts_min(amplitude, 5)


def compute_gp_factor_3(df):
    """因子 #2: min5(log(amount))"""
    amount = df['Close'] * df['Volume']
    return ts_min(np.log(amount + 1e-10), 5)


GP_FACTORS = {
    'factor_1': {'name': 'GP最佳因子', 'compute': compute_gp_factor_1},
    'factor_2': {'name': '量价振幅因子', 'compute': compute_gp_factor_2},
    'factor_3': {'name': '最小成交额因子', 'compute': compute_gp_factor_3},
}


# ============================================================
# 持有期回测
# ============================================================

def backtest_holding_period(
    symbol: str,
    factor_key: str = 'factor_1',
    holding_days: int = 5,
    top_percentile: float = 0.8,
    bottom_percentile: float = 0.2,
    position_size: float = 10000,
    allow_short: bool = False,
    start_date: str = None,
    end_date: str = None,
    lookback_days: int = 60,
):
    """
    持有期回测
    
    策略逻辑:
    - 因子值 > top_percentile: 买入，持有 holding_days 天后卖出
    - 因子值 < bottom_percentile: 做空（如果允许），持有 holding_days 天后平仓
    - 不重叠持仓（上一笔平仓后才能开新仓）
    """
    factor_info = GP_FACTORS.get(factor_key)
    if not factor_info:
        print(f"未知因子: {factor_key}")
        return None
    
    print("=" * 70)
    print(f"GP 因子持有期回测: {symbol}")
    print("=" * 70)
    print(f"因子: {factor_info['name']}")
    print(f"持有期: {holding_days} 天")
    print(f"做多条件: 因子值 > {top_percentile*100:.0f}% 分位")
    if allow_short:
        print(f"做空条件: 因子值 < {bottom_percentile*100:.0f}% 分位")
    print(f"仓位: ${position_size:.0f}")
    print("=" * 70)
    
    # 日期范围
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    fetch_start = (pd.to_datetime(start_date) - timedelta(days=100)).strftime('%Y-%m-%d')
    
    # 获取数据
    print(f"正在获取 {symbol} 数据...")
    df = yf.download(symbol, start=fetch_start, end=end_date, progress=False)
    
    if df.empty:
        print("数据获取失败")
        return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 计算因子
    df['Factor'] = factor_info['compute'](df)
    
    # 计算滚动分位数
    df['Factor_Pct'] = df['Factor'].rolling(window=lookback_days, min_periods=20).rank(pct=True)
    
    # 计算未来 N 天收益
    df['Forward_Return'] = df['Close'].shift(-holding_days) / df['Close'] - 1
    
    # 过滤到回测区间
    df = df[df.index >= start_date].copy()
    df = df.dropna(subset=['Factor', 'Factor_Pct', 'Forward_Return'])
    
    print(f"回测天数: {len(df)}")
    
    # 回测
    trades = []
    position = None  # {'date', 'price', 'direction', 'exit_date'}
    
    for i in range(len(df) - holding_days):
        date = df.index[i]
        row = df.iloc[i]
        close = row['Close']
        factor_pct = row['Factor_Pct']
        
        # 如果有持仓，检查是否到期
        if position is not None:
            if date >= position['exit_date']:
                # 平仓
                exit_idx = df.index.get_loc(position['exit_date'])
                if exit_idx < len(df):
                    exit_price = df.iloc[exit_idx]['Close']
                    
                    if position['direction'] == 'long':
                        pnl_pct = (exit_price - position['price']) / position['price']
                    else:  # short
                        pnl_pct = (position['price'] - exit_price) / position['price']
                    
                    pnl = pnl_pct * position_size
                    
                    trades.append({
                        'entry_date': position['date'],
                        'entry_price': position['price'],
                        'exit_date': position['exit_date'],
                        'exit_price': exit_price,
                        'direction': position['direction'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'factor_pct': position['factor_pct'],
                    })
                position = None
        
        # 如果没有持仓，检查是否开仓
        if position is None:
            # 计算退出日期
            exit_idx = min(i + holding_days, len(df) - 1)
            exit_date = df.index[exit_idx]
            
            if factor_pct > top_percentile:
                # 做多
                position = {
                    'date': date,
                    'price': close,
                    'direction': 'long',
                    'exit_date': exit_date,
                    'factor_pct': factor_pct,
                }
            elif allow_short and factor_pct < bottom_percentile:
                # 做空
                position = {
                    'date': date,
                    'price': close,
                    'direction': 'short',
                    'exit_date': exit_date,
                    'factor_pct': factor_pct,
                }
    
    # 处理最后一笔持仓
    if position is not None and position['exit_date'] in df.index:
        exit_price = df.loc[position['exit_date'], 'Close']
        if position['direction'] == 'long':
            pnl_pct = (exit_price - position['price']) / position['price']
        else:
            pnl_pct = (position['price'] - exit_price) / position['price']
        pnl = pnl_pct * position_size
        
        trades.append({
            'entry_date': position['date'],
            'entry_price': position['price'],
            'exit_date': position['exit_date'],
            'exit_price': exit_price,
            'direction': position['direction'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'factor_pct': position['factor_pct'],
        })
    
    if not trades:
        print("\n无交易记录")
        return None
    
    # 分离多空交易
    long_trades = [t for t in trades if t['direction'] == 'long']
    short_trades = [t for t in trades if t['direction'] == 'short']
    
    # 输出交易记录
    print(f"\n交易记录 ({len(trades)} 笔, 做多 {len(long_trades)} 笔, 做空 {len(short_trades)} 笔):")
    print("-" * 80)
    print(f"{'方向':<6} {'入场日期':<12} {'入场价':>8} {'出场日期':<12} {'出场价':>8} {'盈亏':>10} {'收益率':>8}")
    print("-" * 80)
    
    for t in trades[-20:]:  # 只显示最近20笔
        direction = '做多' if t['direction'] == 'long' else '做空'
        entry_date = str(t['entry_date'])[:10]
        exit_date = str(t['exit_date'])[:10]
        pnl_str = f"${t['pnl']:+.2f}"
        pnl_pct_str = f"{t['pnl_pct']*100:+.1f}%"
        print(f"{direction:<6} {entry_date:<12} ${t['entry_price']:>7.2f} {exit_date:<12} ${t['exit_price']:>7.2f} "
              f"{pnl_str:>10} {pnl_pct_str:>8}")
    
    if len(trades) > 20:
        print(f"... 省略 {len(trades) - 20} 笔交易")
    
    # 统计
    print("\n" + "=" * 70)
    print("回测统计")
    print("=" * 70)
    
    def calc_stats(trade_list, name):
        if not trade_list:
            return
        
        total_pnl = sum(t['pnl'] for t in trade_list)
        pnl_pcts = [t['pnl_pct'] for t in trade_list]
        win_trades = [t for t in trade_list if t['pnl'] > 0]
        win_rate = len(win_trades) / len(trade_list)
        
        avg_return = np.mean(pnl_pcts) * 100
        std_return = np.std(pnl_pcts) * 100
        sharpe = avg_return / std_return * np.sqrt(252 / holding_days) if std_return > 0 else 0
        
        max_win = max(pnl_pcts) * 100
        max_loss = min(pnl_pcts) * 100
        
        print(f"\n【{name}】({len(trade_list)} 笔)")
        print(f"  胜率: {win_rate*100:.1f}%")
        print(f"  总盈亏: ${total_pnl:+,.2f}")
        print(f"  平均收益: {avg_return:+.2f}% ± {std_return:.2f}%")
        print(f"  年化 Sharpe: {sharpe:.2f}")
        print(f"  最大盈利: {max_win:+.1f}% | 最大亏损: {max_loss:+.1f}%")
        
        return {
            'count': len(trade_list),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_return': avg_return,
            'sharpe': sharpe,
        }
    
    stats_all = calc_stats(trades, '全部交易')
    stats_long = calc_stats(long_trades, '做多交易') if long_trades else None
    stats_short = calc_stats(short_trades, '做空交易') if short_trades else None
    
    # 按年份统计
    print("\n" + "-" * 70)
    print("按年份统计:")
    trades_df = pd.DataFrame(trades)
    trades_df['year'] = pd.to_datetime(trades_df['entry_date']).dt.year
    
    for year, group in trades_df.groupby('year'):
        year_pnl = group['pnl'].sum()
        year_win_rate = (group['pnl'] > 0).mean() * 100
        print(f"  {year}: {len(group)} 笔, 胜率 {year_win_rate:.0f}%, 盈亏 ${year_pnl:+,.2f}")
    
    return trades, df, {'all': stats_all, 'long': stats_long, 'short': stats_short}


def compare_holding_periods(
    symbol: str,
    factor_key: str = 'factor_1',
    holding_days_list: list = [3, 5, 10, 20],
    top_percentile: float = 0.8,
    position_size: float = 10000,
):
    """比较不同持有期的表现"""
    
    print("=" * 70)
    print(f"不同持有期对比: {symbol}")
    print("=" * 70)
    
    results = []
    
    for days in holding_days_list:
        print(f"\n>>> 持有期 {days} 天 <<<")
        result = backtest_holding_period(
            symbol=symbol,
            factor_key=factor_key,
            holding_days=days,
            top_percentile=top_percentile,
            position_size=position_size,
            allow_short=False,
        )
        
        if result:
            trades, df, stats = result
            results.append({
                'holding_days': days,
                'trades': len(trades),
                'win_rate': stats['all']['win_rate'] * 100,
                'avg_return': stats['all']['avg_return'],
                'sharpe': stats['all']['sharpe'],
                'total_pnl': stats['all']['total_pnl'],
            })
    
    # 汇总对比
    print("\n" + "=" * 70)
    print("持有期对比汇总")
    print("=" * 70)
    print(f"{'持有期':>8} {'交易数':>8} {'胜率':>8} {'平均收益':>10} {'Sharpe':>8} {'总盈亏':>12}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['holding_days']:>6}天 {r['trades']:>8} {r['win_rate']:>7.1f}% "
              f"{r['avg_return']:>+9.2f}% {r['sharpe']:>8.2f} ${r['total_pnl']:>+10,.2f}")
    
    return results


def plot_holding_backtest(symbol: str, df: pd.DataFrame, trades: list, 
                          holding_days: int, position_size: float = 10000,
                          save_path: str = None):
    """绘制持有期回测图表"""
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=False)
    
    # 统计
    total_pnl = sum(t['pnl'] for t in trades)
    win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100
    pnl_pcts = [t['pnl_pct'] * 100 for t in trades]
    
    fig.suptitle(f'{symbol} GP因子持有期回测 (持有 {holding_days} 天)', fontsize=14, fontweight='bold')
    
    # 图1: 价格 + 交易点
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], label='收盘价', color='black', linewidth=1)
    
    # 标注做多交易
    for t in trades:
        if t['direction'] == 'long':
            color = 'green' if t['pnl'] > 0 else 'red'
            ax1.scatter(t['entry_date'], t['entry_price'], marker='^', color=color, s=50, zorder=5)
            ax1.scatter(t['exit_date'], t['exit_price'], marker='v', color=color, s=50, zorder=5)
    
    ax1.set_ylabel('价格 ($)')
    ax1.set_title('价格走势与交易点 (△买入 ▽卖出)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 图2: 收益分布
    ax2 = axes[1]
    colors = ['green' if p > 0 else 'red' for p in pnl_pcts]
    ax2.bar(range(len(pnl_pcts)), pnl_pcts, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=np.mean(pnl_pcts), color='blue', linestyle='--', linewidth=1, 
                label=f'平均: {np.mean(pnl_pcts):+.2f}%')
    ax2.set_ylabel('收益率 (%)')
    ax2.set_xlabel('交易序号')
    ax2.set_title(f'每笔交易收益分布 (胜率: {win_rate:.1f}%)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 图3: 累计资金曲线
    ax3 = axes[2]
    capital_curve = [position_size]
    dates = [trades[0]['entry_date']]
    
    for t in trades:
        dates.append(t['exit_date'])
        capital_curve.append(capital_curve[-1] + t['pnl'])
    
    ax3.plot(dates, capital_curve, color='steelblue', linewidth=2)
    ax3.fill_between(dates, position_size, capital_curve, alpha=0.3,
                     color='green' if capital_curve[-1] >= position_size else 'red')
    ax3.axhline(y=position_size, color='red', linestyle='--', linewidth=1, label='初始资金')
    ax3.set_ylabel('资金 ($)')
    ax3.set_xlabel('日期')
    ax3.set_title(f'累计资金曲线 (最终: ${capital_curve[-1]:,.2f}, 盈亏: ${total_pnl:+,.2f})')
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
    
    parser = argparse.ArgumentParser(description='GP 因子持有期回测')
    parser.add_argument('symbol', nargs='?', default='AAPL', help='股票代码')
    parser.add_argument('--factor', '-f', default='factor_1', 
                        choices=['factor_1', 'factor_2', 'factor_3'])
    parser.add_argument('--holding', '-d', type=int, default=5, help='持有天数')
    parser.add_argument('--threshold', '-t', type=float, default=0.8, help='因子分位数阈值')
    parser.add_argument('--position', '-p', type=float, default=10000, help='仓位金额')
    parser.add_argument('--short', action='store_true', help='允许做空')
    parser.add_argument('--compare', action='store_true', help='对比不同持有期')
    parser.add_argument('--plot', action='store_true', help='绘制图表')
    parser.add_argument('--save', action='store_true', help='保存图表')
    
    args = parser.parse_args()
    
    if args.compare:
        # 对比不同持有期
        compare_holding_periods(
            symbol=args.symbol.upper(),
            factor_key=args.factor,
            holding_days_list=[3, 5, 10, 20],
            top_percentile=args.threshold,
            position_size=args.position,
        )
    else:
        # 单一持有期回测
        result = backtest_holding_period(
            symbol=args.symbol.upper(),
            factor_key=args.factor,
            holding_days=args.holding,
            top_percentile=args.threshold,
            position_size=args.position,
            allow_short=args.short,
        )
        
        if result and (args.plot or args.save):
            trades, df, stats = result
            save_path = f'reports/{args.symbol.lower()}_gp_holding_backtest.png' if args.save else None
            plot_holding_backtest(args.symbol.upper(), df, trades, args.holding,
                                args.position, save_path)
