"""
GP 因子截面选股回测

正确的因子回测方式：
1. 每天计算所有股票的因子值
2. 选出因子值 Top N% 的股票做多
3. 选出因子值 Bottom N% 的股票做空（可选）
4. 持有固定天数后换仓
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
from src.stock_pool import get_stock_pool

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# GP 因子计算（Panel 版本）
# ============================================================

def ts_delay(panel, d=1):
    return panel.shift(d)

def ts_delta(panel, d=1):
    return panel - panel.shift(d)

def ts_mean(panel, d=5):
    return panel.rolling(window=d, min_periods=1).mean()

def ts_std(panel, d=5):
    return panel.rolling(window=d, min_periods=2).std()

def ts_max(panel, d=5):
    return panel.rolling(window=d, min_periods=1).max()

def ts_min(panel, d=5):
    return panel.rolling(window=d, min_periods=1).min()


def compute_gp_factor_1(close, high, low, volume):
    """
    因子 #4 (最佳): mul(neg(sub(delay5(min5(volume)), mean10(std10(return_5)))), mean5(neg(std10(min5(close)))))
    测试集: IC=0.0724, ICIR=0.4558, Sharpe=2.72
    """
    return_5 = close.pct_change(5)
    min5_volume = ts_min(volume, 5)
    delay5_min5_volume = ts_delay(min5_volume, 5)
    std10_return_5 = ts_std(return_5, 10)
    mean10_std10_return_5 = ts_mean(std10_return_5, 10)
    left = -(delay5_min5_volume - mean10_std10_return_5)
    
    min5_close = ts_min(close, 5)
    std10_min5_close = ts_std(min5_close, 10)
    right = ts_mean(-std10_min5_close, 5)
    
    return left * right


def compute_gp_factor_2(close, high, low, volume):
    """因子 #1: mul(max5(amount), min5(amplitude))"""
    amount = close * volume
    amplitude = (high - low) / close
    return ts_max(amount, 5) * ts_min(amplitude, 5)


def compute_gp_factor_3(close, high, low, volume):
    """因子 #2: min5(log(amount))"""
    amount = close * volume
    return ts_min(np.log(amount + 1e-10), 5)


GP_FACTORS = {
    'factor_1': {'name': 'GP最佳因子', 'compute': compute_gp_factor_1},
    'factor_2': {'name': '量价振幅因子', 'compute': compute_gp_factor_2},
    'factor_3': {'name': '最小成交额因子', 'compute': compute_gp_factor_3},
}


# ============================================================
# 数据获取（带智能缓存）
# ============================================================

from gp_alpha.data_manager import PanelDataManager

# 全局缓存
_data_cache = {}

def fetch_panel_data(symbols, start_date, end_date, pool_type='nasdaq100', verbose=True):
    """
    批量获取股票数据，返回 Panel 格式
    
    智能缓存逻辑：
    - 使用统一的最长时间范围缓存，避免重复下载
    - 不同回测区间共享同一份缓存数据
    """
    global _data_cache
    
    # 统一使用固定的历史起点，确保所有回测共享同一份缓存
    # 这样无论回测 3 年还是 10 年，都用同一份数据
    fixed_start = '2010-01-01'  # 固定起点，覆盖所有可能的回测需求
    
    # 检查内存缓存
    cache_key = f"{pool_type}_{fixed_start}"
    
    if cache_key in _data_cache:
        cached = _data_cache[cache_key]
        cached_end = cached['close'].index[-1].strftime('%Y-%m-%d')
        
        # 如果缓存已经包含到 end_date，直接返回
        if cached_end >= end_date:
            if verbose:
                print(f"使用内存缓存 (截至 {cached_end})")
            return cached
        else:
            if verbose:
                print(f"缓存过期 ({cached_end})，重新获取...")
    
    # 使用 PanelDataManager 获取数据（它有磁盘缓存）
    dm = PanelDataManager()
    dm.fetch(
        symbols=symbols,
        start_date=fixed_start,
        end_date=end_date,
        pool_type=pool_type,
        use_cache=True,
        verbose=verbose
    )
    
    if dm.close_panel is None or dm.close_panel.empty:
        return None
    
    result = {
        'close': dm.close_panel,
        'high': dm.high_panel,
        'low': dm.low_panel,
        'volume': dm.volume_panel,
    }
    
    # 更新内存缓存
    _data_cache[cache_key] = result
    
    if verbose:
        print(f"数据获取完成: {len(dm.close_panel)} 天, {len(dm.close_panel.columns)} 只股票")
    
    return result


# ============================================================
# 截面选股回测
# ============================================================

def backtest_cross_sectional(
    pool_type: str = 'nasdaq100',
    factor_key: str = 'factor_1',
    holding_days: int = 5,
    top_pct: float = 0.2,        # 做多 Top 20%
    bottom_pct: float = 0.2,     # 做空 Bottom 20%（可选）
    initial_capital: float = 100000,
    allow_short: bool = False,
    start_date: str = None,
    end_date: str = None,
    verbose: bool = True,
):
    """
    截面选股回测
    
    策略逻辑:
    - 每 holding_days 天调仓一次
    - 选因子值 Top N% 的股票等权做多
    - 选因子值 Bottom N% 的股票等权做空（可选）
    """
    factor_info = GP_FACTORS.get(factor_key)
    if not factor_info:
        print(f"未知因子: {factor_key}")
        return None
    
    # 获取股票池
    symbols = get_stock_pool(pool_type)
    
    print("=" * 70)
    print(f"GP 因子截面选股回测")
    print("=" * 70)
    print(f"股票池: {pool_type} ({len(symbols)} 只)")
    print(f"因子: {factor_info['name']}")
    print(f"调仓周期: {holding_days} 天")
    print(f"做多: Top {top_pct*100:.0f}% ({int(len(symbols)*top_pct)} 只)")
    if allow_short:
        print(f"做空: Bottom {bottom_pct*100:.0f}% ({int(len(symbols)*bottom_pct)} 只)")
    print(f"初始资金: ${initial_capital:,.0f}")
    print("=" * 70)
    
    # 日期范围
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    
    print(f"回测区间: {start_date} ~ {end_date}")
    
    # 获取数据（带缓存）
    panels = fetch_panel_data(symbols, start_date, end_date, pool_type, verbose)
    if panels is None:
        print("数据获取失败")
        return None
    
    close = panels['close']
    high = panels['high']
    low = panels['low']
    volume = panels['volume']
    
    # 计算因子
    if verbose:
        print("计算因子...")
    factor = factor_info['compute'](close, high, low, volume)
    
    # 计算未来收益
    forward_return = close.shift(-holding_days) / close - 1
    
    # 过滤到回测区间
    factor = factor[factor.index >= start_date]
    close = close[close.index >= start_date]
    forward_return = forward_return[forward_return.index >= start_date]
    
    # 生成调仓日期（每 holding_days 天调仓一次）
    all_dates = factor.index.tolist()
    rebalance_dates = all_dates[::holding_days]
    
    if verbose:
        print(f"调仓次数: {len(rebalance_dates)}")
    
    # 回测
    portfolio_values = [initial_capital]
    portfolio_dates = [pd.to_datetime(start_date)]
    
    all_trades = []
    
    for i, date in enumerate(rebalance_dates[:-1]):  # 最后一天不调仓
        # 获取当天因子值
        factor_today = factor.loc[date].dropna()
        
        if len(factor_today) < 10:
            continue
        
        # 截面排名
        n_long = max(1, int(len(factor_today) * top_pct))
        n_short = max(1, int(len(factor_today) * bottom_pct)) if allow_short else 0
        
        # 选股
        ranked = factor_today.sort_values(ascending=False)
        long_stocks = ranked.head(n_long).index.tolist()
        short_stocks = ranked.tail(n_short).index.tolist() if allow_short else []
        
        # 计算收益
        next_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else all_dates[-1]
        
        # 做多收益
        long_returns = []
        for stock in long_stocks:
            if stock in close.columns:
                try:
                    entry_price = close.loc[date, stock]
                    exit_price = close.loc[next_date, stock]
                    if pd.notna(entry_price) and pd.notna(exit_price) and entry_price > 0:
                        ret = (exit_price - entry_price) / entry_price
                        long_returns.append(ret)
                except:
                    pass
        
        # 做空收益
        short_returns = []
        for stock in short_stocks:
            if stock in close.columns:
                try:
                    entry_price = close.loc[date, stock]
                    exit_price = close.loc[next_date, stock]
                    if pd.notna(entry_price) and pd.notna(exit_price) and entry_price > 0:
                        ret = (entry_price - exit_price) / entry_price  # 做空收益
                        short_returns.append(ret)
                except:
                    pass
        
        # 组合收益
        if long_returns:
            long_avg = np.mean(long_returns)
        else:
            long_avg = 0
        
        if short_returns:
            short_avg = np.mean(short_returns)
        else:
            short_avg = 0
        
        if allow_short:
            period_return = (long_avg + short_avg) / 2  # 多空对冲
        else:
            period_return = long_avg
        
        # 更新组合价值
        new_value = portfolio_values[-1] * (1 + period_return)
        portfolio_values.append(new_value)
        portfolio_dates.append(next_date)
        
        # 记录交易
        all_trades.append({
            'date': date,
            'next_date': next_date,
            'long_stocks': long_stocks,
            'short_stocks': short_stocks,
            'long_return': long_avg * 100,
            'short_return': short_avg * 100 if allow_short else 0,
            'period_return': period_return * 100,
            'portfolio_value': new_value,
        })
    
    # 输出结果
    print("\n" + "-" * 70)
    print("最近 10 次调仓:")
    print("-" * 70)
    print(f"{'日期':<12} {'做多收益':>10} {'做空收益':>10} {'组合收益':>10} {'组合净值':>12}")
    print("-" * 70)
    
    for t in all_trades[-10:]:
        date_str = str(t['date'])[:10]
        print(f"{date_str:<12} {t['long_return']:>+9.2f}% {t['short_return']:>+9.2f}% "
              f"{t['period_return']:>+9.2f}% ${t['portfolio_value']:>10,.2f}")
    
    # 统计
    print("\n" + "=" * 70)
    print("回测统计")
    print("=" * 70)
    
    returns = [t['period_return'] for t in all_trades]
    long_returns = [t['long_return'] for t in all_trades]
    
    total_return = (portfolio_values[-1] / initial_capital - 1) * 100
    win_rate = len([r for r in returns if r > 0]) / len(returns) * 100
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe = avg_return / std_return * np.sqrt(252 / holding_days) if std_return > 0 else 0
    
    # 最大回撤
    peak = portfolio_values[0]
    max_drawdown = 0
    for v in portfolio_values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_drawdown:
            max_drawdown = dd
    
    # 计算实际年数和复利年化收益
    total_days = len(all_trades) * holding_days
    years = total_days / 252
    final_multiple = portfolio_values[-1] / initial_capital
    annualized_return = (final_multiple ** (1 / years) - 1) * 100 if years > 0 else 0
    
    print(f"总收益率: {total_return:+.2f}%")
    print(f"年化收益: {annualized_return:+.2f}%")
    print(f"调仓次数: {len(all_trades)}")
    print(f"胜率: {win_rate:.1f}%")
    print(f"平均每期收益: {avg_return:+.2f}% ± {std_return:.2f}%")
    print(f"年化 Sharpe: {sharpe:.2f}")
    print(f"最大回撤: {max_drawdown*100:.2f}%")
    print(f"最终资金: ${portfolio_values[-1]:,.2f}")
    
    # 按年统计
    print("\n" + "-" * 70)
    print("按年统计:")
    trades_df = pd.DataFrame(all_trades)
    trades_df['year'] = pd.to_datetime(trades_df['date']).dt.year
    
    for year, group in trades_df.groupby('year'):
        year_return = (1 + group['period_return']/100).prod() - 1
        year_win_rate = (group['period_return'] > 0).mean() * 100
        print(f"  {year}: {len(group)} 期, 胜率 {year_win_rate:.0f}%, 收益 {year_return*100:+.1f}%")
    
    return {
        'trades': all_trades,
        'portfolio_values': portfolio_values,
        'portfolio_dates': portfolio_dates,
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
    }


def plot_cross_sectional_backtest(result, save_path=None):
    """绘制截面回测图表"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    portfolio_values = result['portfolio_values']
    portfolio_dates = result['portfolio_dates']
    trades = result['trades']
    
    initial = portfolio_values[0]
    total_return = result['total_return']
    sharpe = result['sharpe']
    max_dd = result['max_drawdown'] * 100
    
    fig.suptitle(f'GP 因子截面选股回测\n总收益: {total_return:+.1f}%, Sharpe: {sharpe:.2f}, 最大回撤: {max_dd:.1f}%', 
                 fontsize=14, fontweight='bold')
    
    # 图1: 资金曲线
    ax1 = axes[0]
    ax1.plot(portfolio_dates, portfolio_values, color='steelblue', linewidth=2)
    ax1.axhline(y=initial, color='red', linestyle='--', linewidth=1, label='初始资金')
    ax1.fill_between(portfolio_dates, initial, portfolio_values, alpha=0.3,
                     color='green' if portfolio_values[-1] >= initial else 'red')
    ax1.set_ylabel('组合价值 ($)')
    ax1.set_title('资金曲线')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 图2: 每期收益分布
    ax2 = axes[1]
    returns = [t['period_return'] for t in trades]
    colors = ['green' if r > 0 else 'red' for r in returns]
    ax2.bar(range(len(returns)), returns, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=np.mean(returns), color='blue', linestyle='--', linewidth=1,
                label=f'平均: {np.mean(returns):+.2f}%')
    ax2.set_ylabel('收益率 (%)')
    ax2.set_xlabel('调仓期数')
    ax2.set_title(f'每期收益分布 (胜率: {len([r for r in returns if r > 0])/len(returns)*100:.1f}%)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存至 {save_path}")
    
    plt.show()
    return fig


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='GP 因子截面选股回测')
    parser.add_argument('--pool', '-p', default='nasdaq100', 
                        choices=['nasdaq100', 'bluechip', 'dow30', 'sp500_non_tech', 'out_of_sample', 'all'])
    parser.add_argument('--factor', '-f', default='factor_1',
                        choices=['factor_1', 'factor_2', 'factor_3'])
    parser.add_argument('--holding', '-d', type=int, default=5, help='持有天数')
    parser.add_argument('--top', '-t', type=float, default=0.2, help='做多比例')
    parser.add_argument('--capital', '-c', type=float, default=100000, help='初始资金')
    parser.add_argument('--short', action='store_true', help='允许做空')
    parser.add_argument('--start', '-s', default=None, help='开始日期')
    parser.add_argument('--end', '-e', default=None, help='结束日期')
    parser.add_argument('--plot', action='store_true', help='绘制图表')
    parser.add_argument('--save', action='store_true', help='保存图表')
    
    args = parser.parse_args()
    
    result = backtest_cross_sectional(
        pool_type=args.pool,
        factor_key=args.factor,
        holding_days=args.holding,
        top_pct=args.top,
        initial_capital=args.capital,
        allow_short=args.short,
        start_date=args.start,
        end_date=args.end,
    )
    
    if result and (args.plot or args.save):
        save_path = 'reports/gp_cross_sectional_backtest.png' if args.save else None
        plot_cross_sectional_backtest(result, save_path)
