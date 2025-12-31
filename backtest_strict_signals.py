"""
严格组合信号策略回测
要求多个信号同时出现才买入

测试不同严格程度：
1. 至少2个信号同时出现
2. 至少3个信号同时出现
3. 全部4个信号同时出现
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 纳斯达克100成分股
NASDAQ_100 = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'GOOG', 'AVGO', 'COST',
    'PEP', 'CSCO', 'ADBE', 'NFLX', 'AMD', 'CMCSA', 'INTC', 'INTU', 'QCOM', 'TXN',
    'AMGN', 'HON', 'AMAT', 'BKNG', 'ISRG', 'SBUX', 'MDLZ', 'GILD', 'ADI', 'VRTX',
    'ADP', 'REGN', 'LRCX', 'PANW', 'MU', 'SNPS', 'KLAC', 'CDNS', 'MELI', 'ASML',
    'PYPL', 'MAR', 'ORLY', 'MNST', 'CTAS', 'MRVL', 'ABNB', 'FTNT', 'NXPI', 'KDP',
    'LULU', 'PCAR', 'WDAY', 'CPRT', 'ROST', 'PAYX', 'AEP', 'ODFL', 'KHC', 'MCHP',
    'IDXX', 'DXCM', 'EXC', 'FAST', 'VRSK', 'EA', 'CTSH', 'XEL', 'BKR', 'GEHC',
    'CSGP', 'FANG', 'ON', 'DLTR', 'WBD', 'ZS', 'ILMN', 'ALGN', 'TTWO',
    'SIRI', 'JD', 'LCID', 'RIVN', 'DDOG', 'TEAM', 'CRWD', 'ZM', 'DOCU',
    'OKTA', 'SNOW', 'NET', 'MDB', 'COIN', 'HOOD', 'RBLX', 'U', 'PATH'
]


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
        return pd.DataFrame()


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标"""
    if len(df) < 60:
        return df
    
    # 均线
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    
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
    df['MACD_below_zero'] = (df['MACD'] < 0).astype(int)
    df['MACD_golden_cross'] = ((df['MACD'] > df['MACD_signal']) & 
                               (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))).astype(int)
    
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


def backtest_stock(symbol: str, start_date: str, end_date: str, 
                   holding_days: int = 20, min_signals: int = 2) -> Dict:
    """回测单只股票"""
    df = load_stock_data(symbol, start_date, end_date)
    if len(df) < 100:
        return None
    
    df = calculate_indicators(df)
    
    trades = []
    last_exit_idx = -1
    
    for i in range(60, len(df) - holding_days):
        if i <= last_exit_idx:
            continue
        
        row = df.iloc[i]
        signal_count, triggered_signals = count_signals(row)
        
        # 检查是否满足最小信号数要求
        if signal_count < min_signals:
            continue
        
        # 计算收益
        entry_idx = i
        exit_idx = min(i + holding_days, len(df) - 1)
        
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
            'signal_count': signal_count,
            'signals': ','.join(triggered_signals),
            'RSI': row['RSI'],
            'K': row['K'],
        })
        
        last_exit_idx = exit_idx
    
    return trades


def run_backtest(min_signals: int, stocks: List[str], start_date: str, 
                 end_date: str, holding_days: int) -> pd.DataFrame:
    """运行回测"""
    all_trades = []
    
    for symbol in stocks:
        trades = backtest_stock(symbol, start_date, end_date, holding_days, min_signals)
        if trades:
            all_trades.extend(trades)
    
    return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()


def main():
    """主函数"""
    print("=" * 70)
    print("严格组合信号策略回测")
    print("=" * 70)
    print("\n4个基础信号：")
    print("  1. 均线空头排列 + RSI超卖(<30)")
    print("  2. RSI超卖(<30) + KDJ超卖(K,D<20)")
    print("  3. 接近60日低点 + KDJ超卖")
    print("  4. MACD零下金叉")
    print("\n测试不同严格程度的效果")
    print("=" * 70)
    
    # 参数设置
    num_stocks = 80
    start_date = "2015-01-01"
    end_date = "2024-12-30"
    holding_days = 20
    
    import random
    random.seed(42)
    selected_stocks = random.sample(NASDAQ_100, min(num_stocks, len(NASDAQ_100)))
    
    print(f"\n分析 {num_stocks} 只股票")
    print(f"回测期间: {start_date} 至 {end_date}")
    print(f"持有天数: {holding_days} 天")
    
    # 测试不同严格程度
    results_summary = []
    all_results = {}
    
    for min_signals in [1, 2, 3, 4]:
        print(f"\n{'='*70}")
        print(f"测试: 至少 {min_signals} 个信号同时出现")
        print("-" * 70)
        
        trades_df = run_backtest(min_signals, selected_stocks, start_date, end_date, holding_days)
        
        if len(trades_df) == 0:
            print(f"  未找到符合条件的交易")
            results_summary.append({
                'min_signals': min_signals,
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'median_return': 0,
            })
            continue
        
        all_results[min_signals] = trades_df
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
        win_rate = winning_trades / total_trades * 100
        avg_return = trades_df['pnl_pct'].mean() * 100
        median_return = trades_df['pnl_pct'].median() * 100
        
        print(f"  交易次数: {total_trades}")
        print(f"  胜率: {win_rate:.2f}%")
        print(f"  平均收益: {avg_return:.2f}%")
        print(f"  中位数收益: {median_return:.2f}%")
        print(f"  最大收益: {trades_df['pnl_pct'].max()*100:.2f}%")
        print(f"  最小收益: {trades_df['pnl_pct'].min()*100:.2f}%")
        
        results_summary.append({
            'min_signals': min_signals,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'median_return': median_return,
            'max_return': trades_df['pnl_pct'].max() * 100,
            'min_return': trades_df['pnl_pct'].min() * 100,
            'std_return': trades_df['pnl_pct'].std() * 100,
        })
        
        # 按年份统计
        if len(trades_df) > 0:
            trades_df['year'] = pd.to_datetime(trades_df['entry_date'], utc=True).dt.year
            yearly = trades_df.groupby('year').agg({
                'pnl_pct': ['count', lambda x: (x > 0).sum() / len(x) * 100, 'mean']
            })
            yearly.columns = ['交易数', '胜率', '平均收益']
            yearly['平均收益'] = yearly['平均收益'] * 100
            
            print(f"\n  按年份统计:")
            for year, row in yearly.iterrows():
                print(f"    {year}: {int(row['交易数']):3d}笔, 胜率{row['胜率']:.1f}%, 平均收益{row['平均收益']:.2f}%")
    
    # 汇总对比
    print("\n" + "=" * 70)
    print("不同严格程度对比汇总")
    print("=" * 70)
    
    summary_df = pd.DataFrame(results_summary)
    print(f"\n{'信号数':>8s} {'交易次数':>10s} {'胜率':>10s} {'平均收益':>10s} {'中位数收益':>12s}")
    print("-" * 55)
    
    for _, row in summary_df.iterrows():
        if row['total_trades'] > 0:
            print(f"{int(row['min_signals']):>8d} {int(row['total_trades']):>10d} "
                  f"{row['win_rate']:>9.2f}% {row['avg_return']:>9.2f}% {row['median_return']:>11.2f}%")
        else:
            print(f"{int(row['min_signals']):>8d} {0:>10d} {'N/A':>10s} {'N/A':>10s} {'N/A':>12s}")
    
    # 保存结果
    os.makedirs('reports', exist_ok=True)
    summary_df.to_csv('reports/strict_signals_summary.csv', index=False)
    
    # 绘制对比图
    plot_comparison(summary_df, all_results)
    
    print(f"\n结果已保存到 reports/strict_signals_summary.csv")


def plot_comparison(summary_df: pd.DataFrame, all_results: Dict):
    """绘制对比图"""
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('不同信号严格程度对比\n(要求N个信号同时出现)', fontsize=14, fontweight='bold')
    
    valid_df = summary_df[summary_df['total_trades'] > 0]
    
    # 1. 交易次数 vs 胜率
    ax1 = axes[0, 0]
    x = valid_df['min_signals'].values
    
    ax1_twin = ax1.twinx()
    bars = ax1.bar(x - 0.15, valid_df['total_trades'], width=0.3, 
                   color='steelblue', alpha=0.7, label='交易次数')
    line = ax1_twin.plot(x, valid_df['win_rate'], 'o-', color='orange', 
                         linewidth=2, markersize=10, label='胜率')
    
    ax1.set_xlabel('最少信号数')
    ax1.set_ylabel('交易次数', color='steelblue')
    ax1_twin.set_ylabel('胜率 (%)', color='orange')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'≥{int(i)}个' for i in x])
    ax1.set_title('交易次数 vs 胜率')
    ax1_twin.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    
    # 添加数值标注
    for i, (trades, wr) in enumerate(zip(valid_df['total_trades'], valid_df['win_rate'])):
        ax1.annotate(f'{int(trades)}', (x[i]-0.15, trades), ha='center', va='bottom', fontsize=10)
        ax1_twin.annotate(f'{wr:.1f}%', (x[i], wr), ha='center', va='bottom', fontsize=10)
    
    # 2. 平均收益对比
    ax2 = axes[0, 1]
    colors = ['green' if r > 0 else 'red' for r in valid_df['avg_return']]
    bars = ax2.bar(x, valid_df['avg_return'], color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax2.set_xlabel('最少信号数')
    ax2.set_ylabel('平均收益率 (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'≥{int(i)}个' for i in x])
    ax2.set_title('平均收益率对比')
    
    for bar, ret in zip(bars, valid_df['avg_return']):
        ax2.annotate(f'{ret:.2f}%', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom' if ret > 0 else 'top', fontsize=11, fontweight='bold')
    
    # 3. 收益分布对比 (boxplot)
    ax3 = axes[1, 0]
    box_data = []
    labels = []
    for min_sig in sorted(all_results.keys()):
        if len(all_results[min_sig]) > 0:
            box_data.append(all_results[min_sig]['pnl_pct'].values * 100)
            labels.append(f'≥{min_sig}个信号\n(n={len(all_results[min_sig])})')
    
    if box_data:
        bp = ax3.boxplot(box_data, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax3.set_ylabel('收益率 (%)')
        ax3.set_title('收益分布对比 (箱线图)')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 风险收益比
    ax4 = axes[1, 1]
    if 'std_return' in valid_df.columns:
        sharpe_like = valid_df['avg_return'] / valid_df['std_return']
        bars = ax4.bar(x, sharpe_like, color='purple', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('最少信号数')
        ax4.set_ylabel('收益/风险比')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'≥{int(i)}个' for i in x])
        ax4.set_title('收益风险比 (平均收益/标准差)')
        ax4.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        
        for bar, ratio in zip(bars, sharpe_like):
            ax4.annotate(f'{ratio:.2f}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    
    chart_file = 'reports/strict_signals_comparison.png'
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {chart_file}")
    plt.close()


if __name__ == "__main__":
    main()
