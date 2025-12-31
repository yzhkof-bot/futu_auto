"""
组合高频特征策略回测
基于买点特征分析结果，组合出现概率最高的特征

策略条件（满足任一组合即触发）：
1. 均线空头 + RSI超卖 (出现频率10.8%)
2. RSI超卖 + KDJ超卖 (出现频率4.8%)
3. 接近60日低点 + KDJ超卖 (出现频率3.7%)
4. MACD零下金叉 (出现频率2.5%)
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
    
    # 均线排列
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


def check_signals(row) -> Dict[str, bool]:
    """检查各种信号组合"""
    signals = {
        'combo1_ma_down_rsi': False,  # 均线空头 + RSI超卖
        'combo2_rsi_kdj': False,       # RSI超卖 + KDJ超卖
        'combo3_low60_kdj': False,     # 接近60日低点 + KDJ超卖
        'combo4_macd_cross': False,    # MACD零下金叉
    }
    
    # 组合1: 均线空头 + RSI超卖
    if row['ma_trend_down'] == 1 and row['RSI_oversold'] == 1:
        signals['combo1_ma_down_rsi'] = True
    
    # 组合2: RSI超卖 + KDJ超卖
    if row['RSI_oversold'] == 1 and row['KDJ_oversold'] == 1:
        signals['combo2_rsi_kdj'] = True
    
    # 组合3: 接近60日低点 + KDJ超卖
    if row['near_60d_low'] == 1 and row['KDJ_oversold'] == 1:
        signals['combo3_low60_kdj'] = True
    
    # 组合4: MACD零下金叉
    if row['MACD_below_zero'] == 1 and row['MACD_golden_cross'] == 1:
        signals['combo4_macd_cross'] = True
    
    return signals


def backtest_stock(symbol: str, start_date: str, end_date: str, 
                   holding_days: int = 20) -> Dict:
    """回测单只股票"""
    df = load_stock_data(symbol, start_date, end_date)
    if len(df) < 100:
        return None
    
    df = calculate_indicators(df)
    
    trades = []
    last_exit_idx = -1
    
    for i in range(60, len(df) - holding_days):
        # 避免持仓期重叠
        if i <= last_exit_idx:
            continue
        
        row = df.iloc[i]
        signals = check_signals(row)
        
        # 检查是否有任何信号触发
        triggered_signals = [k for k, v in signals.items() if v]
        if not triggered_signals:
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
            'signals': ','.join(triggered_signals),
            'RSI': row['RSI'],
            'K': row['K'],
            'D': row['D'],
            'MACD': row['MACD'],
        })
        
        last_exit_idx = exit_idx
    
    if not trades:
        return None
    
    trades_df = pd.DataFrame(trades)
    winning_trades = trades_df[trades_df['pnl_pct'] > 0]
    
    return {
        'symbol': symbol,
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(trades) - len(winning_trades),
        'win_rate': len(winning_trades) / len(trades) * 100,
        'avg_return': trades_df['pnl_pct'].mean() * 100,
        'median_return': trades_df['pnl_pct'].median() * 100,
        'max_return': trades_df['pnl_pct'].max() * 100,
        'min_return': trades_df['pnl_pct'].min() * 100,
        'avg_days_held': trades_df['days_held'].mean(),
        'trades': trades
    }


def main():
    """主函数"""
    print("=" * 70)
    print("组合高频特征策略回测")
    print("=" * 70)
    print("\n策略买入条件（满足任一即触发）：")
    print("  1. 均线空头排列 + RSI超卖(<30)")
    print("  2. RSI超卖(<30) + KDJ超卖(K,D<20)")
    print("  3. 接近60日低点 + KDJ超卖")
    print("  4. MACD零下金叉")
    print("\n卖出条件：持有1个月（约20个交易日）")
    print("=" * 70)
    
    # 参数设置
    num_stocks = 50
    start_date = "2015-01-01"
    end_date = "2024-12-30"
    holding_days = 20
    
    import random
    random.seed(42)
    selected_stocks = random.sample(NASDAQ_100, min(num_stocks, len(NASDAQ_100)))
    
    print(f"\n随机选择 {num_stocks} 只股票进行回测")
    print(f"回测期间: {start_date} 至 {end_date}")
    print("-" * 70)
    
    # 收集所有交易
    all_trades = []
    all_results = []
    
    for i, symbol in enumerate(selected_stocks):
        print(f"\r处理 {symbol} ({i+1}/{num_stocks})...", end="", flush=True)
        
        result = backtest_stock(symbol, start_date, end_date, holding_days)
        if result:
            all_results.append(result)
            all_trades.extend(result['trades'])
    
    print(f"\n\n找到 {len(all_trades)} 笔交易")
    
    if not all_trades:
        print("未找到符合条件的交易")
        return
    
    trades_df = pd.DataFrame(all_trades)
    
    # 总体统计
    print("\n" + "=" * 70)
    print("总体回测结果")
    print("=" * 70)
    
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
    losing_trades = total_trades - winning_trades
    
    print(f"\n总交易次数: {total_trades}")
    print(f"盈利交易: {winning_trades}")
    print(f"亏损交易: {losing_trades}")
    print(f"总体胜率: {winning_trades/total_trades*100:.2f}%")
    print(f"\n平均收益率: {trades_df['pnl_pct'].mean()*100:.2f}%")
    print(f"中位数收益率: {trades_df['pnl_pct'].median()*100:.2f}%")
    print(f"最大收益率: {trades_df['pnl_pct'].max()*100:.2f}%")
    print(f"最小收益率: {trades_df['pnl_pct'].min()*100:.2f}%")
    print(f"收益率标准差: {trades_df['pnl_pct'].std()*100:.2f}%")
    print(f"\n平均持有天数: {trades_df['days_held'].mean():.1f} 天")
    
    # 按信号类型统计
    print("\n" + "-" * 70)
    print("按信号类型统计")
    print("-" * 70)
    
    signal_names = {
        'combo1_ma_down_rsi': '均线空头+RSI超卖',
        'combo2_rsi_kdj': 'RSI超卖+KDJ超卖',
        'combo3_low60_kdj': '60日低点+KDJ超卖',
        'combo4_macd_cross': 'MACD零下金叉'
    }
    
    # 解析信号
    signal_stats = {k: {'count': 0, 'wins': 0, 'returns': []} for k in signal_names.keys()}
    
    for _, row in trades_df.iterrows():
        for signal in row['signals'].split(','):
            if signal in signal_stats:
                signal_stats[signal]['count'] += 1
                signal_stats[signal]['returns'].append(row['pnl_pct'])
                if row['pnl_pct'] > 0:
                    signal_stats[signal]['wins'] += 1
    
    print(f"\n{'信号类型':25s} {'交易数':>8s} {'胜率':>8s} {'平均收益':>10s}")
    print("-" * 55)
    
    for signal, name in signal_names.items():
        stats = signal_stats[signal]
        if stats['count'] > 0:
            win_rate = stats['wins'] / stats['count'] * 100
            avg_return = np.mean(stats['returns']) * 100
            print(f"{name:25s} {stats['count']:>8d} {win_rate:>7.1f}% {avg_return:>9.2f}%")
    
    # 按年份统计
    print("\n" + "-" * 70)
    print("按年份统计")
    print("-" * 70)
    
    trades_df['year'] = pd.to_datetime(trades_df['entry_date'], utc=True).dt.year
    yearly_stats = trades_df.groupby('year').agg({
        'pnl_pct': ['count', lambda x: (x > 0).sum() / len(x) * 100, 'mean']
    })
    yearly_stats.columns = ['交易次数', '胜率', '平均收益']
    yearly_stats['平均收益'] = yearly_stats['平均收益'] * 100
    
    print(f"\n{'年份':>6s} {'交易次数':>8s} {'胜率':>8s} {'平均收益':>10s}")
    print("-" * 40)
    for year, row in yearly_stats.iterrows():
        print(f"{year:>6d} {int(row['交易次数']):>8d} {row['胜率']:>7.1f}% {row['平均收益']:>9.2f}%")
    
    # 保存结果
    output_file = 'reports/combined_signals_backtest.csv'
    os.makedirs('reports', exist_ok=True)
    trades_df.to_csv(output_file, index=False)
    print(f"\n详细结果已保存到: {output_file}")
    
    # 绘制图表
    plot_results(trades_df, signal_stats, signal_names, yearly_stats, holding_days)


def plot_results(trades_df: pd.DataFrame, signal_stats: Dict, signal_names: Dict,
                 yearly_stats: pd.DataFrame, holding_days: int):
    """绘制结果图表"""
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'组合高频特征策略回测结果\n(持有{holding_days}天)', 
                 fontsize=14, fontweight='bold')
    
    # 1. 各信号胜率对比
    ax1 = axes[0, 0]
    signals = []
    win_rates = []
    counts = []
    for signal, name in signal_names.items():
        stats = signal_stats[signal]
        if stats['count'] > 0:
            signals.append(name[:12])
            win_rates.append(stats['wins'] / stats['count'] * 100)
            counts.append(stats['count'])
    
    colors = ['green' if wr > 55 else 'steelblue' for wr in win_rates]
    bars = ax1.bar(range(len(signals)), win_rates, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=50, color='red', linestyle='--', linewidth=1, label='50%基准')
    ax1.set_xticks(range(len(signals)))
    ax1.set_xticklabels(signals, rotation=15, ha='right', fontsize=9)
    ax1.set_ylabel('胜率 (%)')
    ax1.set_title('各信号组合胜率对比')
    ax1.set_ylim(0, 80)
    ax1.legend()
    
    for bar, wr, cnt in zip(bars, win_rates, counts):
        ax1.annotate(f'{wr:.1f}%\n(n={cnt})',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # 2. 收益分布
    ax2 = axes[0, 1]
    returns = trades_df['pnl_pct'] * 100
    ax2.hist(returns, bins=40, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='盈亏平衡')
    ax2.axvline(x=returns.mean(), color='orange', linestyle='-', linewidth=2,
                label=f'平均: {returns.mean():.2f}%')
    ax2.set_xlabel('收益率 (%)')
    ax2.set_ylabel('交易次数')
    ax2.set_title('收益分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 按年份表现
    ax3 = axes[1, 0]
    x = range(len(yearly_stats))
    ax3.bar(x, yearly_stats['胜率'], color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axhline(y=50, color='red', linestyle='--', linewidth=1, label='50%基准')
    ax3.set_xticks(x)
    ax3.set_xticklabels(yearly_stats.index, rotation=45)
    ax3.set_xlabel('年份')
    ax3.set_ylabel('胜率 (%)')
    ax3.set_title('按年份胜率')
    ax3.set_ylim(0, 80)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    for i, (idx, row) in enumerate(yearly_stats.iterrows()):
        ax3.annotate(f'{row["胜率"]:.0f}%\n(n={int(row["交易次数"])})',
                    xy=(i, row['胜率']),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # 4. 年度平均收益与累计
    ax4 = axes[1, 1]
    x = range(len(yearly_stats))
    cumulative = yearly_stats['平均收益'].cumsum()
    
    ax4.bar(x, yearly_stats['平均收益'], color='steelblue', alpha=0.6, label='年度平均收益')
    ax4.plot(x, cumulative, color='orange', linewidth=2, marker='o', label='累计收益')
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax4.set_xticks(x)
    ax4.set_xticklabels(yearly_stats.index, rotation=45)
    ax4.set_xlabel('年份')
    ax4.set_ylabel('收益率 (%)')
    ax4.set_title('年度平均收益与累计收益')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    chart_file = 'reports/combined_signals_backtest_chart.png'
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {chart_file}")
    plt.close()


if __name__ == "__main__":
    main()
