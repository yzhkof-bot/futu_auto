"""
Top 5 高频因子组合策略

基于买点分析结果，选取出现频率最高的5个因子：
1. MACD正值 (95.9%)
2. 强趋势 ADX>25 (84.1%)
3. MACD多头 MACD>Signal (82.4%)
4. 强动量 10日涨幅>5% (66.3%)
5. 均线多头排列 MA5>MA10>MA20>MA50 (60.3%)

买点定义：同时满足4个或以上因子
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.data.data_processor import DataProcessor

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 纳斯达克100成分股
NASDAQ100 = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'COST',
    'NFLX', 'TMUS', 'ASML', 'CSCO', 'ADBE', 'AMD', 'PEP', 'LIN', 'INTC', 'INTU',
    'TXN', 'CMCSA', 'QCOM', 'AMGN', 'AMAT', 'HON', 'ISRG', 'BKNG', 'SBUX', 'VRTX',
    'GILD', 'ADP', 'MDLZ', 'ADI', 'REGN', 'PANW', 'SNPS', 'LRCX', 'KLAC', 'CDNS',
    'MU', 'MELI', 'PYPL', 'MAR', 'ORLY', 'MNST', 'CTAS', 'NXPI', 'MCHP', 'FTNT',
    'ABNB', 'PCAR', 'KDP', 'AEP', 'PAYX', 'KHC', 'ODFL', 'CPRT', 'CHTR', 'ROST',
    'IDXX', 'DXCM', 'FAST', 'AZN', 'MRNA', 'EA', 'CTSH', 'EXC', 'VRSK', 'CSGP',
    'XEL', 'BKR', 'GEHC', 'FANG', 'TTWO', 'ANSS', 'BIIB', 'ON', 'DLTR', 'WBD',
    'CDW', 'ZS', 'ILMN', 'MDB', 'TEAM', 'DDOG', 'GFS', 'WBA', 'LCID', 'SIRI',
    'CEG', 'CRWD', 'DASH', 'SMCI', 'ARM', 'COIN', 'TTD', 'PDD', 'LULU', 'WDAY'
]


class Top5FactorStrategy:
    """
    Top 5 高频因子组合策略
    
    因子：
    1. MACD_positive: MACD > 0
    2. Strong_trend: ADX > 25
    3. MACD_bullish: MACD > MACD_Signal
    4. Strong_momentum: 10日涨幅 > 5%
    5. MA_bullish_alignment: MA5 > MA10 > MA20 > MA50
    
    买点：满足 >= 4 个因子
    """
    
    def __init__(self, min_factors: int = 4):
        """
        Args:
            min_factors: 最少需要满足的因子数量（默认4个）
        """
        self.min_factors = min_factors
        self.processor = DataProcessor()
        self.factor_names = [
            'MACD_positive',
            'Strong_trend', 
            'MACD_bullish',
            'Strong_momentum',
            'MA_bullish_alignment'
        ]
    
    def calculate_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算5个因子"""
        df = data.copy()
        
        # 添加技术指标
        df = self.processor.add_technical_indicators(df)
        df = self.processor.add_regime_indicators(df)
        
        # 因子1: MACD正值
        df['F1_MACD_positive'] = df['MACD'] > 0
        
        # 因子2: 强趋势 ADX > 25
        df['F2_Strong_trend'] = df['ADX'] > 25
        
        # 因子3: MACD多头 (MACD > Signal)
        df['F3_MACD_bullish'] = df['MACD'] > df['MACD_Signal']
        
        # 因子4: 强动量 (10日涨幅 > 5%)
        df['Momentum_10'] = df['Close'].pct_change(periods=10)
        df['F4_Strong_momentum'] = df['Momentum_10'] > 0.05
        
        # 因子5: 均线多头排列
        df['F5_MA_bullish'] = (
            (df['SMA_5'] > df['SMA_10']) & 
            (df['SMA_10'] > df['SMA_20']) & 
            (df['SMA_20'] > df['SMA_50'])
        )
        
        # 计算满足的因子数量
        factor_cols = ['F1_MACD_positive', 'F2_Strong_trend', 'F3_MACD_bullish', 
                       'F4_Strong_momentum', 'F5_MA_bullish']
        df['Factor_Count'] = df[factor_cols].sum(axis=1)
        
        # 生成买入信号
        df['Buy_Signal'] = df['Factor_Count'] >= self.min_factors
        
        return df
    
    def find_signals(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """查找股票的买入信号"""
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if data.empty or len(data) < 60:
                return pd.DataFrame()
            
            df = self.calculate_factors(data)
            df['Symbol'] = symbol
            
            return df
        except Exception as e:
            print(f"处理 {symbol} 出错: {e}")
            return pd.DataFrame()
    
    def backtest(self, symbol: str, data: pd.DataFrame, 
                 hold_days: int = 42, stop_loss: float = 0.10) -> dict:
        """
        回测单只股票
        
        Args:
            symbol: 股票代码
            data: 带因子的DataFrame
            hold_days: 持有天数（约2个月）
            stop_loss: 止损比例
        
        Returns:
            回测结果字典
        """
        trades = []
        
        # 找出所有买入信号
        buy_signals = data[data['Buy_Signal']].index.tolist()
        
        i = 0
        while i < len(buy_signals):
            entry_date = buy_signals[i]
            entry_idx = data.index.get_loc(entry_date)
            entry_price = data.loc[entry_date, 'Close']
            factor_count = data.loc[entry_date, 'Factor_Count']
            
            # 检查是否有足够的后续数据
            if entry_idx + hold_days >= len(data):
                break
            
            # 获取持有期间数据
            hold_data = data.iloc[entry_idx:entry_idx + hold_days + 1]
            
            # 检查是否触发止损
            min_price = hold_data['Low'].min()
            max_drawdown = (entry_price - min_price) / entry_price
            
            if max_drawdown > stop_loss:
                # 触发止损，找到止损日期
                stop_loss_price = entry_price * (1 - stop_loss)
                for idx, row in hold_data.iterrows():
                    if row['Low'] <= stop_loss_price:
                        exit_date = idx
                        exit_price = stop_loss_price
                        break
            else:
                # 正常持有到期
                exit_date = hold_data.index[-1]
                exit_price = hold_data['Close'].iloc[-1]
            
            # 计算收益
            returns = (exit_price - entry_price) / entry_price
            
            trades.append({
                'symbol': symbol,
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': exit_date,
                'exit_price': exit_price,
                'returns': returns,
                'factor_count': factor_count,
                'stopped_out': max_drawdown > stop_loss
            })
            
            # 跳过持有期间的信号
            exit_idx = data.index.get_loc(exit_date)
            while i < len(buy_signals) and data.index.get_loc(buy_signals[i]) <= exit_idx:
                i += 1
        
        return trades


def run_full_backtest():
    """运行完整回测"""
    print("=" * 70)
    print("Top 5 高频因子组合策略回测")
    print("=" * 70)
    print("\n策略因子（出现频率最高的5个）：")
    print("  1. MACD正值 (95.9%)")
    print("  2. 强趋势 ADX>25 (84.1%)")
    print("  3. MACD多头 MACD>Signal (82.4%)")
    print("  4. 强动量 10日涨幅>5% (66.3%)")
    print("  5. 均线多头排列 (60.3%)")
    print("\n买点定义：同时满足 >= 4 个因子")
    print("持有期：42个交易日（约2个月）")
    print("止损：10%")
    print("=" * 70)
    
    strategy = Top5FactorStrategy(min_factors=4)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    print(f"\n回测区间: {start_date} 至 {end_date}")
    print(f"股票池: 纳斯达克100 ({len(NASDAQ100)}只)")
    print("\n开始回测...\n")
    
    all_trades = []
    symbol_stats = {}
    
    for i, symbol in enumerate(NASDAQ100):
        print(f"[{i+1}/{len(NASDAQ100)}] 回测 {symbol}...", end=' ')
        
        df = strategy.find_signals(symbol, start_date, end_date)
        if df.empty:
            print("数据不足")
            continue
        
        trades = strategy.backtest(symbol, df)
        
        if trades:
            all_trades.extend(trades)
            win_trades = [t for t in trades if t['returns'] > 0]
            symbol_stats[symbol] = {
                'total': len(trades),
                'wins': len(win_trades),
                'win_rate': len(win_trades) / len(trades) if trades else 0,
                'avg_return': np.mean([t['returns'] for t in trades])
            }
            print(f"{len(trades)}笔交易, 胜率{symbol_stats[symbol]['win_rate']*100:.1f}%")
        else:
            print("无交易")
    
    if not all_trades:
        print("\n无交易记录")
        return
    
    # 汇总统计
    print("\n" + "=" * 70)
    print("回测结果汇总")
    print("=" * 70)
    
    total_trades = len(all_trades)
    win_trades = [t for t in all_trades if t['returns'] > 0]
    loss_trades = [t for t in all_trades if t['returns'] <= 0]
    stopped_trades = [t for t in all_trades if t['stopped_out']]
    
    returns = [t['returns'] for t in all_trades]
    win_returns = [t['returns'] for t in win_trades]
    loss_returns = [t['returns'] for t in loss_trades]
    
    print(f"\n总交易次数: {total_trades}")
    print(f"盈利交易: {len(win_trades)} ({len(win_trades)/total_trades*100:.1f}%)")
    print(f"亏损交易: {len(loss_trades)} ({len(loss_trades)/total_trades*100:.1f}%)")
    print(f"止损触发: {len(stopped_trades)} ({len(stopped_trades)/total_trades*100:.1f}%)")
    
    print(f"\n平均收益率: {np.mean(returns)*100:.2f}%")
    print(f"收益率中位数: {np.median(returns)*100:.2f}%")
    print(f"收益率标准差: {np.std(returns)*100:.2f}%")
    
    if win_returns:
        print(f"\n平均盈利: {np.mean(win_returns)*100:.2f}%")
    if loss_returns:
        print(f"平均亏损: {np.mean(loss_returns)*100:.2f}%")
    
    # 盈亏比
    if win_returns and loss_returns:
        profit_loss_ratio = abs(np.mean(win_returns) / np.mean(loss_returns))
        print(f"盈亏比: {profit_loss_ratio:.2f}")
    
    # 期望收益
    win_rate = len(win_trades) / total_trades
    expected_return = win_rate * np.mean(win_returns) + (1 - win_rate) * np.mean(loss_returns) if win_returns and loss_returns else 0
    print(f"期望收益: {expected_return*100:.2f}%")
    
    # 按因子数量分组统计
    print("\n" + "-" * 40)
    print("按因子数量分组统计:")
    print("-" * 40)
    
    for fc in [4, 5]:
        fc_trades = [t for t in all_trades if t['factor_count'] == fc]
        if fc_trades:
            fc_wins = [t for t in fc_trades if t['returns'] > 0]
            fc_returns = [t['returns'] for t in fc_trades]
            print(f"\n{fc}个因子满足:")
            print(f"  交易数: {len(fc_trades)}")
            print(f"  胜率: {len(fc_wins)/len(fc_trades)*100:.1f}%")
            print(f"  平均收益: {np.mean(fc_returns)*100:.2f}%")
    
    # 表现最好的股票
    print("\n" + "-" * 40)
    print("表现最好的股票（交易数>=5）:")
    print("-" * 40)
    
    good_symbols = {k: v for k, v in symbol_stats.items() if v['total'] >= 5}
    sorted_symbols = sorted(good_symbols.items(), key=lambda x: x[1]['avg_return'], reverse=True)
    
    for symbol, stats in sorted_symbols[:10]:
        print(f"  {symbol}: {stats['total']}笔, 胜率{stats['win_rate']*100:.1f}%, 平均收益{stats['avg_return']*100:.2f}%")
    
    # 保存交易记录
    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv('reports/top5_factor_trades.csv', index=False)
    print(f"\n交易记录已保存至 reports/top5_factor_trades.csv")
    
    # 绘制图表
    plot_results(all_trades, symbol_stats)
    
    return all_trades, symbol_stats


def plot_results(trades: list, symbol_stats: dict):
    """绘制回测结果图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Top 5 Factor Strategy Backtest Results\n(Min 4 factors, 42-day hold, 10% stop-loss)', fontsize=14)
    
    # 1. 收益分布
    ax1 = axes[0, 0]
    returns = [t['returns'] * 100 for t in trades]
    ax1.hist(returns, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.axvline(np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.1f}%')
    ax1.axvline(np.median(returns), color='orange', linestyle='--', label=f'Median: {np.median(returns):.1f}%')
    ax1.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax1.set_xlabel('Return (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Return Distribution')
    ax1.legend()
    
    # 2. 按因子数量的胜率和收益
    ax2 = axes[0, 1]
    factor_groups = {4: [], 5: []}
    for t in trades:
        fc = int(t['factor_count'])
        if fc in factor_groups:
            factor_groups[fc].append(t['returns'])
    
    x = list(factor_groups.keys())
    win_rates = [len([r for r in factor_groups[fc] if r > 0]) / len(factor_groups[fc]) * 100 
                 if factor_groups[fc] else 0 for fc in x]
    avg_returns = [np.mean(factor_groups[fc]) * 100 if factor_groups[fc] else 0 for fc in x]
    
    width = 0.35
    ax2.bar([i - width/2 for i in range(len(x))], win_rates, width, label='Win Rate (%)', color='green', alpha=0.7)
    ax2.bar([i + width/2 for i in range(len(x))], avg_returns, width, label='Avg Return (%)', color='blue', alpha=0.7)
    ax2.set_xticks(range(len(x)))
    ax2.set_xticklabels([f'{fc} Factors' for fc in x])
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Performance by Factor Count')
    ax2.legend()
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    # 3. 累计收益曲线
    ax3 = axes[1, 0]
    trades_sorted = sorted(trades, key=lambda x: x['entry_date'])
    cum_returns = np.cumprod([1 + t['returns'] for t in trades_sorted])
    dates = [t['entry_date'] for t in trades_sorted]
    ax3.plot(range(len(cum_returns)), cum_returns, color='steelblue', linewidth=1.5)
    ax3.axhline(1, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Trade Number')
    ax3.set_ylabel('Cumulative Return')
    ax3.set_title(f'Cumulative Return (Final: {cum_returns[-1]:.2f}x)')
    ax3.fill_between(range(len(cum_returns)), 1, cum_returns, 
                     where=cum_returns >= 1, color='green', alpha=0.3)
    ax3.fill_between(range(len(cum_returns)), 1, cum_returns, 
                     where=cum_returns < 1, color='red', alpha=0.3)
    
    # 4. 股票胜率分布
    ax4 = axes[1, 1]
    win_rates_list = [v['win_rate'] * 100 for v in symbol_stats.values() if v['total'] >= 3]
    ax4.hist(win_rates_list, bins=20, color='coral', edgecolor='white', alpha=0.8)
    ax4.axvline(np.mean(win_rates_list), color='red', linestyle='--', 
                label=f'Mean: {np.mean(win_rates_list):.1f}%')
    ax4.axvline(50, color='black', linestyle=':', alpha=0.5, label='50% baseline')
    ax4.set_xlabel('Win Rate (%)')
    ax4.set_ylabel('Number of Stocks')
    ax4.set_title('Win Rate Distribution by Stock (min 3 trades)')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('reports/top5_factor_backtest.png', dpi=150, bbox_inches='tight')
    print("回测图表已保存至 reports/top5_factor_backtest.png")
    plt.show()


def scan_current_signals():
    """扫描当前符合条件的买点"""
    print("\n" + "=" * 70)
    print("扫描当前买入信号")
    print("=" * 70)
    
    strategy = Top5FactorStrategy(min_factors=4)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')
    
    current_signals = []
    
    for symbol in NASDAQ100:
        df = strategy.find_signals(symbol, start_date, end_date)
        if df.empty:
            continue
        
        # 检查最近5天是否有信号
        recent = df.tail(5)
        if recent['Buy_Signal'].any():
            last_signal = recent[recent['Buy_Signal']].iloc[-1]
            signal_date = recent[recent['Buy_Signal']].index[-1]
            
            current_signals.append({
                'symbol': symbol,
                'date': signal_date,
                'price': last_signal['Close'],
                'factor_count': last_signal['Factor_Count'],
                'MACD_positive': last_signal['F1_MACD_positive'],
                'Strong_trend': last_signal['F2_Strong_trend'],
                'MACD_bullish': last_signal['F3_MACD_bullish'],
                'Strong_momentum': last_signal['F4_Strong_momentum'],
                'MA_bullish': last_signal['F5_MA_bullish'],
                'RSI': last_signal.get('RSI_14', None),
                'ADX': last_signal.get('ADX', None)
            })
    
    if current_signals:
        print(f"\n找到 {len(current_signals)} 个近期买入信号:\n")
        signals_df = pd.DataFrame(current_signals)
        signals_df = signals_df.sort_values('factor_count', ascending=False)
        
        for _, row in signals_df.iterrows():
            factors = []
            if row['MACD_positive']: factors.append('MACD+')
            if row['Strong_trend']: factors.append('ADX>25')
            if row['MACD_bullish']: factors.append('MACD>Sig')
            if row['Strong_momentum']: factors.append('Mom>5%')
            if row['MA_bullish']: factors.append('MA多头')
            
            print(f"  {row['symbol']:6} | {str(row['date'])[:10]} | ${row['price']:.2f} | "
                  f"{int(row['factor_count'])}因子 | {', '.join(factors)}")
        
        signals_df.to_csv('reports/current_signals.csv', index=False)
        print(f"\n信号已保存至 reports/current_signals.csv")
    else:
        print("\n当前无符合条件的买入信号")
    
    return current_signals


if __name__ == '__main__':
    # 运行完整回测
    trades, stats = run_full_backtest()
    
    # 扫描当前信号
    signals = scan_current_signals()
