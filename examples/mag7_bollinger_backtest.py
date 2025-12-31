"""
七巨头 (Magnificent 7) 布林带下轨策略回测
买入条件：价格低于布林带下轨1%以上
卖出条件：止损10% / 止盈10%
回测近5年数据
"""

import sys
sys.path.insert(0, '/Users/windye/PycharmProjects/FUTU_auto')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

from src.data.data_fetcher import DataFetcher
from src.data.data_processor import DataProcessor
from src.strategies.bollinger_lower_strategy import BollingerLowerStrategy

# 七巨头股票
MAG7_STOCKS = {
    'AAPL': '苹果',
    'MSFT': '微软',
    'GOOGL': '谷歌',
    'AMZN': '亚马逊',
    'NVDA': '英伟达',
    'META': 'Meta',
    'TSLA': '特斯拉'
}


def run_single_backtest(symbol: str,
                        initial_capital: float = 100000,
                        position_size: float = 100000,
                        below_threshold: float = 0.01,
                        stop_loss: float = 0.10,
                        take_profit: float = 0.10,
                        verbose: bool = True):
    """
    对单个股票运行回测
    """
    
    # 计算近5年日期
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    # 获取数据
    fetcher = DataFetcher()
    processor = DataProcessor()
    
    try:
        data = fetcher.fetch_stock_data(symbol, start_date, end_date)
        data = processor.add_technical_indicators(data, ['bollinger', 'sma'])
    except Exception as e:
        print(f"获取 {symbol} 数据失败: {e}")
        return None, None, None
    
    # 初始化策略
    strategy = BollingerLowerStrategy(
        bb_period=20,
        below_threshold=below_threshold,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit
    )
    
    # 生成信号
    data = strategy.generate_signals(data)
    
    # 回测逻辑
    capital = initial_capital
    position = 0
    entry_price = 0
    entry_date = None
    
    trades = []
    equity_curve = []
    
    bb_lower_col = 'BB_Lower_20'
    
    for i in range(20, len(data)):
        current_date = data.index[i]
        current_price = data.iloc[i]['Close']
        bb_lower = data.iloc[i][bb_lower_col]
        
        # 计算当前权益
        if position > 0:
            current_equity = capital + position * current_price
        else:
            current_equity = capital
        
        equity_curve.append({
            'Date': current_date,
            'Equity': current_equity,
            'Price': current_price
        })
        
        # 如果有持仓，检查止损止盈
        if position > 0:
            should_exit, reason = strategy.check_exit_conditions(entry_price, current_price)
            
            if should_exit:
                pnl = (current_price - entry_price) * position
                pnl_pct = (current_price - entry_price) / entry_price * 100
                capital += position * current_price
                
                trades.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': current_date,
                    'Entry_Price': entry_price,
                    'Exit_Price': current_price,
                    'Shares': position,
                    'PnL': pnl,
                    'PnL_Pct': pnl_pct,
                    'Reason': reason,
                    'Holding_Days': (current_date - entry_date).days
                })
                
                if verbose:
                    print(f"  卖出: {current_date.strftime('%Y-%m-%d')} @ ${current_price:.2f} | "
                          f"盈亏: ${pnl:+,.2f} ({pnl_pct:+.2f}%) | {reason}")
                
                position = 0
                entry_price = 0
                entry_date = None
        
        # 如果没有持仓，检查买入信号
        elif position == 0:
            below_pct = (bb_lower - current_price) / bb_lower
            
            if below_pct >= below_threshold:
                buy_amount = min(position_size, capital)
                shares = int(buy_amount / current_price)
                if shares > 0:
                    cost = shares * current_price
                    capital -= cost
                    position = shares
                    entry_price = current_price
                    entry_date = current_date
                    
                    if verbose:
                        print(f"  买入: {current_date.strftime('%Y-%m-%d')} @ ${current_price:.2f} | "
                              f"数量: {shares} | 低于下轨: {below_pct*100:.2f}%")
    
    # 如果还有持仓，在最后一天平仓
    if position > 0:
        final_price = data.iloc[-1]['Close']
        final_date = data.index[-1]
        pnl = (final_price - entry_price) * position
        pnl_pct = (final_price - entry_price) / entry_price * 100
        capital += position * final_price
        
        trades.append({
            'Entry_Date': entry_date,
            'Exit_Date': final_date,
            'Entry_Price': entry_price,
            'Exit_Price': final_price,
            'Shares': position,
            'PnL': pnl,
            'PnL_Pct': pnl_pct,
            'Reason': '回测结束平仓',
            'Holding_Days': (final_date - entry_date).days
        })
        
        if verbose:
            print(f"  平仓: {final_date.strftime('%Y-%m-%d')} @ ${final_price:.2f} | "
                  f"盈亏: ${pnl:+,.2f} ({pnl_pct:+.2f}%) | 回测结束")
        position = 0
    
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_df = pd.DataFrame(equity_curve)
    
    # 计算统计数据
    final_capital = capital
    total_return = (final_capital - initial_capital) / initial_capital * 100
    
    stats = {
        'symbol': symbol,
        'name': MAG7_STOCKS.get(symbol, symbol),
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_return': total_return,
        'total_trades': len(trades),
        'winning_trades': len(trades_df[trades_df['PnL'] > 0]) if len(trades_df) > 0 else 0,
        'losing_trades': len(trades_df[trades_df['PnL'] < 0]) if len(trades_df) > 0 else 0,
        'win_rate': len(trades_df[trades_df['PnL'] > 0]) / len(trades) * 100 if len(trades) > 0 else 0,
        'avg_win': trades_df[trades_df['PnL'] > 0]['PnL_Pct'].mean() if len(trades_df[trades_df['PnL'] > 0]) > 0 else 0,
        'avg_loss': trades_df[trades_df['PnL'] < 0]['PnL_Pct'].mean() if len(trades_df[trades_df['PnL'] < 0]) > 0 else 0,
        'avg_holding_days': trades_df['Holding_Days'].mean() if len(trades_df) > 0 else 0,
        'max_drawdown': 0
    }
    
    # 计算最大回撤
    if len(equity_df) > 0:
        equity_df['Peak'] = equity_df['Equity'].cummax()
        equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak'] * 100
        stats['max_drawdown'] = equity_df['Drawdown'].min()
    
    return stats, trades_df, equity_df


def run_mag7_backtest():
    """
    对七巨头所有股票运行回测
    """
    
    print("=" * 80)
    print("七巨头 (Magnificent 7) 布林带下轨策略回测")
    print("=" * 80)
    print(f"策略参数:")
    print(f"  - 买入条件: 价格低于布林带下轨 1% 以上")
    print(f"  - 每次买入金额: $100,000")
    print(f"  - 止损: 20%")
    print(f"  - 止盈: 20%")
    print(f"  - 回测期间: 近5年")
    print("=" * 80)
    
    all_stats = []
    all_trades = {}
    all_equity = {}
    
    for symbol, name in MAG7_STOCKS.items():
        print(f"\n{'='*60}")
        print(f"回测 {symbol} ({name})")
        print(f"{'='*60}")
        
        stats, trades_df, equity_df = run_single_backtest(
            symbol=symbol,
            initial_capital=100000,
            position_size=100000,
            below_threshold=0.01,
            stop_loss=0.20,
            take_profit=0.20,
            verbose=True
        )
        
        if stats:
            all_stats.append(stats)
            all_trades[symbol] = trades_df
            all_equity[symbol] = equity_df
    
    # 汇总结果
    summary_df = pd.DataFrame(all_stats)
    
    print("\n" + "=" * 80)
    print("七巨头回测汇总")
    print("=" * 80)
    print(f"\n{'股票':<8} {'名称':<8} {'总收益%':>10} {'交易次数':>8} {'胜率%':>8} {'平均盈利%':>10} {'平均亏损%':>10} {'最大回撤%':>10}")
    print("-" * 85)
    
    for _, row in summary_df.iterrows():
        print(f"{row['symbol']:<8} {row['name']:<8} {row['total_return']:>+10.2f} "
              f"{row['total_trades']:>8} {row['win_rate']:>8.1f} "
              f"{row['avg_win']:>+10.2f} {row['avg_loss']:>+10.2f} {row['max_drawdown']:>10.2f}")
    
    print("-" * 85)
    print(f"{'平均':<17} {summary_df['total_return'].mean():>+10.2f} "
          f"{summary_df['total_trades'].mean():>8.1f} {summary_df['win_rate'].mean():>8.1f} "
          f"{summary_df['avg_win'].mean():>+10.2f} {summary_df['avg_loss'].mean():>+10.2f} "
          f"{summary_df['max_drawdown'].mean():>10.2f}")
    
    # 创建汇总图表
    create_summary_chart(summary_df, all_equity)
    
    # 导出交易明细到CSV
    export_trades_to_csv(all_trades, summary_df)
    
    return summary_df, all_trades, all_equity


def export_trades_to_csv(all_trades, summary_df):
    """导出交易明细到CSV"""
    
    # 合并所有交易记录
    all_trades_list = []
    for symbol, trades_df in all_trades.items():
        if len(trades_df) > 0:
            trades_df = trades_df.copy()
            trades_df['Symbol'] = symbol
            trades_df['Name'] = MAG7_STOCKS.get(symbol, symbol)
            all_trades_list.append(trades_df)
    
    if all_trades_list:
        combined_trades = pd.concat(all_trades_list, ignore_index=True)
        
        # 重新排列列顺序
        columns = ['Symbol', 'Name', 'Entry_Date', 'Exit_Date', 'Entry_Price', 
                   'Exit_Price', 'Shares', 'PnL', 'PnL_Pct', 'Holding_Days', 'Reason']
        combined_trades = combined_trades[columns]
        
        # 格式化日期
        combined_trades['Entry_Date'] = pd.to_datetime(combined_trades['Entry_Date']).dt.strftime('%Y-%m-%d')
        combined_trades['Exit_Date'] = pd.to_datetime(combined_trades['Exit_Date']).dt.strftime('%Y-%m-%d')
        
        # 保存合并的交易明细
        trades_path = '/Users/windye/PycharmProjects/FUTU_auto/reports/mag7_trades_detail.csv'
        combined_trades.to_csv(trades_path, index=False, encoding='utf-8-sig')
        print(f"\n交易明细已保存至: {trades_path}")
        
        # 保存汇总统计
        summary_path = '/Users/windye/PycharmProjects/FUTU_auto/reports/mag7_summary.csv'
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"汇总统计已保存至: {summary_path}")


def create_summary_chart(summary_df, all_equity):
    """创建汇总图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('七巨头 (Magnificent 7) 布林带下轨策略回测结果', fontsize=14, fontweight='bold')
    
    # 图1: 总收益对比
    ax1 = axes[0, 0]
    colors = ['green' if x > 0 else 'red' for x in summary_df['total_return']]
    bars = ax1.bar(summary_df['symbol'], summary_df['total_return'], color=colors, alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axhline(y=summary_df['total_return'].mean(), color='blue', linestyle='--', 
                label=f'平均: {summary_df["total_return"].mean():.1f}%')
    ax1.set_ylabel('总收益 (%)')
    ax1.set_title('各股票总收益对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, val in zip(bars, summary_df['total_return']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 图2: 胜率对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(summary_df['symbol'], summary_df['win_rate'], color='steelblue', alpha=0.8)
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50%基准')
    ax2.axhline(y=summary_df['win_rate'].mean(), color='green', linestyle='--', 
                label=f'平均: {summary_df["win_rate"].mean():.1f}%')
    ax2.set_ylabel('胜率 (%)')
    ax2.set_title('各股票胜率对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    for bar, val in zip(bars2, summary_df['win_rate']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 图3: 权益曲线对比
    ax3 = axes[1, 0]
    for symbol, equity_df in all_equity.items():
        if len(equity_df) > 0:
            # 归一化到100
            normalized = equity_df['Equity'] / equity_df['Equity'].iloc[0] * 100
            ax3.plot(equity_df['Date'], normalized, label=symbol, linewidth=1)
    
    ax3.axhline(y=100, color='black', linestyle='--', alpha=0.5)
    ax3.set_ylabel('归一化权益 (起始=100)')
    ax3.set_xlabel('日期')
    ax3.set_title('权益曲线对比')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 图4: 交易次数与最大回撤
    ax4 = axes[1, 1]
    x = np.arange(len(summary_df))
    width = 0.35
    
    bars3 = ax4.bar(x - width/2, summary_df['total_trades'], width, label='交易次数', color='steelblue', alpha=0.8)
    ax4_twin = ax4.twinx()
    bars4 = ax4_twin.bar(x + width/2, -summary_df['max_drawdown'], width, label='最大回撤', color='red', alpha=0.6)
    
    ax4.set_ylabel('交易次数')
    ax4_twin.set_ylabel('最大回撤 (%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(summary_df['symbol'])
    ax4.set_title('交易次数与最大回撤')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = '/Users/windye/PycharmProjects/FUTU_auto/reports/mag7_bollinger_backtest.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存至: {output_path}")
    
    plt.show()


if __name__ == '__main__':
    summary, trades, equity = run_mag7_backtest()
