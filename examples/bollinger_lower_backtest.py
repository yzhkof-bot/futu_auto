"""
布林带下轨策略回测
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


def run_backtest(symbol: str = 'AAPL',
                 initial_capital: float = 100000,
                 position_size: float = 100000,  # 每次买入金额
                 below_threshold: float = 0.01,
                 stop_loss: float = 0.10,
                 take_profit: float = 0.10):
    """
    运行布林带下轨策略回测
    """
    
    # 计算近5年日期
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    print(f"{'='*60}")
    print(f"布林带下轨策略回测 - {symbol}")
    print(f"{'='*60}")
    print(f"回测期间: {start_date} 至 {end_date}")
    print(f"初始资金: ${initial_capital:,.2f}")
    print(f"策略参数:")
    print(f"  - 买入条件: 价格低于布林带下轨 {below_threshold*100:.1f}% 以上")
    print(f"  - 每次买入金额: ${position_size:,.0f}")
    print(f"  - 止损: {stop_loss*100:.1f}%")
    print(f"  - 止盈: {take_profit*100:.1f}%")
    print(f"{'='*60}\n")
    
    # 获取数据
    print("获取数据...")
    fetcher = DataFetcher()
    processor = DataProcessor()
    
    data = fetcher.fetch_stock_data(symbol, start_date, end_date)
    data = processor.add_technical_indicators(data, ['bollinger', 'sma'])
    
    print(f"数据条数: {len(data)}")
    
    # 初始化策略
    strategy = BollingerLowerStrategy(
        bb_period=20,
        below_threshold=below_threshold,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit
    )
    
    # 生成信号
    data = strategy.generate_signals(data)
    
    # 手动回测逻辑
    capital = initial_capital
    position = 0  # 持仓数量
    entry_price = 0
    entry_date = None
    
    trades = []  # 交易记录
    equity_curve = []  # 权益曲线
    
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
                # 卖出
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
                
                print(f"  卖出: {current_date.strftime('%Y-%m-%d')} @ ${current_price:.2f} | "
                      f"盈亏: ${pnl:+,.2f} ({pnl_pct:+.2f}%) | {reason}")
                
                position = 0
                entry_price = 0
                entry_date = None
        
        # 如果没有持仓，检查买入信号
        elif position == 0:
            below_pct = (bb_lower - current_price) / bb_lower
            
            if below_pct >= below_threshold:
                # 买入 (固定金额)
                buy_amount = min(position_size, capital)  # 不超过可用资金
                shares = int(buy_amount / current_price)
                if shares > 0:
                    cost = shares * current_price
                    capital -= cost
                    position = shares
                    entry_price = current_price
                    entry_date = current_date
                    
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
        
        print(f"  平仓: {final_date.strftime('%Y-%m-%d')} @ ${final_price:.2f} | "
              f"盈亏: ${pnl:+,.2f} ({pnl_pct:+.2f}%) | 回测结束")
        position = 0
    
    # 计算统计数据
    final_capital = capital
    total_return = (final_capital - initial_capital) / initial_capital * 100
    
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_df = pd.DataFrame(equity_curve)
    
    print(f"\n{'='*60}")
    print("回测结果")
    print(f"{'='*60}")
    print(f"初始资金: ${initial_capital:,.2f}")
    print(f"最终资金: ${final_capital:,.2f}")
    print(f"总收益: ${final_capital - initial_capital:+,.2f} ({total_return:+.2f}%)")
    print(f"总交易次数: {len(trades)}")
    
    if len(trades) > 0:
        winning_trades = trades_df[trades_df['PnL'] > 0]
        losing_trades = trades_df[trades_df['PnL'] < 0]
        
        win_rate = len(winning_trades) / len(trades) * 100
        avg_win = winning_trades['PnL_Pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['PnL_Pct'].mean() if len(losing_trades) > 0 else 0
        avg_holding = trades_df['Holding_Days'].mean()
        
        print(f"胜率: {win_rate:.2f}%")
        print(f"盈利交易: {len(winning_trades)} 次")
        print(f"亏损交易: {len(losing_trades)} 次")
        print(f"平均盈利: {avg_win:+.2f}%")
        print(f"平均亏损: {avg_loss:+.2f}%")
        print(f"平均持仓天数: {avg_holding:.1f} 天")
        
        # 计算最大回撤
        equity_df['Peak'] = equity_df['Equity'].cummax()
        equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak'] * 100
        max_drawdown = equity_df['Drawdown'].min()
        print(f"最大回撤: {max_drawdown:.2f}%")
        
        # 打印交易明细
        print(f"\n{'='*60}")
        print("交易明细")
        print(f"{'='*60}")
        print(f"{'买入日期':<12} {'卖出日期':<12} {'买入价':>10} {'卖出价':>10} {'盈亏%':>10} {'原因':<15}")
        print("-" * 75)
        for _, trade in trades_df.iterrows():
            entry_str = trade['Entry_Date'].strftime('%Y-%m-%d')
            exit_str = trade['Exit_Date'].strftime('%Y-%m-%d')
            print(f"{entry_str:<12} {exit_str:<12} ${trade['Entry_Price']:>9.2f} ${trade['Exit_Price']:>9.2f} "
                  f"{trade['PnL_Pct']:>+9.2f}% {trade['Reason']:<15}")
    
    # 绘制图表
    create_backtest_chart(data, equity_df, trades_df, symbol, initial_capital)
    
    return trades_df, equity_df, data


def create_backtest_chart(data, equity_df, trades_df, symbol, initial_capital):
    """创建回测图表"""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'{symbol} 布林带下轨策略回测结果', fontsize=14, fontweight='bold')
    
    bb_lower_col = 'BB_Lower_20'
    bb_upper_col = 'BB_Upper_20'
    bb_middle_col = 'BB_Middle_20'
    
    # 图1: 价格与布林带 + 交易点
    ax1 = axes[0]
    ax1.plot(data.index, data['Close'], label='收盘价', color='blue', linewidth=1)
    ax1.plot(data.index, data[bb_upper_col], label='上轨', color='gray', linestyle='--', alpha=0.7)
    ax1.plot(data.index, data[bb_middle_col], label='中轨', color='orange', linestyle='-', alpha=0.7)
    ax1.plot(data.index, data[bb_lower_col], label='下轨', color='gray', linestyle='--', alpha=0.7)
    ax1.fill_between(data.index, data[bb_lower_col], data[bb_upper_col], alpha=0.1, color='blue')
    
    # 标记买入卖出点
    if len(trades_df) > 0:
        for _, trade in trades_df.iterrows():
            ax1.scatter(trade['Entry_Date'], trade['Entry_Price'], 
                       color='green', s=100, marker='^', zorder=5)
            ax1.scatter(trade['Exit_Date'], trade['Exit_Price'], 
                       color='red', s=100, marker='v', zorder=5)
    
    ax1.scatter([], [], color='green', marker='^', s=100, label='买入')
    ax1.scatter([], [], color='red', marker='v', s=100, label='卖出')
    
    ax1.set_ylabel('价格 ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('价格走势与交易信号')
    
    # 图2: 权益曲线
    ax2 = axes[1]
    if len(equity_df) > 0:
        ax2.plot(equity_df['Date'], equity_df['Equity'], label='策略权益', color='blue', linewidth=1.5)
        ax2.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.7, label='初始资金')
        
        # 计算买入持有收益
        first_price = data.iloc[20]['Close']
        buy_hold = initial_capital * data['Close'] / first_price
        ax2.plot(data.index[20:], buy_hold.iloc[20:], label='买入持有', color='orange', alpha=0.7)
    
    ax2.set_ylabel('权益 ($)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('权益曲线对比')
    
    # 图3: 回撤
    ax3 = axes[2]
    if len(equity_df) > 0 and 'Drawdown' in equity_df.columns:
        ax3.fill_between(equity_df['Date'], 0, equity_df['Drawdown'], 
                        color='red', alpha=0.5, label='回撤')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax3.set_ylabel('回撤 (%)')
    ax3.set_xlabel('日期')
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('回撤曲线')
    
    # 格式化x轴
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    output_path = '/Users/windye/PycharmProjects/FUTU_auto/reports/bollinger_lower_backtest.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存至: {output_path}")
    
    plt.show()


if __name__ == '__main__':
    # 运行回测
    trades, equity, data = run_backtest(
        symbol='AAPL',
        initial_capital=100000,
        position_size=100000,  # 每次买入$100,000
        below_threshold=0.01,  # 低于下轨1%
        stop_loss=0.10,        # 止损10%
        take_profit=0.10       # 止盈10%
    )
