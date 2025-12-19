"""
NVDA 5年回测脚本
统计趋势过滤策略的胜率、收益率等指标
"""
import backtrader as bt
import backtrader.analyzers as btanalyzers
from datetime import datetime, timedelta
from trend_filter_strategy import TrendFilterStrategy, download_data, create_cerebro


def run_backtest(symbol: str = 'NVDA', years: int = 5, initial_cash: float = 100000,
                 commission: float = 0.001, print_trades: bool = False):
    """
    运行回测并返回详细统计
    
    Args:
        symbol: 股票代码
        years: 回测年数
        initial_cash: 初始资金
        commission: 手续费率
        print_trades: 是否打印每笔交易
    """
    # 计算起始日期 (多加60天用于指标预热)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365 + 60)
    
    print(f"=" * 60)
    print(f"趋势过滤策略回测报告")
    print(f"=" * 60)
    print(f"标的: {symbol}")
    print(f"回测区间: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    print(f"初始资金: ${initial_cash:,.2f}")
    print(f"手续费率: {commission * 100:.2f}%")
    print(f"=" * 60)
    
    # 下载数据
    data_df = download_data(symbol, start_date.strftime('%Y-%m-%d'))
    
    if data_df.empty:
        print("错误: 无法获取数据")
        return None
    
    print(f"数据量: {len(data_df)} 个交易日")
    print(f"实际数据区间: {data_df.index[0].strftime('%Y-%m-%d')} ~ {data_df.index[-1].strftime('%Y-%m-%d')}")
    print("-" * 60)
    
    # 创建引擎
    cerebro = bt.Cerebro()
    
    # 添加策略
    cerebro.addstrategy(TrendFilterStrategy, print_log=False, print_trades=print_trades)
    
    # 添加数据
    data = bt.feeds.PandasData(dataname=data_df, plot=False)
    data._name = symbol
    cerebro.adddata(data)
    
    # 设置资金和手续费
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    
    # 添加分析器
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
    cerebro.addanalyzer(btanalyzers.SQN, _name='sqn')
    
    # 运行回测
    print("\n正在运行回测...")
    results = cerebro.run()
    strategy = results[0]
    
    # 获取分析结果
    trade_analysis = strategy.analyzers.trades.get_analysis()
    sharpe = strategy.analyzers.sharpe.get_analysis()
    drawdown = strategy.analyzers.drawdown.get_analysis()
    returns = strategy.analyzers.returns.get_analysis()
    sqn = strategy.analyzers.sqn.get_analysis()
    
    # 计算结果
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100
    
    print("\n" + "=" * 60)
    print("回测结果统计")
    print("=" * 60)
    
    # 资金统计
    print(f"\n【资金统计】")
    print(f"  初始资金: ${initial_cash:,.2f}")
    print(f"  最终资金: ${final_value:,.2f}")
    print(f"  总收益率: {total_return:+.2f}%")
    
    # 年化收益
    if 'rnorm100' in returns:
        print(f"  年化收益率: {returns['rnorm100']:.2f}%")
    
    # 交易统计
    print(f"\n【交易统计】")
    total_trades = trade_analysis.get('total', {}).get('total', 0)
    print(f"  总交易次数: {total_trades}")
    
    if total_trades > 0:
        # 盈利交易
        won = trade_analysis.get('won', {})
        won_total = won.get('total', 0)
        won_pnl_total = won.get('pnl', {}).get('total', 0)
        won_pnl_avg = won.get('pnl', {}).get('average', 0)
        won_pnl_max = won.get('pnl', {}).get('max', 0)
        
        # 亏损交易
        lost = trade_analysis.get('lost', {})
        lost_total = lost.get('total', 0)
        lost_pnl_total = lost.get('pnl', {}).get('total', 0)
        lost_pnl_avg = lost.get('pnl', {}).get('average', 0)
        lost_pnl_max = lost.get('pnl', {}).get('max', 0)
        
        # 胜率
        win_rate = (won_total / total_trades * 100) if total_trades > 0 else 0
        
        print(f"  盈利交易: {won_total} 笔")
        print(f"  亏损交易: {lost_total} 笔")
        print(f"  胜率: {win_rate:.2f}%")
        
        print(f"\n【盈亏分析】")
        print(f"  盈利总额: ${won_pnl_total:,.2f}")
        print(f"  亏损总额: ${lost_pnl_total:,.2f}")
        print(f"  净利润: ${won_pnl_total + lost_pnl_total:,.2f}")
        
        if won_total > 0:
            print(f"  平均盈利: ${won_pnl_avg:,.2f}")
            print(f"  最大单笔盈利: ${won_pnl_max:,.2f}")
        if lost_total > 0:
            print(f"  平均亏损: ${lost_pnl_avg:,.2f}")
            print(f"  最大单笔亏损: ${lost_pnl_max:,.2f}")
        
        # 盈亏比
        if lost_pnl_avg != 0 and won_pnl_avg != 0:
            profit_factor = abs(won_pnl_avg / lost_pnl_avg)
            print(f"  盈亏比: {profit_factor:.2f}")
    
    # 风险指标
    print(f"\n【风险指标】")
    if sharpe.get('sharperatio'):
        print(f"  夏普比率: {sharpe['sharperatio']:.3f}")
    else:
        print(f"  夏普比率: N/A")
    
    max_dd = drawdown.get('max', {}).get('drawdown', 0)
    max_dd_len = drawdown.get('max', {}).get('len', 0)
    print(f"  最大回撤: {max_dd:.2f}%")
    print(f"  最大回撤持续: {max_dd_len} 天")
    
    if sqn.get('sqn'):
        print(f"  SQN (系统质量指数): {sqn['sqn']:.2f}")
    
    # 策略评价
    print(f"\n【策略评价】")
    if total_trades > 0:
        if win_rate >= 50 and total_return > 0:
            print("  ✓ 策略整体表现良好")
        elif win_rate >= 40 and total_return > 0:
            print("  ○ 策略表现尚可，胜率偏低但整体盈利")
        elif total_return > 0:
            print("  △ 策略盈利但胜率较低，依赖大盈利覆盖小亏损")
        else:
            print("  ✗ 策略表现不佳，需要优化")
    else:
        print("  无交易记录，无法评价")
    
    print("\n" + "=" * 60)
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'total_trades': total_trades,
        'win_rate': win_rate if total_trades > 0 else 0,
        'max_drawdown': max_dd,
        'sharpe_ratio': sharpe.get('sharperatio'),
        'cerebro': cerebro
    }


if __name__ == '__main__':
    # 运行5年回测
    result = run_backtest(
        symbol='NVDA',
        years=5,
        initial_cash=100000,
        commission=0.001,
        print_trades=True  # 打印每笔交易详情
    )
    
    # 可选：显示图表
    if result and result.get('cerebro'):
        try:
            print("\n正在生成图表...")
            result['cerebro'].plot(style='candlestick', volume=False)
        except Exception as e:
            print(f"图表生成失败: {e}")
