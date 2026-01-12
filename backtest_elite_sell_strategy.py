"""
精英卖点策略回测

基于5个不同类型高频因子的精准卖点策略
"""

import sys
sys.path.append('src')

from strategies.elite_sell_strategy import find_elite_sell_points, EliteSellStrategy
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_elite_sell_points(symbol: str, 
                             start_date: str = None, end_date: str = None,
                             years: int = 15, cooldown_days: int = 5):
    """
    分析精英卖点
    
    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        years: 分析年数（当start_date为None时使用）
        cooldown_days: 冷却期天数
    """
    
    # 设置日期范围
    if start_date is None:
        end_date = datetime.now() if end_date is None else datetime.strptime(end_date, '%Y-%m-%d')
        start_date = (end_date - timedelta(days=years * 365)).strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
    
    print(f"\n🎯 精英卖点策略分析 - {symbol}")
    print(f"📅 时间范围: {start_date} ~ {end_date}")
    print(f"🔥 策略要求: 5个不同类型因子全部满足")
    print(f"⏰ 冷却期: {cooldown_days}天")
    print("=" * 60)
    
    try:
        # 下载数据
        data = yf.download(symbol, start=start_date, end=end_date, 
                          progress=False, auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if data.empty or len(data) < 100:
            print(f"❌ 数据不足，跳过分析")
            return None
        
        # 初始化策略
        strategy = EliteSellStrategy()
        
        # 计算指标和因子
        df = strategy.calculate_indicators(data.copy())
        df = strategy.calculate_factors(df)
        
        # 识别原始卖点（不应用冷却期）
        raw_sell_points = strategy.identify_sell_points(df)
        
        # 应用冷却期
        if cooldown_days > 0 and len(raw_sell_points) > 1:
            filtered_points = [raw_sell_points[0]]
            for point in raw_sell_points[1:]:
                last_date = filtered_points[-1]['date']
                if (point['date'] - last_date).days >= cooldown_days:
                    filtered_points.append(point)
            sell_points = filtered_points
        else:
            sell_points = raw_sell_points
        
        if not sell_points:
            print(f"❌ 未找到满足条件的精英卖点")
            return None
        
        # 统计分析
        print(f"📊 原始卖点数: {len(raw_sell_points)}")
        print(f"📊 冷却后卖点数: {len(sell_points)}")
        print(f"📊 过滤率: {(1-len(sell_points)/len(raw_sell_points))*100:.1f}%")
        
        # 按年份统计
        yearly_stats = {}
        for point in sell_points:
            year = point['date'].year
            yearly_stats[year] = yearly_stats.get(year, 0) + 1
        
        print(f"\n📈 按年份分布:")
        for year in sorted(yearly_stats.keys()):
            print(f"  {year}: {yearly_stats[year]} 个")
        
        # 因子验证（应该都是100%）
        factor_names = ['Trend_Strong', 'Momentum_Positive', 'Overbought_Williams', 
                       'Position_Above_MA50', 'Oscillator_Overbought']
        
        print(f"\n🎯 因子验证 (应该都是100%):")
        total_points = len(sell_points)
        for name in factor_names:
            count = sum(1 for point in sell_points if point['factors'][name])
            print(f"  {name}: {count}/{total_points} = {count/total_points*100:.1f}%")
        
        # 最近卖点
        print(f"\n🔴 最近精英卖点:")
        recent_points = sell_points[-5:] if len(sell_points) >= 5 else sell_points
        for point in recent_points:
            print(f"  {point['date'].strftime('%Y-%m-%d')}: ${point['price']:.2f}")
        
        return sell_points, df
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return None


def backtest_elite_strategy(symbol: str, years: int = 15, cooldown_days: int = 5):
    """
    回测精英策略效果
    """
    print(f"\n🔄 精英策略回测 - {symbol}")
    
    result = analyze_elite_sell_points(symbol, years=years, cooldown_days=cooldown_days)
    if not result:
        return
    
    sell_points, df = result
    
    # 计算卖点后的收益
    returns_1d = []
    returns_3d = []
    returns_5d = []
    returns_10d = []
    returns_20d = []
    
    for point in sell_points:
        sell_date = point['date']
        sell_price = point['price']
        
        # 找到卖点后的价格
        future_data = df[df.index > sell_date]
        
        for days, returns_list in [(1, returns_1d), (3, returns_3d), (5, returns_5d), 
                                  (10, returns_10d), (20, returns_20d)]:
            if len(future_data) >= days:
                future_price = future_data.iloc[days-1]['Close']
                return_pct = (future_price - sell_price) / sell_price * 100
                returns_list.append(return_pct)
    
    # 统计结果
    print(f"\n📊 精英卖点后收益统计:")
    print(f"{'期间':<8} {'平均收益':<10} {'胜率':<8} {'最大收益':<10} {'最大亏损':<10} {'样本数':<8}")
    print("-" * 60)
    
    for days, returns_list, name in [(1, returns_1d, '1天后'), (3, returns_3d, '3天后'), 
                                    (5, returns_5d, '5天后'), (10, returns_10d, '10天后'), 
                                    (20, returns_20d, '20天后')]:
        if returns_list:
            avg_return = np.mean(returns_list)
            win_rate = sum(1 for r in returns_list if r < 0) / len(returns_list) * 100  # 卖点后下跌为胜
            max_return = max(returns_list)
            min_return = min(returns_list)
            sample_size = len(returns_list)
            
            print(f"{name:<8} {avg_return:>9.2f}% {win_rate:>7.1f}% {max_return:>9.2f}% {min_return:>9.2f}% {sample_size:>7}")
    
    # 计算夏普比率（假设无风险利率为0）
    if returns_5d:
        sharpe_5d = np.mean([-r for r in returns_5d]) / np.std(returns_5d) if np.std(returns_5d) > 0 else 0
        print(f"\n📈 5天收益夏普比率: {sharpe_5d:.3f}")
    
    # 连续性分析
    if len(sell_points) > 1:
        intervals = []
        for i in range(1, len(sell_points)):
            interval = (sell_points[i]['date'] - sell_points[i-1]['date']).days
            intervals.append(interval)
        
        print(f"\n⏰ 卖点间隔统计:")
        print(f"  平均间隔: {np.mean(intervals):.1f} 天")
        print(f"  最短间隔: {min(intervals)} 天")
        print(f"  最长间隔: {max(intervals)} 天")


def compare_strategies(symbol: str, years: int = 5):
    """
    比较不同策略效果
    """
    print(f"\n🔍 策略对比分析 - {symbol} (近{years}年)")
    print("=" * 70)
    
    # 精英策略 (5因子全满足)
    elite_points = find_elite_sell_points(symbol, 
                                         start_date=(datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d'),
                                         cooldown_days=5, print_result=False)
    
    # Top10策略 (7因子满足)
    from strategies.top10_sell_strategy import find_sell_points
    top10_points = find_sell_points(symbol, min_factors=7, 
                                   start_date=(datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d'),
                                   print_result=False)
    
    # Top5策略 (5因子全满足)
    from strategies.top5_sell_strategy import find_sell_points as find_top5_points
    top5_points = find_top5_points(symbol, min_factors=5, 
                                  start_date=(datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d'),
                                  print_result=False)
    
    print(f"{'策略':<15} {'卖点数':<8} {'年均卖点':<10}")
    print("-" * 35)
    print(f"{'精英策略':<15} {len(elite_points):<8} {len(elite_points)/years:<9.1f}")
    print(f"{'Top10策略':<15} {len(top10_points):<8} {len(top10_points)/years:<9.1f}")
    print(f"{'Top5策略':<15} {len(top5_points):<8} {len(top5_points)/years:<9.1f}")


if __name__ == '__main__':
    # 获取股票代码
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    cooldown_days = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # 分析精英卖点
    result = analyze_elite_sell_points(
        symbol=symbol,
        years=15,
        cooldown_days=cooldown_days
    )
    
    if result:
        sell_points, df = result
        
        # 绘制图表
        from strategies.elite_sell_strategy import plot_elite_signals
        plot_elite_signals(symbol, df, sell_points, 
                          save_path=f'reports/{symbol.lower()}_elite_sell_points.png')
        
        # 回测分析
        backtest_elite_strategy(symbol, years=15, cooldown_days=cooldown_days)
        
        # 策略对比
        compare_strategies(symbol, years=5)