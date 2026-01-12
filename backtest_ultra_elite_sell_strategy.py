"""
超强精英卖点策略回测

基于15个因子中满足13个以上的超强精准卖点策略
"""

import sys
sys.path.append('src')

from strategies.ultra_elite_sell_strategy import find_ultra_elite_sell_points, UltraEliteSellStrategy
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_ultra_elite_sell_points(symbol: str, 
                                   min_factors: int = 13,
                                   start_date: str = None, end_date: str = None,
                                   years: int = 15, cooldown_days: int = 7):
    """
    分析超强精英卖点
    
    Args:
        symbol: 股票代码
        min_factors: 最少满足因子数（默认13个，范围10-15）
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
    
    print(f"\n⚡ 超强精英卖点策略分析 - {symbol}")
    print(f"📅 时间范围: {start_date} ~ {end_date}")
    print(f"🔥 策略要求: 15个因子中满足{min_factors}个以上 ({min_factors/15*100:.1f}%)")
    print(f"⏰ 冷却期: {cooldown_days}天")
    print("=" * 70)
    
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
        strategy = UltraEliteSellStrategy(min_factors=min_factors)
        
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
            print(f"🎯 未找到满足{min_factors}个因子条件的超强精英卖点")
            print(f"💡 原始卖点数: {len(raw_sell_points)}")
            return None
        
        # 统计分析
        print(f"📊 原始卖点数: {len(raw_sell_points)}")
        print(f"📊 冷却后卖点数: {len(sell_points)}")
        if len(raw_sell_points) > 0:
            print(f"📊 过滤率: {(1-len(sell_points)/len(raw_sell_points))*100:.1f}%")
        
        # 按年份统计
        yearly_stats = {}
        for point in sell_points:
            year = point['date'].year
            yearly_stats[year] = yearly_stats.get(year, 0) + 1
        
        print(f"\n📈 按年份分布:")
        for year in sorted(yearly_stats.keys()):
            print(f"  {year}: {yearly_stats[year]} 个")
        
        # 因子满足度统计
        factor_counts = [point['factor_count'] for point in sell_points]
        print(f"\n📊 因子满足度分布:")
        for count in sorted(set(factor_counts), reverse=True):
            num = factor_counts.count(count)
            print(f"  {count}/15因子: {num}个卖点 ({num/len(sell_points)*100:.1f}%)")
        
        # 最近卖点
        print(f"\n🔴 最近超强精英卖点:")
        recent_points = sell_points[-5:] if len(sell_points) >= 5 else sell_points
        for i, point in enumerate(recent_points, 1):
            print(f"  {i:>2}. {point['date'].strftime('%Y-%m-%d')}: ${point['price']:.2f} "
                  f"({point['factor_count']}/15因子)")
        
        return sell_points, df
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return None


def backtest_ultra_elite_strategy(symbol: str, min_factors: int = 13, 
                                 years: int = 15, cooldown_days: int = 7):
    """
    回测超强精英策略效果
    """
    print(f"\n🔄 超强精英策略回测 - {symbol}")
    
    result = analyze_ultra_elite_sell_points(symbol, min_factors=min_factors, 
                                           years=years, cooldown_days=cooldown_days)
    if not result:
        return
    
    sell_points, df = result
    
    if not sell_points:
        print("❌ 无卖点数据，无法进行回测")
        return
    
    # 计算卖点后的收益
    returns_data = {
        '1天后': [],
        '3天后': [],
        '5天后': [],
        '10天后': [],
        '20天后': []
    }
    
    for point in sell_points:
        sell_date = point['date']
        sell_price = point['price']
        
        # 找到卖点后的价格
        future_data = df[df.index > sell_date]
        
        for days, key in [(1, '1天后'), (3, '3天后'), (5, '5天后'), 
                         (10, '10天后'), (20, '20天后')]:
            if len(future_data) >= days:
                future_price = future_data.iloc[days-1]['Close']
                return_pct = (future_price - sell_price) / sell_price * 100
                returns_data[key].append(return_pct)
    
    # 统计结果
    print(f"\n📊 超强精英卖点后收益统计:")
    print(f"{'期间':<8} {'平均收益':<10} {'胜率':<8} {'最大收益':<10} {'最大亏损':<10} {'样本数':<8}")
    print("-" * 70)
    
    for period, returns_list in returns_data.items():
        if returns_list:
            avg_return = np.mean(returns_list)
            win_rate = sum(1 for r in returns_list if r < 0) / len(returns_list) * 100  # 卖点后下跌为胜
            max_return = max(returns_list)
            min_return = min(returns_list)
            sample_size = len(returns_list)
            
            print(f"{period:<8} {avg_return:>9.2f}% {win_rate:>7.1f}% {max_return:>9.2f}% {min_return:>9.2f}% {sample_size:>7}")
    
    # 计算其他指标
    if returns_data['5天后']:
        returns_5d = returns_data['5天后']
        negative_returns = [-r for r in returns_5d]  # 转换为负收益（卖点策略）
        
        sharpe_5d = np.mean(negative_returns) / np.std(negative_returns) if np.std(negative_returns) > 0 else 0
        print(f"\n📈 5天收益夏普比率: {sharpe_5d:.3f}")
        
        # 胜率分析
        win_count = sum(1 for r in returns_5d if r < 0)
        total_count = len(returns_5d)
        win_rate = win_count / total_count * 100
        
        print(f"🎯 策略表现:")
        print(f"  总信号数: {total_count}")
        print(f"  成功信号: {win_count} ({win_rate:.1f}%)")
        print(f"  失败信号: {total_count - win_count} ({100-win_rate:.1f}%)")
    
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
        print(f"  中位数间隔: {np.median(intervals):.1f} 天")


def test_different_thresholds(symbol: str, years: int = 5):
    """
    测试不同因子阈值的效果
    """
    print(f"\n🧪 因子阈值测试 - {symbol} (近{years}年)")
    print("=" * 60)
    
    thresholds = [10, 11, 12, 13, 14, 15]
    results = []
    
    for threshold in thresholds:
        try:
            points = find_ultra_elite_sell_points(
                symbol, 
                min_factors=threshold,
                start_date=(datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d'),
                cooldown_days=7, 
                print_result=False
            )
            results.append((threshold, len(points)))
        except:
            results.append((threshold, 0))
    
    print(f"{'阈值':<8} {'卖点数':<8} {'年均卖点':<10} {'精准度':<10}")
    print("-" * 40)
    for threshold, count in results:
        annual_avg = count / years
        if annual_avg == 0:
            precision = "无信号"
        elif annual_avg < 3:
            precision = "极高"
        elif annual_avg < 10:
            precision = "高"
        elif annual_avg < 30:
            precision = "中等"
        else:
            precision = "低"
        
        print(f"{threshold}/15{'':<3} {count:<8} {annual_avg:<9.1f} {precision:<10}")


if __name__ == '__main__':
    # 获取股票代码
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    min_factors = int(sys.argv[2]) if len(sys.argv) > 2 else 13
    cooldown_days = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    
    # 分析超强精英卖点
    result = analyze_ultra_elite_sell_points(
        symbol=symbol,
        min_factors=min_factors,
        years=15,
        cooldown_days=cooldown_days
    )
    
    if result:
        sell_points, df = result
        
        # 绘制图表
        from strategies.ultra_elite_sell_strategy import plot_ultra_elite_signals
        plot_ultra_elite_signals(symbol, df, sell_points, min_factors,
                                save_path=f'reports/{symbol.lower()}_ultra_elite_sell_points.png')
        
        # 回测分析
        backtest_ultra_elite_strategy(symbol, min_factors=min_factors, 
                                    years=15, cooldown_days=cooldown_days)
    
    # 阈值测试
    test_different_thresholds(symbol, years=5)