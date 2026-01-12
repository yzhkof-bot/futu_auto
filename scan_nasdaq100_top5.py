"""
使用 Top 5 因子策略扫描纳斯达克100买点

Top 5 因子：
1. MACD > 0 (95.9%)
2. ADX > 25 (84.1%)
3. MACD > Signal (82.4%)
4. 10日涨幅 > 5% (66.3%)
5. MA5 > MA10 > MA20 > MA50 (60.3%)

买点定义：同时满足5个因子
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

from src.strategies.top5_factor_strategy import Top5FactorStrategy
from src.stock_pool import NASDAQ100, BLUECHIP_NON_TECH, ALL_STOCKS, get_pool_info


def scan_nasdaq100(min_factors: int = 5, days: int = 30):
    """
    扫描纳斯达克100最近N天的买点
    
    Args:
        min_factors: 最少满足因子数（默认5）
        days: 扫描最近N天（默认30天）
    """
    print("=" * 80)
    print("纳斯达克100 - Top 5 因子策略买点扫描")
    print("=" * 80)
    print("\nTop 5 因子:")
    print("  1. MACD > 0")
    print("  2. ADX > 25 (强趋势)")
    print("  3. MACD > Signal (MACD多头)")
    print("  4. 10日涨幅 > 5% (强动量)")
    print("  5. MA5 > MA10 > MA20 > MA50 (均线多头排列)")
    print(f"\n买点定义: 同时满足 >= {min_factors} 个因子")
    print("=" * 80)
    
    # 日期范围
    end_date = datetime.now().strftime('%Y-%m-%d')
    # 多获取一些数据用于计算指标
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    scan_start = datetime.now() - timedelta(days=days)
    
    print(f"\n扫描范围: 最近 {days} 天")
    print(f"扫描股票: {len(ALL_STOCKS)} 只 (纳斯达克100: {len(NASDAQ100)}, 蓝筹非科技: {len(BLUECHIP_NON_TECH)})")
    print("\n开始扫描...\n")
    
    strategy = Top5FactorStrategy(min_factors=min_factors)
    
    all_buy_points = []
    symbol_buy_counts = {}
    recent_signals = []  # 最近的买点信号
    
    for i, symbol in enumerate(ALL_STOCKS):
        print(f"[{i+1}/{len(ALL_STOCKS)}] {symbol}...", end=' ')
        
        try:
            # 获取数据
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if data.empty or len(data) < 60:
                print("数据不足")
                continue
            
            # 计算因子
            df = strategy.calculate_factors(data)
            if df.empty:
                print("计算失败")
                continue
            
            # 获取买点
            buy_signals = strategy.get_buy_signals(df)
            
            if len(buy_signals) > 0:
                # 筛选最近N天的买点
                recent_mask = buy_signals.index >= scan_start
                recent_buy = buy_signals[recent_mask]
                
                for date, row in recent_buy.iterrows():
                    bp = {
                        'symbol': symbol,
                        'date': date,
                        'price': row['Close'],
                        'factor_count': int(row['Factor_Count']),
                        'factors': strategy.get_satisfied_factors(row),
                        'RSI_14': row.get('RSI_14', None),
                        'ADX': row.get('ADX', None),
                        'MACD': row.get('MACD', None),
                        'Momentum_10': row.get('Momentum_10', None)
                    }
                    recent_signals.append(bp)
                
                # 统计总买点数
                symbol_buy_counts[symbol] = len(buy_signals)
                all_buy_points.extend([
                    {'symbol': symbol, 'date': d, 'price': r['Close']}
                    for d, r in buy_signals.iterrows()
                ])
                
                print(f"历史 {len(buy_signals)} 个, 近{days}天 {len(recent_buy)} 个")
            else:
                print("无买点")
                
        except Exception as e:
            print(f"错误: {e}")
    
    # 输出结果
    print("\n" + "=" * 80)
    print("扫描完成!")
    print("=" * 80)
    
    print(f"\n总计: {len(all_buy_points)} 个历史买点, 涉及 {len(symbol_buy_counts)} 只股票")
    
    # 按股票统计历史买点
    if symbol_buy_counts:
        print("\n" + "-" * 60)
        print("历史买点数量 Top 20:")
        print("-" * 60)
        sorted_counts = sorted(symbol_buy_counts.items(), key=lambda x: x[1], reverse=True)
        for symbol, count in sorted_counts[:20]:
            print(f"  {symbol}: {count} 个")
    
    # 最近买点详情
    if recent_signals:
        # 按日期排序
        recent_signals.sort(key=lambda x: x['date'], reverse=True)
        
        print("\n" + "=" * 80)
        print(f"最近 {days} 天买点信号 ({len(recent_signals)} 个):")
        print("=" * 80)
        print(f"\n{'日期':<12} {'股票':<6} {'价格':>10} {'因子':>4} {'RSI':>6} {'ADX':>6} {'满足因子'}")
        print("-" * 85)
        
        for bp in recent_signals:
            date_str = str(bp['date'])[:10]
            rsi = f"{bp['RSI_14']:.1f}" if bp['RSI_14'] else 'N/A'
            adx = f"{bp['ADX']:.1f}" if bp['ADX'] else 'N/A'
            factors_str = ', '.join(bp['factors'])
            print(f"{date_str:<12} {bp['symbol']:<6} ${bp['price']:>9.2f} {bp['factor_count']:>4} "
                  f"{rsi:>6} {adx:>6} {factors_str}")
        
        # 按股票分组统计最近买点
        recent_by_symbol = defaultdict(list)
        for bp in recent_signals:
            recent_by_symbol[bp['symbol']].append(bp)
        
        print("\n" + "-" * 60)
        print(f"最近 {days} 天有买点的股票:")
        print("-" * 60)
        for symbol, bps in sorted(recent_by_symbol.items(), key=lambda x: len(x[1]), reverse=True):
            latest = bps[0]
            print(f"  {symbol}: {len(bps)} 个买点, 最新 {str(latest['date'])[:10]} @ ${latest['price']:.2f}")
    else:
        print(f"\n最近 {days} 天无买点信号")
    
    # 保存结果
    if recent_signals:
        results_df = pd.DataFrame(recent_signals)
        results_df['date'] = pd.to_datetime(results_df['date']).dt.strftime('%Y-%m-%d')
        results_df['factors'] = results_df['factors'].apply(lambda x: ', '.join(x))
        results_df.to_csv('reports/nasdaq100_top5_signals.csv', index=False)
        print(f"\n结果已保存至 reports/nasdaq100_top5_signals.csv")
    
    return recent_signals, symbol_buy_counts


def scan_today():
    """扫描今天的买点"""
    print("=" * 70)
    print("全市场精选 - Top 5 因子策略 - 今日买点扫描")
    print("=" * 70)
    print("因子: MACD>0, ADX>25, MACD>Signal, 10日涨幅>5%, 均线多头排列")
    print(f"股票池: 纳斯达克100 ({len(NASDAQ100)}) + 蓝筹非科技 ({len(BLUECHIP_NON_TECH)}) = {len(ALL_STOCKS)} 只")
    print("=" * 70)
    
    strategy = Top5FactorStrategy(min_factors=5)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    today_signals = []
    
    print(f"\n扫描中...", end='')
    for i, symbol in enumerate(ALL_STOCKS):
        if i % 20 == 0:
            print(f" {i}", end='', flush=True)
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if data.empty or len(data) < 60:
                continue
            
            df = strategy.calculate_factors(data)
            if df.empty:
                continue
            
            # 检查最后一天是否有买点
            last_row = df.iloc[-1]
            if last_row.get('Buy_Signal', False):
                today_signals.append({
                    'symbol': symbol,
                    'date': df.index[-1],
                    'price': last_row['Close'],
                    'factor_count': int(last_row['Factor_Count']),
                    'factors': strategy.get_satisfied_factors(last_row),
                    'RSI_14': last_row.get('RSI_14', None),
                    'ADX': last_row.get('ADX', None),
                    'Momentum_10': last_row.get('Momentum_10', None)
                })
        except:
            continue
    
    print(" Done!\n")
    
    if today_signals:
        # 按动量排序
        today_signals.sort(key=lambda x: x.get('Momentum_10', 0) or 0, reverse=True)
        
        print(f"{'='*70}")
        print(f"今日买点信号: {len(today_signals)} 只")
        print(f"{'='*70}")
        print(f"\n{'股票':<6} {'价格':>10} {'RSI':>7} {'ADX':>7} {'10日涨幅':>10}")
        print("-" * 45)
        
        for bp in today_signals:
            rsi = f"{bp['RSI_14']:.1f}" if bp['RSI_14'] else 'N/A'
            adx = f"{bp['ADX']:.1f}" if bp['ADX'] else 'N/A'
            mom = f"{bp['Momentum_10']*100:.1f}%" if bp['Momentum_10'] else 'N/A'
            print(f"{bp['symbol']:<6} ${bp['price']:>9.2f} {rsi:>7} {adx:>7} {mom:>10}")
    else:
        print("今日无买点信号")
    
    return today_signals


if __name__ == '__main__':
    # 默认扫描今天
    scan_today()
