#!/usr/bin/env python
"""
统一扫描脚本 - 合并抄底和追涨扫描
只下载一次数据，同时运行两种策略
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from src.strategies import get_strategy
from src.strategies.top5_factor_strategy import Top5FactorStrategy
from src.stock_pool import ALL_STOCKS, NASDAQ100, BLUECHIP_NON_TECH, get_pool_info


def download_all_data(days: int = 365) -> Dict[str, pd.DataFrame]:
    """
    批量下载所有股票数据
    
    Args:
        days: 下载天数
    
    Returns:
        {symbol: DataFrame} 字典
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    pool_info = get_pool_info()
    print(f"下载数据中...")
    print(f"  股票池: 纳斯达克100 ({pool_info['nasdaq100_count']}) + 蓝筹非科技 ({pool_info['bluechip_count']}) = {pool_info['total_count']} 只")
    print(f"  日期范围: {start_date} ~ {end_date}")
    
    all_data = {}
    failed = []
    
    for i, symbol in enumerate(ALL_STOCKS):
        if (i + 1) % 20 == 0:
            print(f"  已下载 {i + 1}/{len(ALL_STOCKS)}...")
        
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if not data.empty and len(data) >= 60:
                all_data[symbol] = data
            else:
                failed.append(symbol)
        except Exception as e:
            failed.append(symbol)
    
    print(f"  下载完成: 成功 {len(all_data)} 只, 失败 {len(failed)} 只")
    if failed:
        print(f"  失败列表: {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")
    
    return all_data


def scan_bottom_fishing(all_data: Dict[str, pd.DataFrame], min_signals: int = 3) -> List[Tuple[str, dict]]:
    """
    抄底信号扫描 (多信号策略)
    
    Args:
        all_data: 股票数据字典
        min_signals: 最少信号数
    
    Returns:
        买点列表 [(symbol, signal_info), ...]
    """
    strategy = get_strategy(min_signals=min_signals)
    buy_signals = []
    
    for symbol, data in all_data.items():
        try:
            is_buy, signal = strategy.is_buy_signal_today(data)
            if is_buy and signal:
                buy_signals.append((symbol, {
                    'date': signal.date,
                    'price': signal.price,
                    'signal_count': signal.signal_count,
                    'signals': signal.signals,
                    'rsi': signal.rsi,
                    'k': signal.k,
                    'd': signal.d
                }))
        except:
            pass
    
    return buy_signals


def scan_momentum(all_data: Dict[str, pd.DataFrame], min_factors: int = 5) -> List[dict]:
    """
    追涨信号扫描 (Top5因子策略)
    
    Args:
        all_data: 股票数据字典
        min_factors: 最少因子数
    
    Returns:
        买点列表 [signal_info, ...]
    """
    strategy = Top5FactorStrategy(min_factors=min_factors)
    today_signals = []
    
    for symbol, data in all_data.items():
        try:
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
            pass
    
    # 按动量排序
    today_signals.sort(key=lambda x: x.get('Momentum_10', 0) or 0, reverse=True)
    return today_signals


def main():
    print(f"扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 1. 统一下载数据
    all_data = download_all_data(days=365)
    
    if not all_data:
        print("无法获取数据，退出")
        return
    
    # 2. 抄底信号扫描
    print("\n" + "=" * 70)
    print("========== 抄底信号扫描 (≥3个信号) ==========")
    print("=" * 70)
    
    bottom_signals = scan_bottom_fishing(all_data, min_signals=3)
    
    if bottom_signals:
        print(f"\n发现 {len(bottom_signals)} 个抄底买点:")
        print("-" * 60)
        for symbol, signal in bottom_signals:
            print(f"\n{symbol}:")
            print(f"  日期: {signal['date'].strftime('%Y-%m-%d')}")
            print(f"  价格: ${signal['price']:.2f}")
            print(f"  信号数: {signal['signal_count']}")
            print(f"  信号: {', '.join(signal['signals'])}")
            print(f"  RSI: {signal['rsi']:.1f}, K: {signal['k']:.1f}, D: {signal['d']:.1f}")
    else:
        print("\n今日无抄底买点")
    
    # 3. 追涨信号扫描
    print("\n" + "=" * 70)
    print("========== 追涨信号扫描 (Top5因子, 满足≥5个) ==========")
    print("=" * 70)
    print("因子: MACD>0, ADX>25, MACD>Signal, 10日涨幅>5%, 均线多头排列")
    
    momentum_signals = scan_momentum(all_data, min_factors=5)
    
    if momentum_signals:
        print(f"\n发现 {len(momentum_signals)} 个追涨买点:")
        print("-" * 60)
        print(f"{'股票':<6} {'价格':>10} {'RSI':>7} {'ADX':>7} {'10日涨幅':>10}")
        print("-" * 45)
        
        for bp in momentum_signals:
            rsi = f"{bp['RSI_14']:.1f}" if bp['RSI_14'] else 'N/A'
            adx = f"{bp['ADX']:.1f}" if bp['ADX'] else 'N/A'
            mom = f"{bp['Momentum_10']*100:.1f}%" if bp['Momentum_10'] else 'N/A'
            print(f"{bp['symbol']:<6} ${bp['price']:>9.2f} {rsi:>7} {adx:>7} {mom:>10}")
    else:
        print("\n今日无追涨买点")
    
    # 4. 汇总
    print("\n" + "=" * 70)
    print("扫描完成汇总")
    print("=" * 70)
    print(f"  抄底买点: {len(bottom_signals)} 只")
    print(f"  追涨买点: {len(momentum_signals)} 只")
    
    # 找出同时出现的股票
    bottom_symbols = {s[0] for s in bottom_signals}
    momentum_symbols = {s['symbol'] for s in momentum_signals}
    both = bottom_symbols & momentum_symbols
    if both:
        print(f"  同时出现: {', '.join(sorted(both))}")


if __name__ == '__main__':
    main()
