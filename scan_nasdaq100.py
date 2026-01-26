#!/usr/bin/env python
"""
纳斯达克100买点扫描脚本
每天运行，扫描符合多信号策略的买点
"""

import warnings
warnings.filterwarnings('ignore')

from src.strategies import get_strategy
from datetime import datetime


def main():
    print(f"扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    strategy = get_strategy(min_signals=3)
    buy_signals = strategy.scan_nasdaq100()
    
    if buy_signals:
        print(f"\n{'='*50}")
        print(f"发现 {len(buy_signals)} 个抄底买点")
        print(f"{'='*50}")
        
        for ticker, signal in buy_signals:
            print(f"\n{ticker}:")
            print(f"  日期: {signal.date.strftime('%Y-%m-%d')}")
            print(f"  价格: ${signal.price:.2f}")
            print(f"  信号数: {signal.signal_count}")
            print(f"  信号: {', '.join(signal.signals)}")
            print(f"  RSI: {signal.rsi:.1f}, K: {signal.k:.1f}, D: {signal.d:.1f}")
    else:
        print("\n今日纳斯达克100无买点")


if __name__ == '__main__':
    main()
