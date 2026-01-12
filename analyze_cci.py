"""
分析CCI指标的数值分布
"""

import sys
sys.path.append('src')
from strategies.ultra_elite_sell_strategy import UltraEliteSellStrategy
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

def analyze_cci_distribution(symbol='AAPL'):
    print(f'📊 分析{symbol}的CCI指标分布')
    print('=' * 40)

    time.sleep(3)

    # 获取数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)  # 5年数据

    data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), 
                      end=end_date.strftime('%Y-%m-%d'), progress=False, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # 计算CCI
    strategy = UltraEliteSellStrategy()
    df = strategy.calculate_indicators(data.copy())

    # 找到CCI列
    cci_col = [c for c in df.columns if c.startswith('CCI')][0]
    cci_values = df[cci_col].dropna()

    print(f'CCI统计信息:')
    print(f'  最大值: {cci_values.max():.2f}')
    print(f'  最小值: {cci_values.min():.2f}')
    print(f'  平均值: {cci_values.mean():.2f}')
    print(f'  中位数: {cci_values.median():.2f}')
    print(f'  75%分位: {cci_values.quantile(0.75):.2f}')
    print(f'  90%分位: {cci_values.quantile(0.90):.2f}')
    print(f'  95%分位: {cci_values.quantile(0.95):.2f}')
    print(f'  99%分位: {cci_values.quantile(0.99):.2f}')

    # 统计不同阈值的触发频率
    thresholds = [50, 60, 70, 80, 90, 100, 120, 150]
    print(f'\nCCI阈值触发频率:')
    for threshold in thresholds:
        count = (cci_values > threshold).sum()
        percentage = count / len(cci_values) * 100
        print(f'  CCI > {threshold}: {count} 次 ({percentage:.2f}%)')

if __name__ == '__main__':
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    analyze_cci_distribution(symbol)