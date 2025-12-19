#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""yfinance 获取历史K线示例"""

import yfinance as yf
import pandas as pd

print('=' * 60)
print('yfinance 获取 NVDA 历史K线示例')
print('=' * 60)

# 获取NVDA历史数据
data = yf.download('NVDA', start='2024-01-01', end='2024-12-31', progress=False)

# 处理MultiIndex列名
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

print(f'\n数据范围: {data.index[0].strftime("%Y-%m-%d")} 至 {data.index[-1].strftime("%Y-%m-%d")}')
print(f'总共 {len(data)} 个交易日')

print('\n【数据列】')
print(data.columns.tolist())

print('\n【前5条数据】')
print(data.head())

print('\n【后5条数据】')
print(data.tail())

print('\n' + '=' * 60)
print('基本统计')
print('=' * 60)
print(f'最高价: ${data["High"].max():.2f}')
print(f'最低价: ${data["Low"].min():.2f}')
print(f'平均收盘价: ${data["Close"].mean():.2f}')
print(f'总成交量: {data["Volume"].sum():,.0f}')

# 涨跌统计
data['Change'] = data['Close'].pct_change()
up_days = (data['Change'] > 0).sum()
down_days = (data['Change'] < 0).sum()
print(f'\n上涨天数: {up_days}')
print(f'下跌天数: {down_days}')
print(f'年度涨幅: {((data["Close"].iloc[-1] / data["Close"].iloc[0]) - 1) * 100:.2f}%')
