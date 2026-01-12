"""测试 pandas-ta 是否正常工作"""
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')

# 下载AAPL 15年数据
print('下载 AAPL 15年数据...')
data = yf.download('AAPL', start='2010-01-01', end='2025-01-07', progress=False, auto_adjust=False)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

print(f'数据行数: {len(data)}')

# 计算技术指标
print('\n计算技术指标...')
data.ta.rsi(length=14, append=True)
data.ta.stoch(append=True)
data.ta.macd(append=True)
data.ta.bbands(append=True)
data.ta.cci(append=True)
data.ta.willr(append=True)
data.ta.adx(append=True)
data.ta.mfi(append=True)
data.ta.roc(length=10, append=True)
data.ta.sma(length=20, append=True)
data.ta.sma(length=50, append=True)
data.ta.aroon(append=True)

new_cols = [c for c in data.columns if c not in ["Open","High","Low","Close","Adj Close","Volume"]]
print(f'新增指标列: {new_cols}')

# 找卖点（前后两周最高点）
print('\n找卖点（前后两周最高点）...')
high_prices = data['High'].values
dates = data.index.tolist()
sell_points = []
for i in range(10, len(data) - 10):
    current_high = high_prices[i]
    window_max = max(high_prices[i-10:i+11])
    if current_high >= window_max:
        sell_points.append(dates[i])

print(f'找到 {len(sell_points)} 个卖点')

# 分析因子
print('\n分析卖点因子出现频率...')
factor_counts = {}
total = len(sell_points)

# RSI > 70
rsi_count = sum(1 for d in sell_points if d in data.index and data.loc[d, 'RSI_14'] > 70)
factor_counts['RSI_14 > 70'] = rsi_count

# Stoch K > 80
stoch_k_col = [c for c in data.columns if 'STOCHk' in c]
if stoch_k_col:
    stoch_count = sum(1 for d in sell_points if d in data.index and data.loc[d, stoch_k_col[0]] > 80)
    factor_counts['Stoch_K > 80'] = stoch_count

# MACD > 0
macd_col = [c for c in data.columns if c.startswith('MACD_')]
if macd_col:
    macd_count = sum(1 for d in sell_points if d in data.index and data.loc[d, macd_col[0]] > 0)
    factor_counts['MACD > 0'] = macd_count

# BB位置 > 0.8
bbu = [c for c in data.columns if 'BBU' in c]
bbl = [c for c in data.columns if 'BBL' in c]
if bbu and bbl:
    bb_count = 0
    for d in sell_points:
        if d in data.index:
            bb_pos = (data.loc[d, 'Close'] - data.loc[d, bbl[0]]) / (data.loc[d, bbu[0]] - data.loc[d, bbl[0]])
            if bb_pos > 0.8:
                bb_count += 1
    factor_counts['BB位置 > 0.8'] = bb_count

# CCI > 100
cci_col = [c for c in data.columns if c.startswith('CCI')]
if cci_col:
    cci_count = sum(1 for d in sell_points if d in data.index and data.loc[d, cci_col[0]] > 100)
    factor_counts['CCI > 100'] = cci_count

# Williams %R > -20
willr_col = [c for c in data.columns if 'WILLR' in c]
if willr_col:
    willr_count = sum(1 for d in sell_points if d in data.index and data.loc[d, willr_col[0]] > -20)
    factor_counts['Williams%R > -20'] = willr_count

# 价格 > SMA50
if 'SMA_50' in data.columns:
    sma_count = sum(1 for d in sell_points if d in data.index and data.loc[d, 'Close'] > data.loc[d, 'SMA_50'])
    factor_counts['价格 > SMA_50'] = sma_count

# ADX > 25
adx_col = [c for c in data.columns if c.startswith('ADX_')]
if adx_col:
    adx_count = sum(1 for d in sell_points if d in data.index and data.loc[d, adx_col[0]] > 25)
    factor_counts['ADX > 25'] = adx_count

# Aroon Up > 80
aroonu_col = [c for c in data.columns if 'AROONU' in c]
if aroonu_col:
    aroon_count = sum(1 for d in sell_points if d in data.index and data.loc[d, aroonu_col[0]] > 80)
    factor_counts['Aroon_Up > 80'] = aroon_count

# MFI > 80
mfi_col = [c for c in data.columns if c.startswith('MFI')]
if mfi_col:
    mfi_count = sum(1 for d in sell_points if d in data.index and data.loc[d, mfi_col[0]] > 80)
    factor_counts['MFI > 80'] = mfi_count

# 输出结果
print('\n' + '='*50)
print('因子出现频率排名:')
print('='*50)
sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
for i, (name, count) in enumerate(sorted_factors, 1):
    print(f'{i:>2}. {name:<20} {count:>4}/{total} = {count/total*100:>5.1f}%')

print('\n✓ pandas-ta 工作正常!')
