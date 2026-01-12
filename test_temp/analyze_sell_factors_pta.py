"""
卖点因子分析工具 - 使用 pandas-ta 库

卖点定义：前后两周（10个交易日）最高价的那天
分析这些卖点出现最多的技术因子
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
from datetime import datetime, timedelta
from collections import Counter
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.stock_pool import get_stock_pool


def find_local_highs(df: pd.DataFrame, lookback_days: int = 10, lookforward_days: int = 10) -> list:
    """
    找出前后N个交易日的最高点（卖点）
    """
    sell_points = []
    high_prices = df['High'].values
    dates = df.index.tolist()
    
    for i in range(lookback_days, len(df) - lookforward_days):
        current_high = high_prices[i]
        window_start = i - lookback_days
        window_end = i + lookforward_days + 1
        window_max = max(high_prices[window_start:window_end])
        
        if current_high >= window_max:
            sell_points.append(dates[i])
    
    return sell_points


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """使用 pandas-ta 计算所有技术指标"""
    
    # 计算常用技术指标
    # RSI
    df.ta.rsi(length=14, append=True)
    df.ta.rsi(length=7, append=True)
    
    # Stochastic (类似KDJ)
    df.ta.stoch(append=True)
    df.ta.stochrsi(append=True)
    
    # MACD
    df.ta.macd(append=True)
    
    # 布林带
    df.ta.bbands(append=True)
    
    # CCI
    df.ta.cci(append=True)
    
    # Williams %R
    df.ta.willr(append=True)
    
    # ADX
    df.ta.adx(append=True)
    
    # MFI
    df.ta.mfi(append=True)
    
    # ROC
    df.ta.roc(length=10, append=True)
    df.ta.roc(length=20, append=True)
    
    # 均线
    df.ta.sma(length=5, append=True)
    df.ta.sma(length=10, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.ema(length=12, append=True)
    df.ta.ema(length=26, append=True)
    
    # Aroon
    df.ta.aroon(append=True)
    
    # Ultimate Oscillator
    df.ta.uo(append=True)
    
    # TSI
    df.ta.tsi(append=True)
    
    # ATR
    df.ta.atr(append=True)
    
    # 补充一些自定义指标
    # 价格位置
    df['High_20'] = df['High'].rolling(window=20).max()
    df['Low_20'] = df['Low'].rolling(window=20).min()
    df['Price_Position_20'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'])
    
    df['High_50'] = df['High'].rolling(window=50).max()
    df['Low_50'] = df['Low'].rolling(window=50).min()
    df['Price_Position_50'] = (df['Close'] - df['Low_50']) / (df['High_50'] - df['Low_50'])
    
    # 涨跌幅
    df['Daily_Return'] = df['Close'].pct_change()
    df['Weekly_Return'] = df['Close'].pct_change(5)
    
    # 成交量均线
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
    
    # 均线偏离度
    if 'SMA_20' in df.columns:
        df['MA20_Deviation'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
    if 'SMA_50' in df.columns:
        df['MA50_Deviation'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    
    return df


# 定义因子条件
def get_factor_definitions(df: pd.DataFrame) -> dict:
    """根据可用列动态生成因子定义"""
    factors = {}
    
    # RSI相关
    rsi_cols = [c for c in df.columns if c.startswith('RSI')]
    for col in rsi_cols:
        factors[f'{col}_above_70'] = (f'{col} > 70', df[col] > 70)
        factors[f'{col}_above_80'] = (f'{col} > 80', df[col] > 80)
    
    # Stochastic (类似KDJ)
    stoch_k_cols = [c for c in df.columns if 'STOCH' in c and '_k' in c.lower()]
    for col in stoch_k_cols:
        factors[f'{col}_above_80'] = (f'{col} > 80', df[col] > 80)
        factors[f'{col}_above_90'] = (f'{col} > 90', df[col] > 90)
    
    # MACD相关
    macd_cols = [c for c in df.columns if c.startswith('MACD_')]
    for col in macd_cols:
        if 'h' not in col.lower():  # MACD线，非柱状图
            factors[f'{col}_positive'] = (f'{col} > 0', df[col] > 0)
    
    macdh_cols = [c for c in df.columns if 'MACDh' in c or 'MACD_' in c and 'h' in c.lower()]
    for col in macdh_cols:
        factors[f'{col}_positive'] = (f'{col} > 0 (柱状图)', df[col] > 0)
    
    # 布林带相关
    bbu_cols = [c for c in df.columns if 'BBU' in c]
    bbl_cols = [c for c in df.columns if 'BBL' in c]
    bbm_cols = [c for c in df.columns if 'BBM' in c]
    
    for bbu in bbu_cols:
        factors[f'price_above_{bbu}'] = (f'价格 > {bbu}', df['Close'] > df[bbu])
    
    for bbu, bbl in zip(bbu_cols, bbl_cols):
        bb_pos = (df['Close'] - df[bbl]) / (df[bbu] - df[bbl])
        factors[f'BB_position_above_80'] = ('BB位置 > 0.8', bb_pos > 0.8)
        factors[f'BB_position_above_90'] = ('BB位置 > 0.9', bb_pos > 0.9)
        break  # 只取第一个
    
    # CCI相关
    cci_cols = [c for c in df.columns if c.startswith('CCI')]
    for col in cci_cols:
        factors[f'{col}_above_100'] = (f'{col} > 100', df[col] > 100)
        factors[f'{col}_above_200'] = (f'{col} > 200', df[col] > 200)
    
    # Williams %R
    willr_cols = [c for c in df.columns if 'WILLR' in c]
    for col in willr_cols:
        factors[f'{col}_above_minus20'] = (f'{col} > -20 (超买)', df[col] > -20)
    
    # ADX (趋势强度)
    adx_cols = [c for c in df.columns if c.startswith('ADX_')]
    for col in adx_cols:
        factors[f'{col}_above_25'] = (f'{col} > 25 (强趋势)', df[col] > 25)
        factors[f'{col}_above_40'] = (f'{col} > 40 (极强趋势)', df[col] > 40)
    
    # MFI (资金流量指标)
    mfi_cols = [c for c in df.columns if c.startswith('MFI')]
    for col in mfi_cols:
        factors[f'{col}_above_80'] = (f'{col} > 80 (超买)', df[col] > 80)
    
    # ROC (变动率)
    roc_cols = [c for c in df.columns if c.startswith('ROC_')]
    for col in roc_cols:
        factors[f'{col}_above_10'] = (f'{col} > 10%', df[col] > 10)
        factors[f'{col}_above_20'] = (f'{col} > 20%', df[col] > 20)
    
    # 价格位置
    if 'Price_Position_20' in df.columns:
        factors['price_pos_20_above_90'] = ('20日价格位置 > 90%', df['Price_Position_20'] > 0.9)
        factors['price_pos_20_above_95'] = ('20日价格位置 > 95%', df['Price_Position_20'] > 0.95)
    
    if 'Price_Position_50' in df.columns:
        factors['price_pos_50_above_90'] = ('50日价格位置 > 90%', df['Price_Position_50'] > 0.9)
        factors['price_pos_50_above_95'] = ('50日价格位置 > 95%', df['Price_Position_50'] > 0.95)
    
    # 均线相关
    sma_cols = [c for c in df.columns if c.startswith('SMA_')]
    for col in sma_cols:
        factors[f'price_above_{col}'] = (f'价格 > {col}', df['Close'] > df[col])
    
    ema_cols = [c for c in df.columns if c.startswith('EMA_')]
    for col in ema_cols[:3]:  # 只取前3个
        factors[f'price_above_{col}'] = (f'价格 > {col}', df['Close'] > df[col])
    
    # 均线偏离
    if 'MA20_Deviation' in df.columns:
        factors['ma20_dev_above_5pct'] = ('偏离MA20 > 5%', df['MA20_Deviation'] > 0.05)
        factors['ma20_dev_above_10pct'] = ('偏离MA20 > 10%', df['MA20_Deviation'] > 0.10)
    
    if 'MA50_Deviation' in df.columns:
        factors['ma50_dev_above_10pct'] = ('偏离MA50 > 10%', df['MA50_Deviation'] > 0.10)
        factors['ma50_dev_above_15pct'] = ('偏离MA50 > 15%', df['MA50_Deviation'] > 0.15)
    
    # 成交量
    if 'Volume_Ratio' in df.columns:
        factors['volume_above_avg'] = ('成交量 > 均量', df['Volume_Ratio'] > 1)
        factors['volume_above_1.5x'] = ('成交量 > 1.5倍均量', df['Volume_Ratio'] > 1.5)
        factors['volume_above_2x'] = ('成交量 > 2倍均量', df['Volume_Ratio'] > 2)
    
    # 涨跌幅
    if 'Daily_Return' in df.columns:
        factors['daily_gain_above_2pct'] = ('日涨幅 > 2%', df['Daily_Return'] > 0.02)
        factors['daily_gain_above_3pct'] = ('日涨幅 > 3%', df['Daily_Return'] > 0.03)
        factors['daily_gain_above_5pct'] = ('日涨幅 > 5%', df['Daily_Return'] > 0.05)
    
    if 'Weekly_Return' in df.columns:
        factors['weekly_gain_above_5pct'] = ('周涨幅 > 5%', df['Weekly_Return'] > 0.05)
        factors['weekly_gain_above_10pct'] = ('周涨幅 > 10%', df['Weekly_Return'] > 0.1)
    
    # Aroon
    aroonu_cols = [c for c in df.columns if 'AROONU' in c]
    for col in aroonu_cols:
        factors[f'{col}_above_80'] = (f'{col} > 80', df[col] > 80)
    
    # Ultimate Oscillator
    uo_cols = [c for c in df.columns if c.startswith('UO_')]
    for col in uo_cols:
        factors[f'{col}_above_70'] = (f'{col} > 70 (超买)', df[col] > 70)
    
    # TSI
    tsi_cols = [c for c in df.columns if c.startswith('TSI_')]
    for col in tsi_cols:
        factors[f'{col}_above_25'] = (f'{col} > 25', df[col] > 25)
    
    return factors


def analyze_stock(symbol: str, years: int = 5) -> tuple:
    """分析单只股票"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    try:
        data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), 
                          end=end_date.strftime('%Y-%m-%d'), progress=False, auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if data.empty or len(data) < 100:
            return None, None
        
        # 计算所有技术指标
        df = calculate_all_indicators(data)
        
        # 找出卖点
        sell_point_dates = find_local_highs(df, lookback_days=10, lookforward_days=10)
        
        return df, sell_point_dates
        
    except Exception as e:
        print(f"\n  {symbol} 错误: {e}")
        return None, None


def analyze_factors_at_sell_points(df: pd.DataFrame, sell_point_dates: list) -> tuple:
    """分析卖点处各因子的出现情况"""
    factor_counts = Counter()
    total_sell_points = 0
    
    # 获取因子定义
    factor_defs = get_factor_definitions(df)
    
    for date in sell_point_dates:
        if date not in df.index:
            continue
        
        total_sell_points += 1
        idx = df.index.get_loc(date)
        
        for factor_name, (desc, condition_series) in factor_defs.items():
            try:
                if isinstance(condition_series, pd.Series) and len(condition_series) > idx:
                    if condition_series.iloc[idx] == True:
                        factor_counts[factor_name] += 1
            except Exception:
                pass
    
    return factor_counts, total_sell_points, factor_defs


def run_full_analysis(years: int = 15, max_stocks: int = None):
    """运行完整分析"""
    print("=" * 70)
    print("卖点因子分析 (使用 pandas-ta)")
    print("=" * 70)
    print(f"卖点定义: 前后两周(10个交易日)最高价的那天")
    print(f"分析年数: {years}年")
    print("=" * 70)
    
    # 获取股票池
    stocks = get_stock_pool('all')
    if max_stocks:
        stocks = stocks[:max_stocks]
    
    print(f"股票池: {len(stocks)} 只股票")
    print("-" * 70)
    
    # 汇总统计
    total_factor_counts = Counter()
    total_sell_points = 0
    analyzed_stocks = 0
    all_factor_defs = {}
    
    for i, symbol in enumerate(stocks):
        print(f"\r分析进度: {i+1}/{len(stocks)} - {symbol}    ", end='', flush=True)
        
        df, sell_point_dates = analyze_stock(symbol, years)
        
        if df is None or not sell_point_dates:
            continue
        
        factor_counts, num_sell_points, factor_defs = analyze_factors_at_sell_points(df, sell_point_dates)
        
        total_factor_counts.update(factor_counts)
        total_sell_points += num_sell_points
        analyzed_stocks += 1
        all_factor_defs.update({k: v[0] for k, v in factor_defs.items()})
    
    print(f"\r分析完成! 共分析 {analyzed_stocks} 只股票, {total_sell_points} 个卖点, {len(total_factor_counts)} 个因子")
    print("=" * 70)
    
    # 计算频率并排序
    factor_frequencies = {}
    for factor_name, count in total_factor_counts.items():
        freq = count / total_sell_points if total_sell_points > 0 else 0
        desc = all_factor_defs.get(factor_name, factor_name)
        factor_frequencies[factor_name] = {
            'description': desc,
            'count': count,
            'frequency': freq
        }
    
    # 按频率排序
    sorted_factors = sorted(factor_frequencies.items(), 
                           key=lambda x: x[1]['frequency'], 
                           reverse=True)
    
    # 输出结果
    print(f"\n{'排名':<4} {'因子名称':<35} {'出现频率':>10} {'出现次数':>10}")
    print("-" * 70)
    
    for rank, (name, data) in enumerate(sorted_factors[:30], 1):
        desc = data['description'][:32] + '...' if len(data['description']) > 35 else data['description']
        print(f"{rank:<4} {desc:<35} {data['frequency']:>9.1%} {data['count']:>10}")
    
    print("\n" + "=" * 70)
    print("Top 10 因子 (卖点出现频率最高)")
    print("=" * 70)
    
    top10 = sorted_factors[:10]
    for rank, (name, data) in enumerate(top10, 1):
        print(f"{rank:>2}. {data['description']} - {data['frequency']:.1%} ({data['count']}次)")
    
    # 保存结果
    result_df = pd.DataFrame([
        {
            'rank': i+1,
            'factor_name': name,
            'description': data['description'],
            'frequency': data['frequency'],
            'count': data['count']
        }
        for i, (name, data) in enumerate(sorted_factors)
    ])
    result_df.to_csv('sell_factor_analysis_pta.csv', index=False)
    print(f"\n完整结果已保存至 sell_factor_analysis_pta.csv")
    
    return sorted_factors, total_sell_points


if __name__ == '__main__':
    sorted_factors, total = run_full_analysis(years=5, max_stocks=None)
