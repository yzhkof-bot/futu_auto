"""
完整卖点因子分析 - 使用 pandas-ta
分析股票池所有股票15年数据
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
from datetime import datetime, timedelta
from collections import Counter
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.stock_pool import get_stock_pool


def find_local_highs(df, lookback_days=10, lookforward_days=10):
    """找出前后N个交易日的最高点"""
    sell_points = []
    high_prices = df['High'].values
    dates = df.index.tolist()
    
    for i in range(lookback_days, len(df) - lookforward_days):
        current_high = high_prices[i]
        window_max = max(high_prices[i-lookback_days:i+lookforward_days+1])
        if current_high >= window_max:
            sell_points.append(dates[i])
    return sell_points


def calculate_indicators(df):
    """计算技术指标"""
    df.ta.rsi(length=14, append=True)
    df.ta.rsi(length=7, append=True)
    df.ta.stoch(append=True)
    df.ta.macd(append=True)
    df.ta.bbands(append=True)
    df.ta.cci(append=True)
    df.ta.willr(append=True)
    df.ta.adx(append=True)
    df.ta.mfi(append=True)
    df.ta.roc(length=10, append=True)
    df.ta.roc(length=20, append=True)
    df.ta.sma(length=5, append=True)
    df.ta.sma(length=10, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.ema(length=12, append=True)
    df.ta.ema(length=26, append=True)
    df.ta.aroon(append=True)
    df.ta.uo(append=True)
    df.ta.atr(append=True)
    
    # 自定义指标
    df['High_20'] = df['High'].rolling(20).max()
    df['Low_20'] = df['Low'].rolling(20).min()
    df['Price_Pos_20'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'])
    
    df['High_50'] = df['High'].rolling(50).max()
    df['Low_50'] = df['Low'].rolling(50).min()
    df['Price_Pos_50'] = (df['Close'] - df['Low_50']) / (df['High_50'] - df['Low_50'])
    
    df['Daily_Return'] = df['Close'].pct_change()
    df['Weekly_Return'] = df['Close'].pct_change(5)
    df['Volume_MA20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
    
    return df


def analyze_factors(df, sell_points):
    """分析卖点因子"""
    counts = Counter()
    total = len(sell_points)
    
    # 定义因子
    factors = {
        # RSI
        'RSI_14 > 70': lambda d: df.loc[d, 'RSI_14'] > 70 if 'RSI_14' in df.columns else False,
        'RSI_14 > 80': lambda d: df.loc[d, 'RSI_14'] > 80 if 'RSI_14' in df.columns else False,
        'RSI_7 > 70': lambda d: df.loc[d, 'RSI_7'] > 70 if 'RSI_7' in df.columns else False,
        'RSI_7 > 80': lambda d: df.loc[d, 'RSI_7'] > 80 if 'RSI_7' in df.columns else False,
        
        # Stochastic
        'Stoch_K > 80': lambda d: df.loc[d, 'STOCHk_14_3_3'] > 80 if 'STOCHk_14_3_3' in df.columns else False,
        'Stoch_K > 90': lambda d: df.loc[d, 'STOCHk_14_3_3'] > 90 if 'STOCHk_14_3_3' in df.columns else False,
        
        # MACD
        'MACD > 0': lambda d: df.loc[d, 'MACD_12_26_9'] > 0 if 'MACD_12_26_9' in df.columns else False,
        'MACD柱 > 0': lambda d: df.loc[d, 'MACDh_12_26_9'] > 0 if 'MACDh_12_26_9' in df.columns else False,
        
        # Williams %R
        'Williams%R > -20': lambda d: df.loc[d, 'WILLR_14'] > -20 if 'WILLR_14' in df.columns else False,
        'Williams%R > -10': lambda d: df.loc[d, 'WILLR_14'] > -10 if 'WILLR_14' in df.columns else False,
        
        # CCI
        'CCI > 100': lambda d: df.loc[d, 'CCI_14_0.015'] > 100 if 'CCI_14_0.015' in df.columns else False,
        'CCI > 200': lambda d: df.loc[d, 'CCI_14_0.015'] > 200 if 'CCI_14_0.015' in df.columns else False,
        
        # ADX
        'ADX > 25': lambda d: df.loc[d, 'ADX_14'] > 25 if 'ADX_14' in df.columns else False,
        'ADX > 40': lambda d: df.loc[d, 'ADX_14'] > 40 if 'ADX_14' in df.columns else False,
        
        # MFI
        'MFI > 80': lambda d: df.loc[d, 'MFI_14'] > 80 if 'MFI_14' in df.columns else False,
        'MFI > 70': lambda d: df.loc[d, 'MFI_14'] > 70 if 'MFI_14' in df.columns else False,
        
        # ROC
        'ROC_10 > 5%': lambda d: df.loc[d, 'ROC_10'] > 5 if 'ROC_10' in df.columns else False,
        'ROC_10 > 10%': lambda d: df.loc[d, 'ROC_10'] > 10 if 'ROC_10' in df.columns else False,
        'ROC_20 > 10%': lambda d: df.loc[d, 'ROC_20'] > 10 if 'ROC_20' in df.columns else False,
        
        # Aroon
        'Aroon_Up > 80': lambda d: df.loc[d, 'AROONU_14'] > 80 if 'AROONU_14' in df.columns else False,
        'Aroon_Up > 90': lambda d: df.loc[d, 'AROONU_14'] > 90 if 'AROONU_14' in df.columns else False,
        
        # 价格位置
        '价格位置20日 > 90%': lambda d: df.loc[d, 'Price_Pos_20'] > 0.9 if 'Price_Pos_20' in df.columns else False,
        '价格位置20日 > 95%': lambda d: df.loc[d, 'Price_Pos_20'] > 0.95 if 'Price_Pos_20' in df.columns else False,
        '价格位置50日 > 90%': lambda d: df.loc[d, 'Price_Pos_50'] > 0.9 if 'Price_Pos_50' in df.columns else False,
        '价格位置50日 > 95%': lambda d: df.loc[d, 'Price_Pos_50'] > 0.95 if 'Price_Pos_50' in df.columns else False,
        
        # 均线
        '价格 > SMA_5': lambda d: df.loc[d, 'Close'] > df.loc[d, 'SMA_5'] if 'SMA_5' in df.columns else False,
        '价格 > SMA_10': lambda d: df.loc[d, 'Close'] > df.loc[d, 'SMA_10'] if 'SMA_10' in df.columns else False,
        '价格 > SMA_20': lambda d: df.loc[d, 'Close'] > df.loc[d, 'SMA_20'] if 'SMA_20' in df.columns else False,
        '价格 > SMA_50': lambda d: df.loc[d, 'Close'] > df.loc[d, 'SMA_50'] if 'SMA_50' in df.columns else False,
        
        # 成交量
        '成交量 > 均量': lambda d: df.loc[d, 'Volume_Ratio'] > 1 if 'Volume_Ratio' in df.columns else False,
        '成交量 > 1.5倍均量': lambda d: df.loc[d, 'Volume_Ratio'] > 1.5 if 'Volume_Ratio' in df.columns else False,
        '成交量 > 2倍均量': lambda d: df.loc[d, 'Volume_Ratio'] > 2 if 'Volume_Ratio' in df.columns else False,
        
        # 涨跌幅
        '日涨幅 > 2%': lambda d: df.loc[d, 'Daily_Return'] > 0.02 if 'Daily_Return' in df.columns else False,
        '日涨幅 > 3%': lambda d: df.loc[d, 'Daily_Return'] > 0.03 if 'Daily_Return' in df.columns else False,
        '周涨幅 > 5%': lambda d: df.loc[d, 'Weekly_Return'] > 0.05 if 'Weekly_Return' in df.columns else False,
        '周涨幅 > 10%': lambda d: df.loc[d, 'Weekly_Return'] > 0.1 if 'Weekly_Return' in df.columns else False,
    }
    
    # BB位置需要特殊处理
    bbu_cols = [c for c in df.columns if 'BBU' in c]
    bbl_cols = [c for c in df.columns if 'BBL' in c]
    
    for d in sell_points:
        if d not in df.index:
            continue
        
        for name, func in factors.items():
            try:
                if func(d):
                    counts[name] += 1
            except:
                pass
        
        # BB位置
        if bbu_cols and bbl_cols:
            try:
                bb_pos = (df.loc[d, 'Close'] - df.loc[d, bbl_cols[0]]) / (df.loc[d, bbu_cols[0]] - df.loc[d, bbl_cols[0]])
                if bb_pos > 0.8:
                    counts['BB位置 > 0.8'] += 1
                if bb_pos > 0.9:
                    counts['BB位置 > 0.9'] += 1
                if bb_pos > 1.0:
                    counts['BB位置 > 1.0 (突破上轨)'] += 1
            except:
                pass
    
    return counts, total


def main():
    print("=" * 70)
    print("卖点因子分析 (pandas-ta)")
    print("=" * 70)
    print("卖点定义: 前后两周(10个交易日)最高价的那天")
    print("分析年数: 15年")
    print("=" * 70)
    
    stocks = get_stock_pool('all')
    print(f"股票池: {len(stocks)} 只股票")
    print("-" * 70)
    
    total_counts = Counter()
    total_sell_points = 0
    analyzed = 0
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15*365)
    
    for i, symbol in enumerate(stocks):
        print(f"\r进度: {i+1}/{len(stocks)} - {symbol}    ", end='', flush=True)
        
        try:
            data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'),
                              end=end_date.strftime('%Y-%m-%d'), 
                              progress=False, auto_adjust=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if data.empty or len(data) < 100:
                continue
            
            df = calculate_indicators(data)
            sell_points = find_local_highs(df)
            
            if not sell_points:
                continue
            
            counts, num = analyze_factors(df, sell_points)
            total_counts.update(counts)
            total_sell_points += num
            analyzed += 1
            
        except Exception as e:
            continue
    
    print(f"\r完成! 分析 {analyzed} 只股票, {total_sell_points} 个卖点")
    print("=" * 70)
    
    # 排序输出
    sorted_factors = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'排名':<4} {'因子':<25} {'频率':>8} {'次数':>8}")
    print("-" * 50)
    for i, (name, count) in enumerate(sorted_factors[:30], 1):
        freq = count / total_sell_points * 100
        print(f"{i:<4} {name:<25} {freq:>7.1f}% {count:>8}")
    
    print("\n" + "=" * 70)
    print("Top 10 卖点因子")
    print("=" * 70)
    for i, (name, count) in enumerate(sorted_factors[:10], 1):
        freq = count / total_sell_points * 100
        print(f"{i:>2}. {name} - {freq:.1f}% ({count}次)")
    
    # 保存
    result = pd.DataFrame([
        {'rank': i+1, 'factor': name, 'frequency': count/total_sell_points, 'count': count}
        for i, (name, count) in enumerate(sorted_factors)
    ])
    result.to_csv('factor_analysis_15y.csv', index=False)
    print(f"\n结果已保存至 factor_analysis_15y.csv")
    
    return sorted_factors


if __name__ == '__main__':
    main()
