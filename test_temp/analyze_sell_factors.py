"""
卖点因子分析工具

卖点定义：前后两周（10个交易日）最高价的那天
分析这些卖点出现最多的技术因子
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from collections import Counter
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.stock_pool import get_stock_pool

# 定义所有待分析的因子
FACTOR_DEFINITIONS = {
    # RSI相关
    'rsi_above_70': ('RSI > 70', lambda df: df['RSI'] > 70),
    'rsi_above_75': ('RSI > 75', lambda df: df['RSI'] > 75),
    'rsi_above_80': ('RSI > 80', lambda df: df['RSI'] > 80),
    
    # KDJ相关
    'kdj_k_above_80': ('KDJ_K > 80', lambda df: df['KDJ_K'] > 80),
    'kdj_k_above_85': ('KDJ_K > 85', lambda df: df['KDJ_K'] > 85),
    'kdj_k_above_90': ('KDJ_K > 90', lambda df: df['KDJ_K'] > 90),
    'kdj_death_cross': ('KDJ死叉(K下穿D)', lambda df: (df['KDJ_K'] < df['KDJ_D']) & (df['KDJ_K'].shift(1) >= df['KDJ_D'].shift(1))),
    
    # MACD相关
    'macd_positive': ('MACD > 0', lambda df: df['MACD'] > 0),
    'macd_histogram_positive': ('MACD柱 > 0', lambda df: df['MACD_Histogram'] > 0),
    'macd_death_cross': ('MACD死叉', lambda df: (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))),
    'macd_histogram_shrinking': ('MACD柱缩短', lambda df: df['MACD_Histogram'] < df['MACD_Histogram'].shift(1)),
    
    # 布林带相关
    'bb_position_above_80': ('BB位置 > 0.8', lambda df: df['BB_Position'] > 0.8),
    'bb_position_above_90': ('BB位置 > 0.9', lambda df: df['BB_Position'] > 0.9),
    'bb_position_above_100': ('BB位置 > 1.0 (突破上轨)', lambda df: df['BB_Position'] > 1.0),
    'price_above_bb_upper': ('价格 > 布林上轨', lambda df: df['Close'] > df['BB_Upper']),
    
    # 价格位置相关
    'price_position_20d_above_90': ('20日价格位置 > 90%', lambda df: df['Price_Position_20'] > 0.9),
    'price_position_50d_above_90': ('50日价格位置 > 90%', lambda df: df['Price_Position_50'] > 0.9),
    'price_position_20d_above_95': ('20日价格位置 > 95%', lambda df: df['Price_Position_20'] > 0.95),
    'price_position_50d_above_95': ('50日价格位置 > 95%', lambda df: df['Price_Position_50'] > 0.95),
    
    # 均线相关
    'price_above_ma5': ('价格 > MA5', lambda df: df['Close'] > df['SMA_5']),
    'price_above_ma10': ('价格 > MA10', lambda df: df['Close'] > df['SMA_10']),
    'price_above_ma20': ('价格 > MA20', lambda df: df['Close'] > df['SMA_20']),
    'price_above_ma50': ('价格 > MA50', lambda df: df['Close'] > df['SMA_50']),
    'ma5_above_ma10': ('MA5 > MA10', lambda df: df['SMA_5'] > df['SMA_10']),
    'ma5_above_ma20': ('MA5 > MA20', lambda df: df['SMA_5'] > df['SMA_20']),
    'price_far_from_ma20': ('价格偏离MA20 > 10%', lambda df: (df['Close'] - df['SMA_20']) / df['SMA_20'] > 0.1),
    'price_far_from_ma50': ('价格偏离MA50 > 15%', lambda df: (df['Close'] - df['SMA_50']) / df['SMA_50'] > 0.15),
    
    # 成交量相关
    'volume_above_avg': ('成交量 > 20日均量', lambda df: df['Volume'] > df['Volume_MA20']),
    'volume_above_avg_150': ('成交量 > 1.5倍均量', lambda df: df['Volume'] > df['Volume_MA20'] * 1.5),
    'volume_above_avg_200': ('成交量 > 2倍均量', lambda df: df['Volume'] > df['Volume_MA20'] * 2),
    'volume_shrinking': ('成交量萎缩', lambda df: df['Volume'] < df['Volume'].shift(1)),
    
    # 涨跌幅相关
    'daily_gain_above_2pct': ('日涨幅 > 2%', lambda df: df['Daily_Return'] > 0.02),
    'daily_gain_above_3pct': ('日涨幅 > 3%', lambda df: df['Daily_Return'] > 0.03),
    'daily_gain_above_5pct': ('日涨幅 > 5%', lambda df: df['Daily_Return'] > 0.05),
    'weekly_gain_above_5pct': ('周涨幅 > 5%', lambda df: df['Weekly_Return'] > 0.05),
    'weekly_gain_above_10pct': ('周涨幅 > 10%', lambda df: df['Weekly_Return'] > 0.1),
    
    # 波动率相关
    'high_volatility': ('波动率 > 均值', lambda df: df['Volatility'] > df['Volatility'].rolling(50).mean()),
    'atr_above_avg': ('ATR > 均值', lambda df: df['ATR'] > df['ATR'].rolling(50).mean()),
    
    # K线形态
    'upper_shadow_long': ('上影线长(>实体)', lambda df: (df['High'] - df[['Open', 'Close']].max(axis=1)) > abs(df['Close'] - df['Open'])),
    'doji': ('十字星', lambda df: abs(df['Close'] - df['Open']) < (df['High'] - df['Low']) * 0.1),
    'bearish_candle': ('阴线', lambda df: df['Close'] < df['Open']),
    
    # 连续上涨
    'up_3_days': ('连涨3天', lambda df: (df['Close'] > df['Close'].shift(1)) & (df['Close'].shift(1) > df['Close'].shift(2)) & (df['Close'].shift(2) > df['Close'].shift(3))),
    'up_5_days': ('连涨5天', lambda df: (df['Close'] > df['Close'].shift(1)) & (df['Close'].shift(1) > df['Close'].shift(2)) & (df['Close'].shift(2) > df['Close'].shift(3)) & (df['Close'].shift(3) > df['Close'].shift(4)) & (df['Close'].shift(4) > df['Close'].shift(5))),
}


def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """计算所有技术指标"""
    df = data.copy()
    
    # 基础计算
    df['Daily_Return'] = df['Close'].pct_change()
    df['Weekly_Return'] = df['Close'].pct_change(5)
    
    # 均线
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # 成交量均线
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # KDJ
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['KDJ_K'] = rsv.ewm(com=2, adjust=False).mean()
    df['KDJ_D'] = df['KDJ_K'].ewm(com=2, adjust=False).mean()
    df['KDJ_J'] = 3 * df['KDJ_K'] - 2 * df['KDJ_D']
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # 布林带
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # 价格位置
    df['High_20'] = df['High'].rolling(window=20).max()
    df['Low_20'] = df['Low'].rolling(window=20).min()
    df['Price_Position_20'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'])
    
    df['High_50'] = df['High'].rolling(window=50).max()
    df['Low_50'] = df['Low'].rolling(window=50).min()
    df['Price_Position_50'] = (df['Close'] - df['Low_50']) / (df['High_50'] - df['Low_50'])
    
    # 波动率
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    
    # ATR
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    return df


def find_local_highs(df: pd.DataFrame, lookback_days: int = 10, lookforward_days: int = 10) -> list:
    """
    找出前后N个交易日的最高点
    
    Args:
        df: 带有High列的DataFrame
        lookback_days: 向前看的天数（默认10个交易日≈2周）
        lookforward_days: 向后看的天数（默认10个交易日≈2周）
    
    Returns:
        卖点日期列表
    """
    sell_points = []
    high_prices = df['High'].values
    dates = df.index.tolist()
    
    for i in range(lookback_days, len(df) - lookforward_days):
        current_high = high_prices[i]
        
        # 检查是否是前后N天的最高点
        window_start = i - lookback_days
        window_end = i + lookforward_days + 1
        window_max = max(high_prices[window_start:window_end])
        
        if current_high >= window_max:
            sell_points.append(dates[i])
    
    return sell_points


def analyze_stock(symbol: str, years: int = 5) -> tuple:
    """分析单只股票"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    try:
        data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), 
                          end=end_date.strftime('%Y-%m-%d'), progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if data.empty or len(data) < 60:
            return None, None
        
        # 计算指标
        df = calculate_all_indicators(data)
        
        # 找出卖点（前后两周最高点）
        sell_point_dates = find_local_highs(df, lookback_days=10, lookforward_days=10)
        
        return df, sell_point_dates
        
    except Exception as e:
        print(f"  {symbol} 分析失败: {e}")
        return None, None


def analyze_factors_at_sell_points(df: pd.DataFrame, sell_point_dates: list) -> dict:
    """分析卖点处各因子的出现情况"""
    factor_counts = Counter()
    total_sell_points = 0
    
    for date in sell_point_dates:
        if date not in df.index:
            continue
        
        row = df.loc[date]
        total_sell_points += 1
        
        for factor_name, (desc, condition_func) in FACTOR_DEFINITIONS.items():
            try:
                # 创建单行DataFrame来评估条件
                single_row_df = df.loc[[date]]
                result = condition_func(single_row_df)
                if result.iloc[0] == True:
                    factor_counts[factor_name] += 1
            except Exception:
                pass
    
    return factor_counts, total_sell_points


def run_full_analysis(years: int = 5, max_stocks: int = None):
    """运行完整分析"""
    print("=" * 70)
    print("卖点因子分析")
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
    
    for i, symbol in enumerate(stocks):
        print(f"\r分析进度: {i+1}/{len(stocks)} - {symbol}    ", end='', flush=True)
        
        df, sell_point_dates = analyze_stock(symbol, years)
        
        if df is None or not sell_point_dates:
            continue
        
        factor_counts, num_sell_points = analyze_factors_at_sell_points(df, sell_point_dates)
        
        total_factor_counts.update(factor_counts)
        total_sell_points += num_sell_points
        analyzed_stocks += 1
    
    print(f"\r分析完成! 共分析 {analyzed_stocks} 只股票, {total_sell_points} 个卖点")
    print("=" * 70)
    
    # 计算频率并排序
    factor_frequencies = {}
    for factor_name, count in total_factor_counts.items():
        freq = count / total_sell_points if total_sell_points > 0 else 0
        desc = FACTOR_DEFINITIONS[factor_name][0]
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
    print(f"\n{'排名':<4} {'因子名称':<30} {'出现频率':>10} {'出现次数':>10}")
    print("-" * 70)
    
    for rank, (name, data) in enumerate(sorted_factors[:20], 1):
        print(f"{rank:<4} {data['description']:<30} {data['frequency']:>9.1%} {data['count']:>10}")
    
    print("\n" + "=" * 70)
    print("Top 5 因子")
    print("=" * 70)
    
    top5 = sorted_factors[:5]
    for rank, (name, data) in enumerate(top5, 1):
        print(f"{rank}. {data['description']} - {data['frequency']:.1%} ({data['count']}次)")
    
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
    result_df.to_csv('sell_factor_analysis_result.csv', index=False)
    print(f"\n完整结果已保存至 sell_factor_analysis_result.csv")
    
    return sorted_factors, total_sell_points


if __name__ == '__main__':
    sorted_factors, total = run_full_analysis(years=5, max_stocks=None)
