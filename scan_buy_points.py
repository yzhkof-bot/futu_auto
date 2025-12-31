"""
扫描纳斯达克100中符合条件的买点，并统计高频特征因子

买点定义：
1. 近一个月涨幅10%以上
2. 后两个月总涨幅15%以上
3. 回撤在买入价的10%以内
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

from src.data.data_processor import DataProcessor

# 纳斯达克100成分股
NASDAQ100 = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'COST',
    'NFLX', 'TMUS', 'ASML', 'CSCO', 'ADBE', 'AMD', 'PEP', 'LIN', 'INTC', 'INTU',
    'TXN', 'CMCSA', 'QCOM', 'AMGN', 'AMAT', 'HON', 'ISRG', 'BKNG', 'SBUX', 'VRTX',
    'GILD', 'ADP', 'MDLZ', 'ADI', 'REGN', 'PANW', 'SNPS', 'LRCX', 'KLAC', 'CDNS',
    'MU', 'MELI', 'PYPL', 'MAR', 'ORLY', 'MNST', 'CTAS', 'NXPI', 'MCHP', 'FTNT',
    'ABNB', 'PCAR', 'KDP', 'AEP', 'PAYX', 'KHC', 'ODFL', 'CPRT', 'CHTR', 'ROST',
    'IDXX', 'DXCM', 'FAST', 'AZN', 'MRNA', 'EA', 'CTSH', 'EXC', 'VRSK', 'CSGP',
    'XEL', 'BKR', 'GEHC', 'FANG', 'TTWO', 'ANSS', 'BIIB', 'ON', 'DLTR', 'WBD',
    'CDW', 'ZS', 'ILMN', 'MDB', 'TEAM', 'DDOG', 'GFS', 'WBA', 'LCID', 'SIRI',
    'CEG', 'CRWD', 'DASH', 'SMCI', 'ARM', 'COIN', 'TTD', 'PDD', 'LULU', 'WDAY'
]


def fetch_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取股票数据"""
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        print(f"获取 {symbol} 数据失败: {e}")
        return pd.DataFrame()


def find_buy_points(symbol: str, data: pd.DataFrame, 
                    month_gain_threshold: float = 0.10,
                    two_month_gain_threshold: float = 0.15,
                    max_drawdown: float = 0.10) -> list:
    """
    查找符合条件的买点
    
    Args:
        symbol: 股票代码
        data: 带技术指标的DataFrame
        month_gain_threshold: 近一月涨幅阈值
        two_month_gain_threshold: 后两月涨幅阈值
        max_drawdown: 最大回撤阈值
    
    Returns:
        买点列表，每个买点包含日期、特征等信息
    """
    buy_points = []
    
    if len(data) < 90:  # 至少需要3个月数据
        return buy_points
    
    # 计算20日涨幅（约一个月）
    data['Month_Gain'] = data['Close'].pct_change(periods=20)
    
    # 遍历每个可能的买点（需要前20天和后40天数据）
    for i in range(20, len(data) - 42):
        current_date = data.index[i]
        current_price = data['Close'].iloc[i]
        
        # 条件1: 近一月涨幅10%以上
        month_gain = data['Month_Gain'].iloc[i]
        if pd.isna(month_gain) or month_gain < month_gain_threshold:
            continue
        
        # 获取后两个月数据（约42个交易日）
        future_data = data.iloc[i:i+43]
        if len(future_data) < 43:
            continue
        
        # 条件2: 后两月总涨幅15%以上
        end_price = future_data['Close'].iloc[-1]
        two_month_gain = (end_price - current_price) / current_price
        if two_month_gain < two_month_gain_threshold:
            continue
        
        # 条件3: 期间回撤在10%以内
        min_price = future_data['Low'].min()
        drawdown = (current_price - min_price) / current_price
        if drawdown > max_drawdown:
            continue
        
        # 符合所有条件，记录买点和特征
        buy_point = {
            'symbol': symbol,
            'date': current_date,
            'price': current_price,
            'month_gain': month_gain,
            'two_month_gain': two_month_gain,
            'max_drawdown': drawdown,
            'features': extract_features(data, i)
        }
        buy_points.append(buy_point)
    
    return buy_points


def extract_features(data: pd.DataFrame, idx: int) -> dict:
    """提取买点时刻的特征因子"""
    row = data.iloc[idx]
    features = {}
    
    # RSI特征
    if 'RSI_14' in data.columns:
        rsi = row['RSI_14']
        if not pd.isna(rsi):
            features['RSI_14'] = rsi
            if rsi < 30:
                features['RSI_oversold'] = True
            elif rsi > 70:
                features['RSI_overbought'] = True
            elif 40 <= rsi <= 60:
                features['RSI_neutral'] = True
    
    # MACD特征
    if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
        macd = row['MACD']
        signal = row['MACD_Signal']
        if not pd.isna(macd) and not pd.isna(signal):
            features['MACD'] = macd
            features['MACD_Signal'] = signal
            if macd > signal:
                features['MACD_bullish'] = True
            if macd > 0:
                features['MACD_positive'] = True
            # MACD金叉（当前MACD>Signal，前一天MACD<Signal）
            if idx > 0:
                prev_macd = data['MACD'].iloc[idx-1]
                prev_signal = data['MACD_Signal'].iloc[idx-1]
                if not pd.isna(prev_macd) and not pd.isna(prev_signal):
                    if macd > signal and prev_macd <= prev_signal:
                        features['MACD_golden_cross'] = True
    
    # 布林带特征
    if 'BB_Position_20' in data.columns:
        bb_pos = row['BB_Position_20']
        if not pd.isna(bb_pos):
            features['BB_Position'] = bb_pos
            if bb_pos < 0.2:
                features['BB_near_lower'] = True
            elif bb_pos > 0.8:
                features['BB_near_upper'] = True
            elif 0.4 <= bb_pos <= 0.6:
                features['BB_middle'] = True
    
    # 均线特征
    close = row['Close']
    for ma in ['SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200']:
        if ma in data.columns and not pd.isna(row[ma]):
            features[f'above_{ma}'] = close > row[ma]
    
    # 均线排列
    if all(ma in data.columns for ma in ['SMA_5', 'SMA_10', 'SMA_20', 'SMA_50']):
        sma5, sma10, sma20, sma50 = row['SMA_5'], row['SMA_10'], row['SMA_20'], row['SMA_50']
        if not any(pd.isna([sma5, sma10, sma20, sma50])):
            if sma5 > sma10 > sma20 > sma50:
                features['MA_bullish_alignment'] = True
            elif sma5 < sma10 < sma20 < sma50:
                features['MA_bearish_alignment'] = True
    
    # 成交量特征
    if 'Volume_Ratio' in data.columns:
        vol_ratio = row['Volume_Ratio']
        if not pd.isna(vol_ratio):
            features['Volume_Ratio'] = vol_ratio
            if vol_ratio > 1.5:
                features['Volume_high'] = True
            elif vol_ratio < 0.5:
                features['Volume_low'] = True
    
    # 波动率特征
    if 'Volatility_20' in data.columns:
        vol = row['Volatility_20']
        if not pd.isna(vol):
            features['Volatility_20'] = vol
            if vol > 0.4:
                features['High_volatility'] = True
            elif vol < 0.2:
                features['Low_volatility'] = True
    
    # 动量特征
    if 'Momentum_10' in data.columns:
        mom = row['Momentum_10']
        if not pd.isna(mom):
            features['Momentum_10'] = mom
            if mom > 0.05:
                features['Strong_momentum'] = True
    
    if 'Momentum_20' in data.columns:
        mom20 = row['Momentum_20']
        if not pd.isna(mom20):
            features['Momentum_20'] = mom20
    
    # Stochastic特征
    if 'Stoch_K_14' in data.columns and 'Stoch_D_14' in data.columns:
        stoch_k = row['Stoch_K_14']
        stoch_d = row['Stoch_D_14']
        if not pd.isna(stoch_k) and not pd.isna(stoch_d):
            features['Stoch_K'] = stoch_k
            features['Stoch_D'] = stoch_d
            if stoch_k < 20:
                features['Stoch_oversold'] = True
            elif stoch_k > 80:
                features['Stoch_overbought'] = True
            if stoch_k > stoch_d:
                features['Stoch_bullish'] = True
    
    # ADX趋势强度
    if 'ADX' in data.columns:
        adx = row['ADX']
        if not pd.isna(adx):
            features['ADX'] = adx
            if adx > 25:
                features['Strong_trend'] = True
            elif adx < 20:
                features['Weak_trend'] = True
    
    # ATR特征
    if 'ATR_14' in data.columns:
        atr = row['ATR_14']
        if not pd.isna(atr):
            features['ATR_14'] = atr
            features['ATR_pct'] = atr / close  # ATR占价格比例
    
    # 价格位置特征（相对于52周高低点）
    if idx >= 252:
        high_52w = data['High'].iloc[idx-252:idx+1].max()
        low_52w = data['Low'].iloc[idx-252:idx+1].min()
        price_position = (close - low_52w) / (high_52w - low_52w) if high_52w != low_52w else 0.5
        features['Price_52w_position'] = price_position
        if price_position > 0.9:
            features['Near_52w_high'] = True
        elif price_position < 0.1:
            features['Near_52w_low'] = True
    
    # 近期涨幅特征
    if 'Month_Gain' in data.columns:
        features['Month_Gain'] = row['Month_Gain']
    
    return features


def analyze_feature_frequency(buy_points: list) -> dict:
    """分析买点特征出现频率"""
    feature_counter = Counter()
    numeric_features = defaultdict(list)
    total_points = len(buy_points)
    
    for bp in buy_points:
        features = bp['features']
        for key, value in features.items():
            if isinstance(value, bool) and value:
                feature_counter[key] += 1
            elif isinstance(value, (int, float)) and not pd.isna(value):
                numeric_features[key].append(value)
    
    # 计算布尔特征频率
    bool_feature_freq = {k: v / total_points for k, v in feature_counter.items()}
    
    # 计算数值特征统计
    numeric_feature_stats = {}
    for key, values in numeric_features.items():
        if values:
            numeric_feature_stats[key] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
    
    return {
        'bool_features': bool_feature_freq,
        'numeric_features': numeric_feature_stats,
        'total_buy_points': total_points
    }


def main():
    print("=" * 80)
    print("纳斯达克100买点扫描与特征分析")
    print("=" * 80)
    print("\n买点定义:")
    print("  1. 近一个月涨幅 >= 10%")
    print("  2. 后两个月总涨幅 >= 15%")
    print("  3. 期间最大回撤 <= 10%")
    print("\n" + "=" * 80)
    
    # 获取近5年数据
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    print(f"\n数据范围: {start_date} 至 {end_date}")
    print(f"扫描股票数量: {len(NASDAQ100)}")
    print("\n开始扫描...\n")
    
    processor = DataProcessor()
    all_buy_points = []
    symbol_buy_counts = {}
    
    for i, symbol in enumerate(NASDAQ100):
        print(f"[{i+1}/{len(NASDAQ100)}] 处理 {symbol}...", end=' ')
        
        # 获取数据
        data = fetch_stock_data(symbol, start_date, end_date)
        if data.empty or len(data) < 100:
            print("数据不足，跳过")
            continue
        
        # 添加技术指标
        data = processor.add_technical_indicators(data)
        data = processor.add_regime_indicators(data)
        
        # 查找买点
        buy_points = find_buy_points(symbol, data)
        
        if buy_points:
            all_buy_points.extend(buy_points)
            symbol_buy_counts[symbol] = len(buy_points)
            print(f"找到 {len(buy_points)} 个买点")
        else:
            print("无符合条件的买点")
    
    print("\n" + "=" * 80)
    print("扫描完成!")
    print("=" * 80)
    
    if not all_buy_points:
        print("\n未找到任何符合条件的买点")
        return
    
    # 分析结果
    print(f"\n总共找到 {len(all_buy_points)} 个买点")
    print(f"涉及 {len(symbol_buy_counts)} 只股票")
    
    # 按股票统计
    print("\n" + "-" * 40)
    print("各股票买点数量（前20名）:")
    print("-" * 40)
    sorted_counts = sorted(symbol_buy_counts.items(), key=lambda x: x[1], reverse=True)
    for symbol, count in sorted_counts[:20]:
        print(f"  {symbol}: {count} 个买点")
    
    # 特征频率分析
    analysis = analyze_feature_frequency(all_buy_points)
    
    print("\n" + "=" * 80)
    print("高频布尔特征（出现频率 > 30%）:")
    print("=" * 80)
    sorted_bool = sorted(analysis['bool_features'].items(), key=lambda x: x[1], reverse=True)
    for feature, freq in sorted_bool:
        if freq > 0.30:
            print(f"  {feature}: {freq*100:.1f}%")
    
    print("\n" + "=" * 80)
    print("数值特征统计:")
    print("=" * 80)
    for feature, stats in analysis['numeric_features'].items():
        if stats['count'] > len(all_buy_points) * 0.5:  # 至少50%的买点有该特征
            print(f"\n  {feature}:")
            print(f"    均值: {stats['mean']:.4f}")
            print(f"    中位数: {stats['median']:.4f}")
            print(f"    标准差: {stats['std']:.4f}")
            print(f"    范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # 保存详细结果
    results_df = pd.DataFrame([
        {
            'symbol': bp['symbol'],
            'date': bp['date'],
            'price': bp['price'],
            'month_gain': bp['month_gain'],
            'two_month_gain': bp['two_month_gain'],
            'max_drawdown': bp['max_drawdown'],
            **{k: v for k, v in bp['features'].items() if not isinstance(v, bool)}
        }
        for bp in all_buy_points
    ])
    
    results_df.to_csv('reports/buy_points_analysis.csv', index=False)
    print(f"\n详细结果已保存至 reports/buy_points_analysis.csv")
    
    # 打印特征组合建议
    print("\n" + "=" * 80)
    print("高频特征因子组合（买点共性特征）:")
    print("=" * 80)
    
    high_freq_features = [f for f, freq in sorted_bool if freq > 0.50]
    if high_freq_features:
        print("\n超过50%买点具有的特征:")
        for f in high_freq_features:
            print(f"  - {f}")
    
    # 分析RSI分布
    if 'RSI_14' in analysis['numeric_features']:
        rsi_stats = analysis['numeric_features']['RSI_14']
        print(f"\nRSI_14 分布: 均值={rsi_stats['mean']:.1f}, 中位数={rsi_stats['median']:.1f}")
    
    # 分析动量分布
    if 'Momentum_10' in analysis['numeric_features']:
        mom_stats = analysis['numeric_features']['Momentum_10']
        print(f"Momentum_10 分布: 均值={mom_stats['mean']*100:.1f}%, 中位数={mom_stats['median']*100:.1f}%")
    
    return all_buy_points, analysis


if __name__ == '__main__':
    buy_points, analysis = main()
