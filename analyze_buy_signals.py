"""
买点特征因子分析
定义：买点 = 之后1个月涨幅超过10%的点
目标：找出这些买点的共同技术特征
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os
import warnings
warnings.filterwarnings('ignore')

# 纳斯达克100成分股（部分代表性股票）
NASDAQ_100 = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'GOOG', 'AVGO', 'COST',
    'PEP', 'CSCO', 'ADBE', 'NFLX', 'AMD', 'CMCSA', 'INTC', 'INTU', 'QCOM', 'TXN',
    'AMGN', 'HON', 'AMAT', 'BKNG', 'ISRG', 'SBUX', 'MDLZ', 'GILD', 'ADI', 'VRTX',
    'ADP', 'REGN', 'LRCX', 'PANW', 'MU', 'SNPS', 'KLAC', 'CDNS', 'MELI', 'ASML',
    'PYPL', 'MAR', 'ORLY', 'MNST', 'CTAS', 'MRVL', 'ABNB', 'FTNT', 'NXPI', 'KDP',
    'LULU', 'PCAR', 'WDAY', 'CPRT', 'ROST', 'PAYX', 'AEP', 'ODFL', 'KHC', 'MCHP',
    'IDXX', 'DXCM', 'EXC', 'FAST', 'VRSK', 'EA', 'CTSH', 'XEL', 'BKR', 'GEHC',
    'CSGP', 'FANG', 'ON', 'ANSS', 'DLTR', 'WBD', 'ZS', 'ILMN', 'ALGN', 'TTWO',
    'WBA', 'SIRI', 'JD', 'LCID', 'RIVN', 'DDOG', 'TEAM', 'CRWD', 'ZM', 'DOCU',
    'SPLK', 'OKTA', 'SNOW', 'NET', 'MDB', 'COIN', 'HOOD', 'RBLX', 'U', 'PATH'
]


def load_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """加载股票数据，优先使用缓存"""
    cache_dir = '.cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{symbol}_{start_date}_{end_date}.csv')
    
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if len(df) > 0:
            return df
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        if len(df) > 0:
            df.to_csv(cache_file)
        return df
    except Exception as e:
        print(f"  获取 {symbol} 数据失败: {e}")
        return pd.DataFrame()


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标"""
    if len(df) < 60:
        return df
    
    # 价格相关
    df['return_1d'] = df['Close'].pct_change()
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_10d'] = df['Close'].pct_change(10)
    df['return_20d'] = df['Close'].pct_change(20)
    
    # 均线
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    # 均线偏离度
    df['ma5_bias'] = (df['Close'] - df['MA5']) / df['MA5']
    df['ma10_bias'] = (df['Close'] - df['MA10']) / df['MA10']
    df['ma20_bias'] = (df['Close'] - df['MA20']) / df['MA20']
    df['ma60_bias'] = (df['Close'] - df['MA60']) / df['MA60']
    
    # 均线排列
    df['ma_trend'] = ((df['MA5'] > df['MA10']) & (df['MA10'] > df['MA20'])).astype(int)
    df['ma_trend_down'] = ((df['MA5'] < df['MA10']) & (df['MA10'] < df['MA20'])).astype(int)
    
    # 布林带
    df['BB_mid'] = df['Close'].rolling(20).mean()
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
    df['bb_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    df['below_bb_lower'] = (df['Close'] < df['BB_lower']).astype(int)
    df['below_bb_lower_5pct'] = (df['Close'] < df['BB_lower'] * 0.95).astype(int)
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_oversold'] = (df['RSI'] < 30).astype(int)
    df['RSI_overbought'] = (df['RSI'] > 70).astype(int)
    
    # KDJ
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    df['KDJ_golden_cross'] = ((df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))).astype(int)
    df['KDJ_oversold'] = ((df['K'] < 20) & (df['D'] < 20)).astype(int)
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    df['MACD_golden_cross'] = ((df['MACD'] > df['MACD_signal']) & 
                               (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))).astype(int)
    df['MACD_below_zero'] = (df['MACD'] < 0).astype(int)
    
    # 成交量
    df['vol_ma5'] = df['Volume'].rolling(5).mean()
    df['vol_ma20'] = df['Volume'].rolling(20).mean()
    df['vol_ratio'] = df['Volume'] / df['vol_ma20']
    df['vol_surge'] = (df['vol_ratio'] > 2).astype(int)  # 放量
    df['vol_shrink'] = (df['vol_ratio'] < 0.5).astype(int)  # 缩量
    
    # 波动率
    df['volatility_20d'] = df['return_1d'].rolling(20).std() * np.sqrt(252)
    df['high_volatility'] = (df['volatility_20d'] > df['volatility_20d'].rolling(60).mean()).astype(int)
    
    # 价格位置
    df['high_20d'] = df['High'].rolling(20).max()
    df['low_20d'] = df['Low'].rolling(20).min()
    df['high_60d'] = df['High'].rolling(60).max()
    df['low_60d'] = df['Low'].rolling(60).min()
    df['price_position_20d'] = (df['Close'] - df['low_20d']) / (df['high_20d'] - df['low_20d'])
    df['price_position_60d'] = (df['Close'] - df['low_60d']) / (df['high_60d'] - df['low_60d'])
    df['near_20d_low'] = (df['price_position_20d'] < 0.1).astype(int)
    df['near_60d_low'] = (df['price_position_60d'] < 0.1).astype(int)
    
    # 连续下跌/上涨天数
    df['up_day'] = (df['Close'] > df['Close'].shift(1)).astype(int)
    df['down_day'] = (df['Close'] < df['Close'].shift(1)).astype(int)
    
    # 计算连续下跌天数
    consecutive_down = []
    count = 0
    for i, down in enumerate(df['down_day']):
        if down == 1:
            count += 1
        else:
            count = 0
        consecutive_down.append(count)
    df['consecutive_down_days'] = consecutive_down
    df['down_3_plus_days'] = (df['consecutive_down_days'] >= 3).astype(int)
    df['down_5_plus_days'] = (df['consecutive_down_days'] >= 5).astype(int)
    
    # 跳空缺口
    df['gap_down'] = ((df['High'] < df['Low'].shift(1))).astype(int)
    df['gap_up'] = ((df['Low'] > df['High'].shift(1))).astype(int)
    
    # 大阴线/大阳线
    df['body_pct'] = abs(df['Close'] - df['Open']) / df['Open']
    df['big_red_candle'] = ((df['Close'] < df['Open']) & (df['body_pct'] > 0.03)).astype(int)
    df['big_green_candle'] = ((df['Close'] > df['Open']) & (df['body_pct'] > 0.03)).astype(int)
    
    return df


def find_buy_points(df: pd.DataFrame, future_days: int = 20, min_return: float = 0.10) -> List[int]:
    """找出所有买点（之后N天涨幅超过阈值的点）"""
    buy_points = []
    
    for i in range(len(df) - future_days):
        current_price = df.iloc[i]['Close']
        future_price = df.iloc[i + future_days]['Close']
        future_return = (future_price - current_price) / current_price
        
        if future_return >= min_return:
            buy_points.append(i)
    
    return buy_points


def analyze_stock(symbol: str, start_date: str, end_date: str, 
                  future_days: int = 20, min_return: float = 0.10) -> Dict:
    """分析单只股票的买点特征"""
    df = load_stock_data(symbol, start_date, end_date)
    if len(df) < 100:
        return None
    
    df = calculate_technical_indicators(df)
    buy_points = find_buy_points(df, future_days, min_return)
    
    if len(buy_points) == 0:
        return None
    
    # 提取买点时的特征
    feature_columns = [
        # 均线相关
        'ma5_bias', 'ma10_bias', 'ma20_bias', 'ma60_bias',
        'ma_trend', 'ma_trend_down',
        # 布林带
        'bb_position', 'below_bb_lower', 'below_bb_lower_5pct',
        # RSI
        'RSI', 'RSI_oversold', 'RSI_overbought',
        # KDJ
        'K', 'D', 'J', 'KDJ_golden_cross', 'KDJ_oversold',
        # MACD
        'MACD_golden_cross', 'MACD_below_zero',
        # 成交量
        'vol_ratio', 'vol_surge', 'vol_shrink',
        # 波动率
        'volatility_20d', 'high_volatility',
        # 价格位置
        'price_position_20d', 'price_position_60d', 'near_20d_low', 'near_60d_low',
        # 连续涨跌
        'consecutive_down_days', 'down_3_plus_days', 'down_5_plus_days',
        # 缺口和K线形态
        'gap_down', 'big_red_candle', 'big_green_candle',
        # 历史收益
        'return_5d', 'return_10d', 'return_20d'
    ]
    
    buy_point_features = []
    for idx in buy_points:
        if idx < 60:  # 确保有足够历史数据计算指标
            continue
        row = df.iloc[idx]
        features = {col: row[col] for col in feature_columns if col in df.columns and pd.notna(row[col])}
        features['future_return'] = (df.iloc[idx + future_days]['Close'] - row['Close']) / row['Close']
        features['date'] = df.index[idx]
        features['symbol'] = symbol
        buy_point_features.append(features)
    
    return {
        'symbol': symbol,
        'total_days': len(df),
        'buy_points_count': len(buy_point_features),
        'buy_point_features': buy_point_features
    }


def main():
    """主函数"""
    print("=" * 70)
    print("买点特征因子分析")
    print("定义：买点 = 之后1个月（20交易日）涨幅超过10%的点")
    print("=" * 70)
    
    # 参数设置
    num_stocks = 100  # 分析的股票数量
    start_date = "2015-01-01"
    end_date = "2024-12-30"
    future_days = 20  # 1个月约20个交易日
    min_return = 0.10  # 10%涨幅
    
    print(f"\n分析 {num_stocks} 只纳斯达克100股票")
    print(f"回测期间: {start_date} 至 {end_date}")
    print(f"买点定义: 之后{future_days}天涨幅 >= {min_return*100:.0f}%")
    print("-" * 70)
    
    # 收集所有买点特征
    all_buy_points = []
    stocks_with_signals = 0
    
    for i, symbol in enumerate(NASDAQ_100[:num_stocks]):
        print(f"\r处理 {symbol} ({i+1}/{num_stocks})...", end="", flush=True)
        
        result = analyze_stock(symbol, start_date, end_date, future_days, min_return)
        if result and result['buy_points_count'] > 0:
            all_buy_points.extend(result['buy_point_features'])
            stocks_with_signals += 1
    
    print(f"\n\n找到 {len(all_buy_points)} 个买点，来自 {stocks_with_signals} 只股票")
    
    if len(all_buy_points) == 0:
        print("未找到符合条件的买点")
        return
    
    # 转换为DataFrame
    df_buy_points = pd.DataFrame(all_buy_points)
    
    # ========== 分析特征 ==========
    print("\n" + "=" * 70)
    print("特征因子分析结果")
    print("=" * 70)
    
    # 1. 二值特征的出现频率
    binary_features = [
        'ma_trend', 'ma_trend_down',
        'below_bb_lower', 'below_bb_lower_5pct',
        'RSI_oversold', 'RSI_overbought',
        'KDJ_golden_cross', 'KDJ_oversold',
        'MACD_golden_cross', 'MACD_below_zero',
        'vol_surge', 'vol_shrink',
        'high_volatility',
        'near_20d_low', 'near_60d_low',
        'down_3_plus_days', 'down_5_plus_days',
        'gap_down', 'big_red_candle', 'big_green_candle'
    ]
    
    print("\n【二值特征出现频率】（买点中该特征为True的比例）")
    print("-" * 50)
    
    feature_freq = {}
    for feat in binary_features:
        if feat in df_buy_points.columns:
            freq = df_buy_points[feat].mean() * 100
            feature_freq[feat] = freq
    
    # 按频率排序
    sorted_freq = sorted(feature_freq.items(), key=lambda x: x[1], reverse=True)
    
    feature_descriptions = {
        'ma_trend': '均线多头排列(MA5>MA10>MA20)',
        'ma_trend_down': '均线空头排列(MA5<MA10<MA20)',
        'below_bb_lower': '价格低于布林下轨',
        'below_bb_lower_5pct': '价格低于布林下轨5%',
        'RSI_oversold': 'RSI超卖(<30)',
        'RSI_overbought': 'RSI超买(>70)',
        'KDJ_golden_cross': 'KDJ金叉',
        'KDJ_oversold': 'KDJ超卖区(K,D<20)',
        'MACD_golden_cross': 'MACD金叉',
        'MACD_below_zero': 'MACD在零轴下方',
        'vol_surge': '成交量放大(>2倍均量)',
        'vol_shrink': '成交量萎缩(<0.5倍均量)',
        'high_volatility': '高波动率',
        'near_20d_low': '接近20日低点(<10%位置)',
        'near_60d_low': '接近60日低点(<10%位置)',
        'down_3_plus_days': '连续下跌3天以上',
        'down_5_plus_days': '连续下跌5天以上',
        'gap_down': '向下跳空缺口',
        'big_red_candle': '大阴线(跌幅>3%)',
        'big_green_candle': '大阳线(涨幅>3%)'
    }
    
    for feat, freq in sorted_freq:
        desc = feature_descriptions.get(feat, feat)
        bar = "█" * int(freq / 2)
        print(f"{desc:30s} {freq:5.1f}% {bar}")
    
    # 2. 连续值特征的统计
    print("\n【连续值特征统计】")
    print("-" * 70)
    
    continuous_features = [
        ('RSI', 'RSI值'),
        ('K', 'KDJ-K值'),
        ('bb_position', '布林带位置(0-1)'),
        ('ma20_bias', 'MA20偏离度'),
        ('ma60_bias', 'MA60偏离度'),
        ('price_position_20d', '20日价格位置(0-1)'),
        ('price_position_60d', '60日价格位置(0-1)'),
        ('vol_ratio', '成交量比率'),
        ('return_5d', '过去5日收益'),
        ('return_10d', '过去10日收益'),
        ('return_20d', '过去20日收益'),
        ('consecutive_down_days', '连续下跌天数'),
        ('volatility_20d', '20日波动率')
    ]
    
    print(f"{'特征':20s} {'均值':>10s} {'中位数':>10s} {'25%分位':>10s} {'75%分位':>10s}")
    print("-" * 70)
    
    for feat, desc in continuous_features:
        if feat in df_buy_points.columns:
            data = df_buy_points[feat].dropna()
            if len(data) > 0:
                mean = data.mean()
                median = data.median()
                q25 = data.quantile(0.25)
                q75 = data.quantile(0.75)
                print(f"{desc:20s} {mean:10.2f} {median:10.2f} {q25:10.2f} {q75:10.2f}")
    
    # 3. 特征组合分析
    print("\n【高频特征组合】（同时出现的特征）")
    print("-" * 70)
    
    # 定义一些有意义的特征组合
    combinations = [
        (['RSI_oversold', 'KDJ_oversold'], 'RSI超卖 + KDJ超卖'),
        (['RSI_oversold', 'below_bb_lower'], 'RSI超卖 + 低于布林下轨'),
        (['MACD_golden_cross', 'KDJ_golden_cross'], 'MACD金叉 + KDJ金叉'),
        (['near_20d_low', 'vol_surge'], '接近20日低点 + 放量'),
        (['ma_trend_down', 'RSI_oversold'], '均线空头 + RSI超卖'),
        (['down_3_plus_days', 'RSI_oversold'], '连跌3天+ + RSI超卖'),
        (['MACD_below_zero', 'MACD_golden_cross'], 'MACD零下金叉'),
        (['near_60d_low', 'KDJ_oversold'], '接近60日低点 + KDJ超卖'),
        (['big_red_candle', 'vol_surge'], '放量大阴线'),
        (['ma_trend', 'KDJ_golden_cross'], '均线多头 + KDJ金叉'),
    ]
    
    combo_freq = []
    for features, desc in combinations:
        if all(f in df_buy_points.columns for f in features):
            mask = df_buy_points[features[0]] == 1
            for f in features[1:]:
                mask = mask & (df_buy_points[f] == 1)
            freq = mask.mean() * 100
            combo_freq.append((desc, freq, mask.sum()))
    
    combo_freq.sort(key=lambda x: x[1], reverse=True)
    
    for desc, freq, count in combo_freq:
        bar = "█" * int(freq)
        print(f"{desc:30s} {freq:5.1f}% (n={count:4d}) {bar}")
    
    # 4. 按收益分组分析
    print("\n【按未来收益分组的特征差异】")
    print("-" * 70)
    
    # 分为普通买点(10-20%)和优质买点(>20%)
    df_buy_points['return_group'] = pd.cut(
        df_buy_points['future_return'], 
        bins=[0.10, 0.20, 0.30, 1.0],
        labels=['10-20%', '20-30%', '>30%']
    )
    
    print(f"\n收益分布:")
    print(df_buy_points['return_group'].value_counts().sort_index())
    
    # 比较不同收益组的特征
    print(f"\n不同收益组的特征均值:")
    print("-" * 70)
    
    key_features = ['RSI', 'K', 'bb_position', 'ma20_bias', 'return_20d', 'vol_ratio']
    for feat in key_features:
        if feat in df_buy_points.columns:
            group_means = df_buy_points.groupby('return_group')[feat].mean()
            print(f"\n{feat}:")
            for group, mean in group_means.items():
                print(f"  {group}: {mean:.3f}")
    
    # 5. 总结最重要的特征
    print("\n" + "=" * 70)
    print("【关键发现总结】")
    print("=" * 70)
    
    # 找出频率最高的特征
    top_features = sorted_freq[:5]
    print("\n出现频率最高的5个特征（买点的共同特征）:")
    for i, (feat, freq) in enumerate(top_features, 1):
        desc = feature_descriptions.get(feat, feat)
        print(f"  {i}. {desc}: {freq:.1f}%")
    
    # 连续值特征的典型范围
    print("\n买点时的典型指标范围:")
    for feat, desc in [('RSI', 'RSI'), ('K', 'KDJ-K'), ('bb_position', '布林带位置')]:
        if feat in df_buy_points.columns:
            data = df_buy_points[feat].dropna()
            print(f"  {desc}: {data.quantile(0.25):.1f} - {data.quantile(0.75):.1f} (中位数: {data.median():.1f})")
    
    # 保存结果
    output_file = 'reports/buy_signal_analysis.csv'
    os.makedirs('reports', exist_ok=True)
    df_buy_points.to_csv(output_file, index=False)
    print(f"\n详细数据已保存到: {output_file}")
    
    # 绘制图表
    plot_analysis(df_buy_points, feature_freq, feature_descriptions)


def plot_analysis(df: pd.DataFrame, feature_freq: Dict, feature_descriptions: Dict):
    """绘制分析图表"""
    import matplotlib.pyplot as plt
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('买点特征因子分析\n(买点定义: 之后1月涨幅>10%)', fontsize=14, fontweight='bold')
    
    # 1. 二值特征频率
    ax1 = axes[0, 0]
    sorted_freq = sorted(feature_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    features = [feature_descriptions.get(f, f)[:15] for f, _ in sorted_freq]
    freqs = [f for _, f in sorted_freq]
    colors = ['green' if f > 30 else 'steelblue' for f in freqs]
    bars = ax1.barh(range(len(features)), freqs, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(features)))
    ax1.set_yticklabels(features, fontsize=9)
    ax1.set_xlabel('出现频率 (%)')
    ax1.set_title('买点中各特征出现频率 (Top 10)')
    ax1.invert_yaxis()
    for bar, freq in zip(bars, freqs):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{freq:.1f}%', va='center', fontsize=9)
    
    # 2. RSI分布
    ax2 = axes[0, 1]
    if 'RSI' in df.columns:
        ax2.hist(df['RSI'].dropna(), bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(x=30, color='green', linestyle='--', linewidth=2, label='超卖线(30)')
        ax2.axvline(x=70, color='red', linestyle='--', linewidth=2, label='超买线(70)')
        ax2.axvline(x=df['RSI'].median(), color='orange', linestyle='-', linewidth=2, 
                   label=f'中位数({df["RSI"].median():.1f})')
        ax2.set_xlabel('RSI')
        ax2.set_ylabel('买点数量')
        ax2.set_title('买点时RSI分布')
        ax2.legend()
    
    # 3. 布林带位置分布
    ax3 = axes[1, 0]
    if 'bb_position' in df.columns:
        ax3.hist(df['bb_position'].dropna(), bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='green', linestyle='--', linewidth=2, label='下轨')
        ax3.axvline(x=1, color='red', linestyle='--', linewidth=2, label='上轨')
        ax3.axvline(x=df['bb_position'].median(), color='orange', linestyle='-', linewidth=2,
                   label=f'中位数({df["bb_position"].median():.2f})')
        ax3.set_xlabel('布林带位置 (0=下轨, 1=上轨)')
        ax3.set_ylabel('买点数量')
        ax3.set_title('买点时布林带位置分布')
        ax3.legend()
    
    # 4. 过去20日收益分布
    ax4 = axes[1, 1]
    if 'return_20d' in df.columns:
        returns = df['return_20d'].dropna() * 100
        ax4.hist(returns, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax4.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        ax4.axvline(x=returns.median(), color='orange', linestyle='-', linewidth=2,
                   label=f'中位数({returns.median():.1f}%)')
        ax4.set_xlabel('过去20日收益率 (%)')
        ax4.set_ylabel('买点数量')
        ax4.set_title('买点前20日收益分布')
        ax4.legend()
    
    plt.tight_layout()
    
    chart_file = 'reports/buy_signal_analysis_chart.png'
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {chart_file}")
    plt.close()


if __name__ == "__main__":
    main()
