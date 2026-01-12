"""
Top 10 高频因子组合策略 - 卖点识别

基于前后两周最高点分析结果，选取出现频率最高的10个因子：
1. 价格 > SMA_10 (95.5%) - 短期均线上方
2. 价格 > SMA_20 (94.2%) - 中期均线上方  
3. 价格 > SMA_5 (88.2%) - 超短期均线上方
4. Aroon_Up > 80 (88.1%) - 上升趋势强劲
5. Aroon_Up > 90 (88.1%) - 上升趋势极强
6. MACD柱 > 0 (85.1%) - 多头动能
7. 价格 > SMA_50 (83.9%) - 长期均线上方
8. MACD > 0 (79.4%) - 多头趋势
9. Williams%R > -20 (64.8%) - 超买状态
10. Stoch_K > 80 (64.3%) - 随机指标超买

卖点定义：满足N个或以上因子（默认7个）
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

class Top10SellStrategy:
    """
    Top 10 高频因子组合策略 - 卖点识别
    
    因子：
    1. F1_Price_above_SMA10: 价格 > SMA_10
    2. F2_Price_above_SMA20: 价格 > SMA_20
    3. F3_Price_above_SMA5: 价格 > SMA_5
    4. F4_Aroon_Up_80: Aroon_Up > 80
    5. F5_Aroon_Up_90: Aroon_Up > 90
    6. F6_MACD_Histogram_positive: MACD柱 > 0
    7. F7_Price_above_SMA50: 价格 > SMA_50
    8. F8_MACD_positive: MACD > 0
    9. F9_Williams_overbought: Williams%R > -20
    10. F10_Stoch_overbought: Stoch_K > 80
    
    卖点：满足 >= min_factors 个因子（默认7个）
    """
    
    def __init__(self, min_factors: int = 7):
        """
        初始化策略
        
        Args:
            min_factors: 最少需要满足的因子数量（默认7个，范围1-10）
        """
        self.min_factors = max(1, min(10, min_factors))
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所需技术指标"""
        # 均线
        df.ta.sma(length=5, append=True)
        df.ta.sma(length=10, append=True)
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=50, append=True)
        
        # Aroon
        df.ta.aroon(append=True)
        
        # MACD
        df.ta.macd(append=True)
        
        # Williams %R
        df.ta.willr(append=True)
        
        # Stochastic
        df.ta.stoch(append=True)
        
        return df
    
    def calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算10个因子"""
        
        # F1: 价格 > SMA_10
        df['F1_Price_above_SMA10'] = df['Close'] > df['SMA_10']
        
        # F2: 价格 > SMA_20
        df['F2_Price_above_SMA20'] = df['Close'] > df['SMA_20']
        
        # F3: 价格 > SMA_5
        df['F3_Price_above_SMA5'] = df['Close'] > df['SMA_5']
        
        # F4: Aroon_Up > 80
        aroon_up_col = [c for c in df.columns if 'AROONU' in c][0]
        df['F4_Aroon_Up_80'] = df[aroon_up_col] > 80
        
        # F5: Aroon_Up > 90
        df['F5_Aroon_Up_90'] = df[aroon_up_col] > 90
        
        # F6: MACD柱 > 0
        macd_hist_col = [c for c in df.columns if 'MACDh' in c][0]
        df['F6_MACD_Histogram_positive'] = df[macd_hist_col] > 0
        
        # F7: 价格 > SMA_50
        df['F7_Price_above_SMA50'] = df['Close'] > df['SMA_50']
        
        # F8: MACD > 0
        macd_col = [c for c in df.columns if c.startswith('MACD_')][0]
        df['F8_MACD_positive'] = df[macd_col] > 0
        
        # F9: Williams%R > -20
        willr_col = [c for c in df.columns if 'WILLR' in c][0]
        df['F9_Williams_overbought'] = df[willr_col] > -20
        
        # F10: Stoch_K > 80
        stoch_k_col = [c for c in df.columns if 'STOCHk' in c][0]
        df['F10_Stoch_overbought'] = df[stoch_k_col] > 80
        
        return df
    
    def identify_sell_points(self, df: pd.DataFrame) -> List[Dict]:
        """识别卖点"""
        
        # 因子列
        factor_cols = [f'F{i}_{name}' for i, name in enumerate([
            'Price_above_SMA10', 'Price_above_SMA20', 'Price_above_SMA5',
            'Aroon_Up_80', 'Aroon_Up_90', 'MACD_Histogram_positive',
            'Price_above_SMA50', 'MACD_positive', 'Williams_overbought', 'Stoch_overbought'
        ], 1)]
        
        # 计算每日满足的因子数量
        df['factor_count'] = df[factor_cols].sum(axis=1)
        
        # 识别卖点
        df['sell_signal'] = df['factor_count'] >= self.min_factors
        
        # 提取卖点信息
        sell_points = []
        for idx, row in df[df['sell_signal']].iterrows():
            sell_points.append({
                'date': idx,
                'price': row['Close'],
                'factor_count': int(row['factor_count']),
                'factors': {col: bool(row[col]) for col in factor_cols}
            })
        
        return sell_points


def find_sell_points(symbol: str, min_factors: int = 7,
                    start_date: str = None, end_date: str = None,
                    cooldown_days: int = 0,
                    print_result: bool = True,
                    plot: bool = False,
                    save_path: str = None) -> List[Dict]:
    """
    快速查找股票卖点
    
    Args:
        symbol: 股票代码
        min_factors: 最少满足因子数（默认7）
        start_date: 开始日期
        end_date: 结束日期
        cooldown_days: 冷却期天数（默认0，不冷却）
        print_result: 是否打印结果
        plot: 是否绘制图表
        save_path: 图表保存路径
    
    Returns:
        卖点列表
    """
    
    # 下载数据
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=15*365)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        data = yf.download(symbol, start=start_date, end=end_date, 
                          progress=False, auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if data.empty or len(data) < 100:
            if print_result:
                print(f"❌ {symbol}: 数据不足")
            return []
        
        # 初始化策略
        strategy = Top10SellStrategy(min_factors=min_factors)
        
        # 计算指标
        df = strategy.calculate_indicators(data.copy())
        
        # 计算因子
        df = strategy.calculate_factors(df)
        
        # 识别卖点
        sell_points = strategy.identify_sell_points(df)
        
        # 应用冷却期
        if cooldown_days > 0 and len(sell_points) > 1:
            filtered_points = [sell_points[0]]
            for point in sell_points[1:]:
                last_date = filtered_points[-1]['date']
                if (point['date'] - last_date).days >= cooldown_days:
                    filtered_points.append(point)
            sell_points = filtered_points
        
        # 打印结果
        if print_result:
            print(f"\n📊 {symbol} 卖点分析 (Top 10因子策略)")
            print(f"分析区间: {start_date} ~ {end_date}")
            print(f"最少因子数: {min_factors}/10")
            print(f"总卖点数: {len(sell_points)}")
            
            if sell_points:
                print(f"\n最近卖点:")
                for point in sell_points[-5:]:
                    print(f"  {point['date'].strftime('%Y-%m-%d')}: ${point['price']:.2f} "
                          f"({point['factor_count']}/10因子)")
        
        # 绘制图表
        if plot and sell_points:
            plot_sell_signals(symbol, df, sell_points, min_factors, save_path)
        
        return sell_points
        
    except Exception as e:
        if print_result:
            print(f"❌ {symbol}: 分析失败 - {e}")
        return []


def plot_sell_signals(symbol: str, df: pd.DataFrame, sell_points: list, 
                     min_factors: int = 7, save_path: str = None):
    """绘制卖点图表"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 上图：价格和卖点
    ax1.plot(df.index, df['Close'], label='Close Price', linewidth=1)
    ax1.plot(df.index, df['SMA_10'], label='SMA 10', alpha=0.7)
    ax1.plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.7)
    ax1.plot(df.index, df['SMA_50'], label='SMA 50', alpha=0.7)
    
    # 标记卖点
    if sell_points:
        sell_dates = [p['date'] for p in sell_points]
        sell_prices = [p['price'] for p in sell_points]
        ax1.scatter(sell_dates, sell_prices, color='red', s=50, 
                   label=f'Sell Points ({len(sell_points)})', zorder=5)
    
    ax1.set_title(f'{symbol} - Top 10 Factor Sell Strategy (min_factors={min_factors})')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 下图：因子计数
    ax2.plot(df.index, df['factor_count'], label='Factor Count', color='blue')
    ax2.axhline(y=min_factors, color='red', linestyle='--', 
               label=f'Threshold ({min_factors})')
    ax2.fill_between(df.index, df['factor_count'], min_factors, 
                    where=(df['factor_count'] >= min_factors), 
                    color='red', alpha=0.3, label='Sell Zone')
    
    ax2.set_ylabel('Factor Count')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 图表已保存: {save_path}")
    
    plt.show()


if __name__ == '__main__':
    # 示例：查找 AAPL 卖点并绘图（7个因子满足）
    sell_points = find_sell_points('AAPL', min_factors=7, plot=True, 
                                   save_path='reports/aapl_top10_sell_points.png')