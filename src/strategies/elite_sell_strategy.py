"""
精英卖点策略 - 基于不同类型高频因子组合

从Top 10因子中选择5个不同类型的代表性因子：
1. 趋势类：Aroon_Up > 90 (88.1%) - 强势上升趋势
2. 动量类：MACD柱 > 0 (85.1%) - 多头动能
3. 超买类：Williams%R > -20 (64.8%) - 超买状态
4. 位置类：价格 > SMA_50 (83.9%) - 长期趋势上方
5. 振荡类：Stoch_K > 80 (64.3%) - 随机指标超买

策略逻辑：5个因子全部同时满足 = 精准卖点
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

class EliteSellStrategy:
    """
    精英卖点策略 - 5个不同类型因子全部满足
    
    因子分类：
    1. F1_Trend_Strong: Aroon_Up > 90 (强势上升趋势)
    2. F2_Momentum_Positive: MACD柱 > 0 (多头动能)
    3. F3_Overbought_Williams: Williams%R > -20 (超买状态)
    4. F4_Position_Above_MA50: 价格 > SMA_50 (长期趋势上方)
    5. F5_Oscillator_Overbought: Stoch_K > 80 (随机指标超买)
    
    卖点：5个因子全部同时满足
    """
    
    def __init__(self):
        """初始化策略"""
        self.factor_names = [
            'F1_Trend_Strong',
            'F2_Momentum_Positive', 
            'F3_Overbought_Williams',
            'F4_Position_Above_MA50',
            'F5_Oscillator_Overbought'
        ]
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所需技术指标"""
        # Aroon - 趋势指标
        df.ta.aroon(append=True)
        
        # MACD - 动量指标
        df.ta.macd(append=True)
        
        # Williams %R - 超买超卖指标
        df.ta.willr(append=True)
        
        # SMA50 - 趋势位置指标
        df.ta.sma(length=50, append=True)
        
        # Stochastic - 振荡指标
        df.ta.stoch(append=True)
        
        return df
    
    def calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算5个精英因子"""
        
        # F1: 强势上升趋势 - Aroon_Up > 90
        aroon_up_col = [c for c in df.columns if 'AROONU' in c][0]
        df['F1_Trend_Strong'] = df[aroon_up_col] > 90
        
        # F2: 多头动能 - MACD柱 > 0
        macd_hist_col = [c for c in df.columns if 'MACDh' in c][0]
        df['F2_Momentum_Positive'] = df[macd_hist_col] > 0
        
        # F3: 超买状态 - Williams%R > -20
        willr_col = [c for c in df.columns if 'WILLR' in c][0]
        df['F3_Overbought_Williams'] = df[willr_col] > -20
        
        # F4: 长期趋势上方 - 价格 > SMA_50
        df['F4_Position_Above_MA50'] = df['Close'] > df['SMA_50']
        
        # F5: 随机指标超买 - Stoch_K > 80
        stoch_k_col = [c for c in df.columns if 'STOCHk' in c][0]
        df['F5_Oscillator_Overbought'] = df[stoch_k_col] > 80
        
        return df
    
    def identify_sell_points(self, df: pd.DataFrame) -> List[Dict]:
        """识别精英卖点 - 5个因子全部满足"""
        
        # 计算每日满足的因子数量
        df['factor_count'] = df[self.factor_names].sum(axis=1)
        
        # 精英卖点：5个因子全部满足
        df['elite_sell_signal'] = df['factor_count'] == 5
        
        # 提取卖点信息
        sell_points = []
        for idx, row in df[df['elite_sell_signal']].iterrows():
            sell_points.append({
                'date': idx,
                'price': row['Close'],
                'factor_count': 5,  # 全部满足
                'factors': {
                    'Trend_Strong': bool(row['F1_Trend_Strong']),
                    'Momentum_Positive': bool(row['F2_Momentum_Positive']),
                    'Overbought_Williams': bool(row['F3_Overbought_Williams']),
                    'Position_Above_MA50': bool(row['F4_Position_Above_MA50']),
                    'Oscillator_Overbought': bool(row['F5_Oscillator_Overbought'])
                }
            })
        
        return sell_points


def find_elite_sell_points(symbol: str, 
                          start_date: str = None, end_date: str = None,
                          cooldown_days: int = 5,  # 默认5天冷却期
                          print_result: bool = True,
                          plot: bool = False,
                          save_path: str = None) -> List[Dict]:
    """
    查找精英卖点
    
    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        cooldown_days: 冷却期天数（默认5天）
        print_result: 是否打印结果
        plot: 是否绘制图表
        save_path: 图表保存路径
    
    Returns:
        精英卖点列表
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
        strategy = EliteSellStrategy()
        
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
            print(f"\n🎯 {symbol} 精英卖点分析")
            print(f"📅 分析区间: {start_date} ~ {end_date}")
            print(f"🔥 策略要求: 5个不同类型因子全部满足")
            print(f"⏰ 冷却期: {cooldown_days}天")
            print(f"📊 精英卖点数: {len(sell_points)}")
            
            if sell_points:
                print(f"\n🔴 精英卖点列表:")
                for i, point in enumerate(sell_points[-10:], 1):  # 显示最近10个
                    print(f"  {i:>2}. {point['date'].strftime('%Y-%m-%d')}: ${point['price']:.2f}")
                
                # 统计年份分布
                yearly_stats = {}
                for point in sell_points:
                    year = point['date'].year
                    yearly_stats[year] = yearly_stats.get(year, 0) + 1
                
                print(f"\n📈 按年份分布:")
                for year in sorted(yearly_stats.keys()):
                    print(f"  {year}: {yearly_stats[year]} 个")
        
        # 绘制图表
        if plot and sell_points:
            plot_elite_signals(symbol, df, sell_points, save_path)
        
        return sell_points
        
    except Exception as e:
        if print_result:
            print(f"❌ {symbol}: 分析失败 - {e}")
        return []


def plot_elite_signals(symbol: str, df: pd.DataFrame, sell_points: list, save_path: str = None):
    """绘制精英卖点图表"""
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    
    # 第1图：价格和卖点
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], label='Close Price', linewidth=1.5, color='black')
    ax1.plot(df.index, df['SMA_50'], label='SMA 50', alpha=0.7, color='red')
    
    # 标记精英卖点
    if sell_points:
        sell_dates = [p['date'] for p in sell_points]
        sell_prices = [p['price'] for p in sell_points]
        ax1.scatter(sell_dates, sell_prices, color='red', s=80, 
                   label=f'Elite Sell Points ({len(sell_points)})', zorder=5, marker='v')
    
    ax1.set_title(f'{symbol} - Elite Sell Strategy (5 Factors All Required)')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 第2图：Aroon Up (趋势)
    ax2 = axes[1]
    aroon_up_col = [c for c in df.columns if 'AROONU' in c][0]
    ax2.plot(df.index, df[aroon_up_col], label='Aroon Up', color='green')
    ax2.axhline(y=90, color='red', linestyle='--', label='Threshold (90)')
    ax2.fill_between(df.index, df[aroon_up_col], 90, 
                    where=(df[aroon_up_col] >= 90), 
                    color='green', alpha=0.3, label='Strong Trend Zone')
    ax2.set_ylabel('Aroon Up')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # 第3图：MACD柱状图 (动量)
    ax3 = axes[2]
    macd_hist_col = [c for c in df.columns if 'MACDh' in c][0]
    ax3.bar(df.index, df[macd_hist_col], label='MACD Histogram', 
           color=['green' if x > 0 else 'red' for x in df[macd_hist_col]], alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_ylabel('MACD Histogram')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 第4图：Williams %R 和 Stoch K (超买指标)
    ax4 = axes[3]
    willr_col = [c for c in df.columns if 'WILLR' in c][0]
    stoch_k_col = [c for c in df.columns if 'STOCHk' in c][0]
    
    ax4_twin = ax4.twinx()
    
    ax4.plot(df.index, df[willr_col], label='Williams %R', color='purple', alpha=0.7)
    ax4.axhline(y=-20, color='purple', linestyle='--', alpha=0.7, label='Williams Threshold (-20)')
    
    ax4_twin.plot(df.index, df[stoch_k_col], label='Stoch K', color='orange', alpha=0.7)
    ax4_twin.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Stoch Threshold (80)')
    
    ax4.set_ylabel('Williams %R')
    ax4_twin.set_ylabel('Stoch K')
    ax4.set_xlabel('Date')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 图表已保存: {save_path}")
    
    plt.show()


if __name__ == '__main__':
    # 示例：查找 AAPL 精英卖点并绘图
    sell_points = find_elite_sell_points('AAPL', cooldown_days=5, plot=True, 
                                        save_path='reports/aapl_elite_sell_points.png')