"""
超强精英卖点策略 - 15个因子中满足13个以上

基于Top 10因子 + 5个精英因子，要求满足13个以上（87%以上）：

Top 10因子（基于前后两周最高点分析）：
1. 价格 > SMA_10 (95.5%)
2. 价格 > SMA_20 (94.2%)  
3. 价格 > SMA_5 (88.2%)
4. Aroon_Up > 80 (88.1%)
5. Aroon_Up > 90 (88.1%)
6. MACD柱 > 0 (85.1%)
7. 价格 > SMA_50 (83.9%)
8. MACD > 0 (79.4%)
9. Williams%R > -20 (64.8%)
10. Stoch_K > 80 (64.3%)

额外5个精英因子：
11. RSI_14 > 70 (超买)
12. CCI > 100 (商品通道指数超买)
13. MFI > 80 (资金流量指数超买)
14. ROC_10 > 5% (10日变化率)
15. 价格位置50日 > 95% (接近50日高点)

策略逻辑：15个因子中满足13个以上 = 超强精准卖点
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

class UltraEliteSellStrategy:
    """
    超强精英卖点策略 - 15个因子中满足13个以上
    
    Top 10 因子：
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
    
    额外5个精英因子：
    11. F11_RSI_overbought: RSI_14 > 70
    12. F12_CCI_overbought: CCI > 100
    13. F13_MFI_overbought: MFI > 80
    14. F14_ROC_strong: ROC_10 > 5%
    15. F15_Price_near_high: 价格位置50日 > 95%
    
    卖点：15个因子中满足13个以上（87%以上）
    """
    
    def __init__(self, min_factors: int = 13):
        """
        初始化策略
        
        Args:
            min_factors: 最少需要满足的因子数量（默认13个，范围10-15）
        """
        self.min_factors = max(10, min(15, min_factors))
        self.factor_names = [
            # Top 10 因子
            'F1_Price_above_SMA10', 'F2_Price_above_SMA20', 'F3_Price_above_SMA5',
            'F4_Aroon_Up_80', 'F5_Aroon_Up_90', 'F6_MACD_Histogram_positive',
            'F7_Price_above_SMA50', 'F8_MACD_positive', 'F9_Williams_overbought', 
            'F10_Stoch_overbought',
            # 额外5个精英因子
            'F11_RSI_overbought', 'F12_CCI_overbought', 'F13_MFI_overbought',
            'F14_ROC_strong', 'F15_Price_near_high'
        ]
    
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
        
        # RSI
        df.ta.rsi(length=14, append=True)
        
        # CCI (手动计算确保正确)
        tp = (df['High'] + df['Low'] + df['Close']) / 3  # Typical Price
        sma_tp = tp.rolling(14).mean()
        mad = tp.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean())
        df['CCI_14'] = (tp - sma_tp) / (0.015 * mad)
        
        # MFI
        df.ta.mfi(append=True)
        
        # ROC
        df.ta.roc(length=10, append=True)
        
        # 价格位置指标
        df['High_50'] = df['High'].rolling(50).max()
        df['Low_50'] = df['Low'].rolling(50).min()
        df['Price_Pos_50'] = (df['Close'] - df['Low_50']) / (df['High_50'] - df['Low_50'])
        
        return df
    
    def calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算15个因子"""
        
        # Top 10 因子
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
        
        # 额外5个精英因子
        # F11: RSI_14 > 70
        df['F11_RSI_overbought'] = df['RSI_14'] > 70
        
        # F12: CCI > 100 (恢复原始阈值)
        df['F12_CCI_overbought'] = df['CCI_14'] > 100
        
        # F13: MFI > 80
        mfi_col = [c for c in df.columns if c.startswith('MFI')][0]
        df['F13_MFI_overbought'] = df[mfi_col] > 80
        
        # F14: ROC_10 > 5%
        df['F14_ROC_strong'] = df['ROC_10'] > 5
        
        # F15: 价格位置50日 > 95%
        df['F15_Price_near_high'] = df['Price_Pos_50'] > 0.95
        
        return df
    
    def identify_sell_points(self, df: pd.DataFrame) -> List[Dict]:
        """识别超强精英卖点 - 15个因子中满足min_factors个以上"""
        
        # 计算每日满足的因子数量
        df['factor_count'] = df[self.factor_names].sum(axis=1)
        
        # 超强精英卖点：满足min_factors个以上因子
        df['ultra_elite_sell_signal'] = df['factor_count'] >= self.min_factors
        
        # 提取卖点信息
        sell_points = []
        for idx, row in df[df['ultra_elite_sell_signal']].iterrows():
            factor_details = {}
            for i, name in enumerate(self.factor_names, 1):
                factor_details[f'F{i}'] = bool(row[name])
            
            sell_points.append({
                'date': idx,
                'price': row['Close'],
                'factor_count': int(row['factor_count']),
                'factors': factor_details
            })
        
        return sell_points


def find_ultra_elite_sell_points(symbol: str, 
                                 min_factors: int = 13,
                                 start_date: str = None, end_date: str = None,
                                 cooldown_days: int = 7,  # 默认7天冷却期
                                 print_result: bool = True,
                                 plot: bool = False,
                                 save_path: str = None) -> List[Dict]:
    """
    查找超强精英卖点
    
    Args:
        symbol: 股票代码
        min_factors: 最少满足因子数（默认13个，范围10-15）
        start_date: 开始日期
        end_date: 结束日期
        cooldown_days: 冷却期天数（默认7天）
        print_result: 是否打印结果
        plot: 是否绘制图表
        save_path: 图表保存路径
    
    Returns:
        超强精英卖点列表
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
        strategy = UltraEliteSellStrategy(min_factors=min_factors)
        
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
            print(f"\n⚡ {symbol} 超强精英卖点分析")
            print(f"📅 分析区间: {start_date} ~ {end_date}")
            print(f"🔥 策略要求: 15个因子中满足{min_factors}个以上 ({min_factors/15*100:.1f}%)")
            print(f"⏰ 冷却期: {cooldown_days}天")
            print(f"📊 超强精英卖点数: {len(sell_points)}")
            
            if sell_points:
                print(f"\n🔴 超强精英卖点列表:")
                for i, point in enumerate(sell_points[-10:], 1):  # 显示最近10个
                    print(f"  {i:>2}. {point['date'].strftime('%Y-%m-%d')}: ${point['price']:.2f} "
                          f"({point['factor_count']}/15因子)")
                
                # 统计年份分布
                yearly_stats = {}
                for point in sell_points:
                    year = point['date'].year
                    yearly_stats[year] = yearly_stats.get(year, 0) + 1
                
                if yearly_stats:
                    print(f"\n📈 按年份分布:")
                    for year in sorted(yearly_stats.keys()):
                        print(f"  {year}: {yearly_stats[year]} 个")
                
                # 因子满足度统计
                factor_counts = [point['factor_count'] for point in sell_points]
                print(f"\n📊 因子满足度分布:")
                for count in sorted(set(factor_counts), reverse=True):
                    num = factor_counts.count(count)
                    print(f"  {count}/15因子: {num}个卖点 ({num/len(sell_points)*100:.1f}%)")
            else:
                print(f"🎯 未找到满足{min_factors}个因子条件的超强精英卖点")
                print("💡 建议：可以尝试降低因子要求数量")
        
        # 绘制图表
        if plot:
            plot_ultra_elite_signals(symbol, df, sell_points, min_factors, save_path)
        
        return sell_points
        
    except Exception as e:
        if print_result:
            print(f"❌ {symbol}: 分析失败 - {e}")
        return []


def plot_ultra_elite_signals(symbol: str, df: pd.DataFrame, sell_points: list, 
                             min_factors: int = 13, save_path: str = None):
    """绘制超强精英卖点图表"""
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 第1图：价格和卖点
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], label='Close Price', linewidth=1.5, color='black')
    ax1.plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.6, color='blue')
    ax1.plot(df.index, df['SMA_50'], label='SMA 50', alpha=0.6, color='red')
    
    # 标记超强精英卖点
    if sell_points:
        sell_dates = [p['date'] for p in sell_points]
        sell_prices = [p['price'] for p in sell_points]
        factor_counts = [p['factor_count'] for p in sell_points]
        
        # 根据因子数量设置颜色
        colors = ['red' if fc >= 14 else 'orange' if fc >= 13 else 'yellow' for fc in factor_counts]
        
        ax1.scatter(sell_dates, sell_prices, c=colors, s=80, 
                   label=f'Ultra Elite Sell Points ({len(sell_points)})', 
                   zorder=5, marker='v', edgecolors='black', linewidth=1)
    
    ax1.set_title(f'{symbol} - Ultra Elite Sell Strategy ({min_factors}+/15 Factors)')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 第2图：因子计数
    ax2 = axes[1]
    ax2.plot(df.index, df['factor_count'], label='Factor Count', color='blue', linewidth=1)
    ax2.axhline(y=min_factors, color='red', linestyle='--', linewidth=2, 
               label=f'Ultra Elite Threshold ({min_factors})')
    ax2.axhline(y=15, color='purple', linestyle=':', alpha=0.7, label='Perfect (15)')
    ax2.fill_between(df.index, df['factor_count'], min_factors, 
                    where=(df['factor_count'] >= min_factors), 
                    color='red', alpha=0.3, label='Ultra Elite Zone')
    
    ax2.set_ylabel('Factor Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 15)
    
    # 第3图：关键指标组合
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    
    # RSI和MFI
    ax3.plot(df.index, df['RSI_14'], label='RSI 14', color='purple', alpha=0.7)
    ax3.axhline(y=70, color='purple', linestyle='--', alpha=0.5)
    
    # ROC
    ax3_twin.plot(df.index, df['ROC_10'], label='ROC 10', color='green', alpha=0.7)
    ax3_twin.axhline(y=5, color='green', linestyle='--', alpha=0.5)
    
    ax3.set_ylabel('RSI')
    ax3_twin.set_ylabel('ROC (%)')
    ax3.set_xlabel('Date')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 图表已保存: {save_path}")
    
    plt.show()


if __name__ == '__main__':
    # 示例：查找 AAPL 超强精英卖点并绘图
    sell_points = find_ultra_elite_sell_points('AAPL', min_factors=13, cooldown_days=7, plot=True, 
                                              save_path='reports/aapl_ultra_elite_sell_points.png')