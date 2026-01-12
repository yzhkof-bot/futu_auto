"""
Top 5 高频因子组合策略 - 卖点识别

基于卖点分析结果，选取出现频率最高的5个因子：
1. MACD正值 (90.06%) - 多头趋势中见顶
2. 布林带上轨突破 (73.11%) - 价格接近布林带上轨
3. 价格接近50日高点 (60.03%) - 价格位置 > 90%
4. KDJ超买 (48.41%) - K值 > 80
5. RSI超买 (47.46%) - RSI > 70

卖点定义：5个因子同时满足
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Top5SellStrategy:
    """
    Top 5 高频因子组合策略 - 卖点识别
    
    因子：
    1. F1_MACD_positive: MACD > 0 (多头趋势中见顶)
    2. F2_BB_upper: 价格接近布林带上轨 (bb_position > 0.8)
    3. F3_Price_high_50d: 价格接近50日高点 (位置 > 90%)
    4. F4_KDJ_overbought: KDJ超买 (K > 80)
    5. F5_RSI_overbought: RSI超买 (RSI > 70)
    
    卖点：5个因子同时满足
    """
    
    FACTOR_NAMES = [
        'F1_MACD_positive',
        'F2_BB_upper',
        'F3_Price_high_50d',
        'F4_KDJ_overbought',
        'F5_RSI_overbought'
    ]
    
    FACTOR_DESCRIPTIONS = {
        'F1_MACD_positive': 'MACD > 0 (多头见顶)',
        'F2_BB_upper': '布林带上轨突破 (BB位置 > 0.8)',
        'F3_Price_high_50d': '价格接近50日高点 (> 90%)',
        'F4_KDJ_overbought': 'KDJ超买 (K > 80)',
        'F5_RSI_overbought': 'RSI超买 (RSI > 70)'
    }
    
    FACTOR_FREQUENCIES = {
        'F1_MACD_positive': 0.9006,
        'F2_BB_upper': 0.7311,
        'F3_Price_high_50d': 0.6003,
        'F4_KDJ_overbought': 0.4841,
        'F5_RSI_overbought': 0.4746
    }
    
    def __init__(self, min_factors: int = 5):
        """
        初始化策略
        
        Args:
            min_factors: 最少需要满足的因子数量（默认5个，即全部因子同时满足）
        """
        self.min_factors = max(1, min(5, min_factors))
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            添加了技术指标的 DataFrame
        """
        df = data.copy()
        
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
        
        # 价格位置 (50日)
        df['High_50'] = df['High'].rolling(window=50).max()
        df['Low_50'] = df['Low'].rolling(window=50).min()
        df['Price_Position_50'] = (df['Close'] - df['Low_50']) / (df['High_50'] - df['Low_50'])
        
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
        
        # 均线
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        return df
    
    def calculate_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算5个因子并生成卖出信号
        
        Args:
            data: OHLCV DataFrame（需包含 Open, High, Low, Close, Volume）
        
        Returns:
            添加了因子和信号的 DataFrame
        """
        if data.empty or len(data) < 60:
            return pd.DataFrame()
        
        df = self.calculate_indicators(data)
        
        # 因子1: MACD正值 (多头趋势中见顶)
        df['F1_MACD_positive'] = df['MACD'] > 0
        
        # 因子2: 布林带上轨突破 (BB位置 > 0.8)
        df['F2_BB_upper'] = df['BB_Position'] > 0.8
        
        # 因子3: 价格接近50日高点 (位置 > 90%)
        df['F3_Price_high_50d'] = df['Price_Position_50'] > 0.9
        
        # 因子4: KDJ超买 (K > 80)
        df['F4_KDJ_overbought'] = df['KDJ_K'] > 80
        
        # 因子5: RSI超买 (RSI > 70)
        df['F5_RSI_overbought'] = df['RSI'] > 70
        
        # 计算满足的因子数量
        df['Factor_Count'] = df[self.FACTOR_NAMES].sum(axis=1)
        
        # 生成卖出信号
        df['Sell_Signal'] = df['Factor_Count'] >= self.min_factors
        
        return df
    
    def get_satisfied_factors(self, row: pd.Series) -> List[str]:
        """
        获取满足条件的因子名称列表
        
        Args:
            row: DataFrame 的一行
        
        Returns:
            满足条件的因子简称列表
        """
        factors = []
        if row.get('F1_MACD_positive', False): factors.append('MACD+')
        if row.get('F2_BB_upper', False): factors.append('BB上轨')
        if row.get('F3_Price_high_50d', False): factors.append('50日高')
        if row.get('F4_KDJ_overbought', False): factors.append('KDJ超买')
        if row.get('F5_RSI_overbought', False): factors.append('RSI超买')
        return factors
    
    def fetch_and_calculate(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取股票数据并计算因子
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        
        Returns:
            带因子的 DataFrame
        """
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if data.empty or len(data) < 60:
                return pd.DataFrame()
            
            return self.calculate_factors(data)
        except Exception as e:
            print(f"获取 {symbol} 数据失败: {e}")
            return pd.DataFrame()
    
    def get_sell_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        获取所有卖出信号
        
        Args:
            df: 带因子的 DataFrame
        
        Returns:
            卖出信号的 DataFrame
        """
        if df.empty or 'Sell_Signal' not in df.columns:
            return pd.DataFrame()
        return df[df['Sell_Signal']].copy()
    
    def find_sell_points(self, symbol: str, start_date: str = None, 
                        end_date: str = None, data: pd.DataFrame = None,
                        cooldown_days: int = 0) -> List[Dict]:
        """
        查找股票的所有卖点
        
        Args:
            symbol: 股票代码
            start_date: 开始日期（默认15年前）
            end_date: 结束日期（默认今天）
            data: 可选，直接传入 OHLCV 数据
            cooldown_days: 冷却期天数，卖点后N天内不再触发新卖点（默认0，不冷却）
        
        Returns:
            卖点列表，每个卖点包含日期、价格、因子等信息
        """
        # 设置默认日期
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=15*365)).strftime('%Y-%m-%d')
        
        # 获取数据并计算因子
        if data is not None:
            df = self.calculate_factors(data)
        else:
            df = self.fetch_and_calculate(symbol, start_date, end_date)
        
        if df.empty:
            return []
        
        # 提取卖点
        sell_points = []
        sell_signals = self.get_sell_signals(df)
        
        last_sell_idx = -999  # 上次卖点的索引位置
        
        for date, row in sell_signals.iterrows():
            current_idx = df.index.get_loc(date)
            
            # 检查冷却期
            if cooldown_days > 0 and (current_idx - last_sell_idx) < cooldown_days:
                continue
            
            sell_points.append({
                'symbol': symbol,
                'date': date,
                'price': row['Close'],
                'factor_count': int(row['Factor_Count']),
                'factors': self.get_satisfied_factors(row),
                'MACD': row.get('MACD', None),
                'RSI': row.get('RSI', None),
                'KDJ_K': row.get('KDJ_K', None),
                'BB_Position': row.get('BB_Position', None),
                'Price_Position_50': row.get('Price_Position_50', None)
            })
            last_sell_idx = current_idx
        
        return sell_points
    
    def print_sell_points(self, sell_points: List[Dict]):
        """
        打印卖点列表
        
        Args:
            sell_points: 卖点列表
        """
        if not sell_points:
            print("无卖点")
            return
        
        print(f"\n{'日期':<12} {'价格':>10} {'因子数':>6} {'满足因子':<40} {'RSI':>6} {'KDJ_K':>6}")
        print("-" * 90)
        
        for sp in sell_points:
            date_str = str(sp['date'])[:10]
            factors_str = ', '.join(sp['factors'])
            rsi = f"{sp['RSI']:.1f}" if sp['RSI'] else 'N/A'
            kdj_k = f"{sp['KDJ_K']:.1f}" if sp['KDJ_K'] else 'N/A'
            
            print(f"{date_str:<12} ${sp['price']:>9.2f} {sp['factor_count']:>6} "
                  f"{factors_str:<40} {rsi:>6} {kdj_k:>6}")
        
        print(f"\n共 {len(sell_points)} 个卖点")
    
    def plot_sell_points(self, symbol: str, df: pd.DataFrame = None, 
                        sell_points: List[Dict] = None,
                        start_date: str = None, end_date: str = None,
                        save_path: str = None):
        """
        绘制卖点图表
        
        Args:
            symbol: 股票代码
            df: 带因子的 DataFrame（可选）
            sell_points: 卖点列表（可选）
            start_date: 开始日期
            end_date: 结束日期
            save_path: 保存路径（可选）
        """
        # 获取数据
        if df is None:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=15*365)).strftime('%Y-%m-%d')
            df = self.fetch_and_calculate(symbol, start_date, end_date)
        
        if df.empty:
            print("无数据")
            return
        
        # 获取卖点
        if sell_points is None:
            sell_points = self.find_sell_points(symbol, start_date, end_date, data=df)
        
        # 创建图表
        fig, axes = plt.subplots(4, 1, figsize=(16, 14), 
                                  gridspec_kw={'height_ratios': [3, 1, 1, 1]},
                                  sharex=True)
        fig.suptitle(f'{symbol} - Top 5 卖出因子策略 (满足 >= {self.min_factors} 个因子)', 
                     fontsize=14, fontweight='bold')
        
        # 图1: 价格和卖点
        ax1 = axes[0]
        close_data = df['Close'].dropna()
        ax1.plot(close_data.index, close_data.values, label='收盘价', color='steelblue', linewidth=1)
        
        # 绘制均线
        if 'SMA_20' in df.columns:
            sma20 = df['SMA_20'].dropna()
            ax1.plot(sma20.index, sma20.values, label='MA20', color='orange', 
                     linewidth=0.8, alpha=0.7)
        if 'SMA_50' in df.columns:
            sma50 = df['SMA_50'].dropna()
            ax1.plot(sma50.index, sma50.values, label='MA50', color='green', 
                     linewidth=0.8, alpha=0.7)
        
        # 绘制布林带
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            bb_upper = df['BB_Upper'].dropna()
            bb_lower = df['BB_Lower'].dropna()
            ax1.fill_between(bb_upper.index, bb_lower.values, bb_upper.values, 
                            alpha=0.1, color='gray', label='布林带')
        
        # 标记卖点
        if sell_points:
            sell_dates = [sp['date'] for sp in sell_points]
            sell_prices = [sp['price'] for sp in sell_points]
            ax1.scatter(sell_dates, sell_prices, marker='v', color='red', 
                       s=80, label=f'卖出信号 ({len(sell_points)})', zorder=5)
        
        ax1.set_ylabel('价格 ($)')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('价格走势与卖出信号', fontsize=11, loc='left')
        
        # 图2: RSI
        ax2 = axes[1]
        if 'RSI' in df.columns:
            rsi = df['RSI'].dropna()
            ax2.plot(rsi.index, rsi.values, label='RSI(14)', color='purple', linewidth=1)
            ax2.axhline(70, color='red', linestyle='--', linewidth=0.8, label='超买线(70)')
            ax2.axhline(30, color='green', linestyle='--', linewidth=0.8, label='超卖线(30)')
            ax2.fill_between(rsi.index, 70, rsi.values, where=(rsi.values > 70), 
                            alpha=0.3, color='red')
            
            # 标记卖点
            if sell_points:
                for sp in sell_points:
                    if sp['date'] in df.index:
                        ax2.axvline(sp['date'], color='red', alpha=0.2, linewidth=0.8)
        
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('RSI指标', fontsize=11, loc='left')
        
        # 图3: KDJ
        ax3 = axes[2]
        if 'KDJ_K' in df.columns and 'KDJ_D' in df.columns:
            kdj_k = df['KDJ_K'].dropna()
            kdj_d = df['KDJ_D'].dropna()
            ax3.plot(kdj_k.index, kdj_k.values, label='K', color='blue', linewidth=1)
            ax3.plot(kdj_d.index, kdj_d.values, label='D', color='orange', linewidth=1)
            ax3.axhline(80, color='red', linestyle='--', linewidth=0.8, label='超买线(80)')
            ax3.axhline(20, color='green', linestyle='--', linewidth=0.8, label='超卖线(20)')
            ax3.fill_between(kdj_k.index, 80, kdj_k.values, where=(kdj_k.values > 80), 
                            alpha=0.3, color='red')
            
            # 标记卖点
            if sell_points:
                for sp in sell_points:
                    if sp['date'] in df.index:
                        ax3.axvline(sp['date'], color='red', alpha=0.2, linewidth=0.8)
        
        ax3.set_ylabel('KDJ')
        ax3.set_ylim(0, 100)
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_title('KDJ指标', fontsize=11, loc='left')
        
        # 图4: MACD
        ax4 = axes[3]
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            macd = df['MACD'].dropna()
            macd_signal = df['MACD_Signal'].dropna()
            ax4.plot(macd.index, macd.values, label='MACD', color='blue', linewidth=1)
            ax4.plot(macd_signal.index, macd_signal.values, label='Signal', color='orange', linewidth=1)
            
            # 柱状图
            macd_hist = df['MACD_Histogram'].dropna()
            colors = ['green' if x >= 0 else 'red' for x in macd_hist.values]
            ax4.bar(macd_hist.index, macd_hist.values, color=colors, alpha=0.5, width=1)
            ax4.axhline(0, color='black', linewidth=0.5)
            
            # 标记卖点
            if sell_points:
                for sp in sell_points:
                    if sp['date'] in df.index:
                        ax4.axvline(sp['date'], color='red', alpha=0.2, linewidth=0.8)
        
        ax4.set_ylabel('MACD')
        ax4.legend(loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.set_title('MACD指标', fontsize=11, loc='left')
        ax4.set_xlabel('日期')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至 {save_path}")
        
        plt.show()
        
        return fig


# 便捷函数
def create_strategy(min_factors: int = 3) -> Top5SellStrategy:
    """创建策略实例"""
    return Top5SellStrategy(min_factors=min_factors)


def find_sell_points(symbol: str, min_factors: int = 5,
                    start_date: str = None, end_date: str = None,
                    cooldown_days: int = 0,
                    print_result: bool = True,
                    plot: bool = False,
                    save_path: str = None) -> List[Dict]:
    """
    快速查找股票卖点
    
    Args:
        symbol: 股票代码
        min_factors: 最少满足因子数（默认5，即全部因子同时满足）
        start_date: 开始日期
        end_date: 结束日期
        cooldown_days: 冷却期天数（默认0，不冷却）
        print_result: 是否打印结果
        plot: 是否绘制图表
        save_path: 图表保存路径
    
    Returns:
        卖点列表
    """
    strategy = Top5SellStrategy(min_factors=min_factors)
    sell_points = strategy.find_sell_points(symbol, start_date, end_date, 
                                            cooldown_days=cooldown_days)
    
    if print_result:
        print(f"\n{'='*70}")
        print(f"Top 5 卖出因子策略: {symbol}")
        print(f"{'='*70}")
        print(f"因子要求: >= {min_factors} 个")
        print(f"因子列表 (按出现频率排序):")
        for name, desc in strategy.FACTOR_DESCRIPTIONS.items():
            freq = strategy.FACTOR_FREQUENCIES.get(name, 0)
            print(f"  - {desc} ({freq:.1%})")
        strategy.print_sell_points(sell_points)
    
    if plot:
        strategy.plot_sell_points(symbol, start_date=start_date, end_date=end_date,
                                 sell_points=sell_points, save_path=save_path)
    
    return sell_points


if __name__ == '__main__':
    # 示例：查找 AAPL 卖点并绘图（5个因子同时满足）
    sell_points = find_sell_points('AAPL', min_factors=5, plot=True, 
                                   save_path='reports/aapl_sell_points.png')