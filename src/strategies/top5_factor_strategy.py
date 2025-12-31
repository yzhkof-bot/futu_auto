"""
Top 5 高频因子组合策略 - 买点识别

基于买点分析结果，选取出现频率最高的5个因子：
1. MACD正值 (95.9%)
2. 强趋势 ADX>25 (84.1%)
3. MACD多头 MACD>Signal (82.4%)
4. 强动量 10日涨幅>5% (66.3%)
5. 均线多头排列 MA5>MA10>MA20>MA50 (60.3%)

买点定义：同时满足4个或以上因子
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

from src.data.data_processor import DataProcessor

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Top5FactorStrategy:
    """
    Top 5 高频因子组合策略 - 仅识别买点
    
    因子：
    1. MACD_positive: MACD > 0
    2. Strong_trend: ADX > 25
    3. MACD_bullish: MACD > MACD_Signal
    4. Strong_momentum: 10日涨幅 > 5%
    5. MA_bullish_alignment: MA5 > MA10 > MA20 > MA50
    
    买点：满足 >= min_factors 个因子（默认4个）
    """
    
    FACTOR_NAMES = [
        'F1_MACD_positive',
        'F2_Strong_trend', 
        'F3_MACD_bullish',
        'F4_Strong_momentum',
        'F5_MA_bullish'
    ]
    
    FACTOR_DESCRIPTIONS = {
        'F1_MACD_positive': 'MACD > 0',
        'F2_Strong_trend': 'ADX > 25',
        'F3_MACD_bullish': 'MACD > Signal',
        'F4_Strong_momentum': '10日涨幅 > 5%',
        'F5_MA_bullish': 'MA5 > MA10 > MA20 > MA50'
    }
    
    def __init__(self, min_factors: int = 4):
        """
        初始化策略
        
        Args:
            min_factors: 最少需要满足的因子数量（默认4个，范围1-5）
        """
        self.min_factors = max(1, min(5, min_factors))
        self.processor = DataProcessor()
    
    def calculate_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算5个因子并生成买入信号
        
        Args:
            data: OHLCV DataFrame（需包含 Open, High, Low, Close, Volume）
        
        Returns:
            添加了因子和信号的 DataFrame
        """
        if data.empty or len(data) < 60:
            return pd.DataFrame()
        
        df = data.copy()
        
        # 添加技术指标
        df = self.processor.add_technical_indicators(df)
        df = self.processor.add_regime_indicators(df)
        
        # 因子1: MACD正值
        df['F1_MACD_positive'] = df['MACD'] > 0
        
        # 因子2: 强趋势 ADX > 25
        df['F2_Strong_trend'] = df['ADX'] > 25
        
        # 因子3: MACD多头 (MACD > Signal)
        df['F3_MACD_bullish'] = df['MACD'] > df['MACD_Signal']
        
        # 因子4: 强动量 (10日涨幅 > 5%)
        df['Momentum_10'] = df['Close'].pct_change(periods=10)
        df['F4_Strong_momentum'] = df['Momentum_10'] > 0.05
        
        # 因子5: 均线多头排列
        df['F5_MA_bullish'] = (
            (df['SMA_5'] > df['SMA_10']) & 
            (df['SMA_10'] > df['SMA_20']) & 
            (df['SMA_20'] > df['SMA_50'])
        )
        
        # 计算满足的因子数量
        df['Factor_Count'] = df[self.FACTOR_NAMES].sum(axis=1)
        
        # 生成买入信号
        df['Buy_Signal'] = df['Factor_Count'] >= self.min_factors
        
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
        if row.get('F2_Strong_trend', False): factors.append('ADX>25')
        if row.get('F3_MACD_bullish', False): factors.append('MACD>Sig')
        if row.get('F4_Strong_momentum', False): factors.append('Mom>5%')
        if row.get('F5_MA_bullish', False): factors.append('MA多头')
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
    
    def get_buy_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        获取所有买入信号
        
        Args:
            df: 带因子的 DataFrame
        
        Returns:
            买入信号的 DataFrame
        """
        if df.empty or 'Buy_Signal' not in df.columns:
            return pd.DataFrame()
        return df[df['Buy_Signal']].copy()
    
    def find_buy_points(self, symbol: str, start_date: str = None, 
                        end_date: str = None, data: pd.DataFrame = None,
                        cooldown_days: int = 0) -> List[Dict]:
        """
        查找股票的所有买点
        
        Args:
            symbol: 股票代码
            start_date: 开始日期（默认5年前）
            end_date: 结束日期（默认今天）
            data: 可选，直接传入 OHLCV 数据
            cooldown_days: 冷却期天数，买点后N天内不再触发新买点（默认0，不冷却）
        
        Returns:
            买点列表，每个买点包含日期、价格、因子等信息
        """
        # 设置默认日期
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        # 获取数据并计算因子
        if data is not None:
            df = self.calculate_factors(data)
        else:
            df = self.fetch_and_calculate(symbol, start_date, end_date)
        
        if df.empty:
            return []
        
        # 提取买点
        buy_points = []
        buy_signals = self.get_buy_signals(df)
        
        last_buy_idx = -999  # 上次买点的索引位置
        
        for date, row in buy_signals.iterrows():
            current_idx = df.index.get_loc(date)
            
            # 检查冷却期
            if cooldown_days > 0 and (current_idx - last_buy_idx) < cooldown_days:
                continue
            
            buy_points.append({
                'symbol': symbol,
                'date': date,
                'price': row['Close'],
                'factor_count': int(row['Factor_Count']),
                'factors': self.get_satisfied_factors(row),
                'MACD': row.get('MACD', None),
                'ADX': row.get('ADX', None),
                'RSI_14': row.get('RSI_14', None),
                'Momentum_10': row.get('Momentum_10', None)
            })
            last_buy_idx = current_idx
        
        return buy_points
    
    def print_buy_points(self, buy_points: List[Dict]):
        """
        打印买点列表
        
        Args:
            buy_points: 买点列表
        """
        if not buy_points:
            print("无买点")
            return
        
        print(f"\n{'日期':<12} {'价格':>10} {'因子数':>6} {'满足因子':<35} {'RSI':>6} {'ADX':>6}")
        print("-" * 85)
        
        for bp in buy_points:
            date_str = str(bp['date'])[:10]
            factors_str = ', '.join(bp['factors'])
            rsi = f"{bp['RSI_14']:.1f}" if bp['RSI_14'] else 'N/A'
            adx = f"{bp['ADX']:.1f}" if bp['ADX'] else 'N/A'
            
            print(f"{date_str:<12} ${bp['price']:>9.2f} {bp['factor_count']:>6} "
                  f"{factors_str:<35} {rsi:>6} {adx:>6}")
        
        print(f"\n共 {len(buy_points)} 个买点")
    
    def plot_buy_points(self, symbol: str, df: pd.DataFrame = None, 
                        buy_points: List[Dict] = None,
                        start_date: str = None, end_date: str = None,
                        save_path: str = None):
        """
        绘制买点图表
        
        Args:
            symbol: 股票代码
            df: 带因子的 DataFrame（可选）
            buy_points: 买点列表（可选）
            start_date: 开始日期
            end_date: 结束日期
            save_path: 保存路径（可选）
        """
        # 获取数据
        if df is None:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
            df = self.fetch_and_calculate(symbol, start_date, end_date)
        
        if df.empty:
            print("无数据")
            return
        
        # 获取买点
        if buy_points is None:
            buy_points = self.find_buy_points(symbol, start_date, end_date, data=df)
        
        # 创建图表
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), 
                                  gridspec_kw={'height_ratios': [3, 1.2, 1.2]},
                                  sharex=True)
        fig.suptitle(f'{symbol} - Top 5 Factor Strategy Buy Points (min {self.min_factors} factors)', 
                     fontsize=14, fontweight='bold')
        
        # 图1: 价格和买点
        ax1 = axes[0]
        # 过滤掉 NaN 值，确保曲线连续
        close_data = df['Close'].dropna()
        ax1.plot(close_data.index, close_data.values, label='Close', color='steelblue', linewidth=1)
        
        # 绘制均线
        if 'SMA_20' in df.columns:
            sma20 = df['SMA_20'].dropna()
            ax1.plot(sma20.index, sma20.values, label='SMA20', color='orange', 
                     linewidth=0.8, alpha=0.7)
        if 'SMA_50' in df.columns:
            sma50 = df['SMA_50'].dropna()
            ax1.plot(sma50.index, sma50.values, label='SMA50', color='green', 
                     linewidth=0.8, alpha=0.7)
        
        # 标记买点
        if buy_points:
            buy_dates = [bp['date'] for bp in buy_points]
            buy_prices = [bp['price'] for bp in buy_points]
            ax1.scatter(buy_dates, buy_prices, marker='^', color='red', 
                       s=80, label=f'Buy Signal ({len(buy_points)})', zorder=5)
        
        ax1.set_ylabel('Price', fontsize=10)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Price & Buy Signals', fontsize=11, loc='left')
        
        # 图2: MACD
        ax2 = axes[1]
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            macd = df['MACD'].dropna()
            macd_signal = df['MACD_Signal'].dropna()
            ax2.plot(macd.index, macd.values, label='MACD', color='blue', linewidth=1)
            ax2.plot(macd_signal.index, macd_signal.values, label='Signal', color='orange', linewidth=1)
            
            # 柱状图 - 使用有效数据
            macd_hist = df['MACD_Histogram'].dropna()
            colors = ['green' if x >= 0 else 'red' for x in macd_hist.values]
            ax2.bar(macd_hist.index, macd_hist.values, color=colors, alpha=0.5, width=1)
            ax2.axhline(0, color='black', linewidth=0.5)
            
            # 标记买点
            if buy_points:
                for bp in buy_points:
                    if bp['date'] in df.index:
                        ax2.axvline(bp['date'], color='red', alpha=0.2, linewidth=0.8)
        
        ax2.set_ylabel('MACD', fontsize=10)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('MACD', fontsize=11, loc='left')
        
        # 图3: RSI 和 ADX
        ax3 = axes[2]
        if 'RSI_14' in df.columns:
            rsi = df['RSI_14'].dropna()
            ax3.plot(rsi.index, rsi.values, label='RSI(14)', color='purple', linewidth=1)
            ax3.axhline(70, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
            ax3.axhline(30, color='green', linestyle='--', linewidth=0.5, alpha=0.7)
        
        if 'ADX' in df.columns:
            adx = df['ADX'].dropna()
            ax3.plot(adx.index, adx.values, label='ADX', color='brown', linewidth=1)
            ax3.axhline(25, color='orange', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # 标记买点
        if buy_points:
            for bp in buy_points:
                if bp['date'] in df.index:
                    ax3.axvline(bp['date'], color='red', alpha=0.2, linewidth=0.8)
        
        ax3.set_ylabel('RSI / ADX', fontsize=10)
        ax3.set_ylim(0, 100)
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_title('RSI & ADX', fontsize=11, loc='left')
        ax3.set_xlabel('Date', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至 {save_path}")
        
        plt.show()
        
        return fig


# 便捷函数
def create_strategy(min_factors: int = 4) -> Top5FactorStrategy:
    """创建策略实例"""
    return Top5FactorStrategy(min_factors=min_factors)


def find_buy_points(symbol: str, min_factors: int = 4,
                    start_date: str = None, end_date: str = None,
                    cooldown_days: int = 0,
                    print_result: bool = True,
                    plot: bool = False,
                    save_path: str = None) -> List[Dict]:
    """
    快速查找股票买点
    
    Args:
        symbol: 股票代码
        min_factors: 最少满足因子数（默认4）
        start_date: 开始日期
        end_date: 结束日期
        cooldown_days: 冷却期天数（默认0，不冷却）
        print_result: 是否打印结果
        plot: 是否绘制图表
        save_path: 图表保存路径
    
    Returns:
        买点列表
    """
    strategy = Top5FactorStrategy(min_factors=min_factors)
    buy_points = strategy.find_buy_points(symbol, start_date, end_date, 
                                          cooldown_days=cooldown_days)
    
    if print_result:
        print(f"\n{'='*60}")
        print(f"Top 5 因子策略买点: {symbol}")
        print(f"{'='*60}")
        print(f"因子要求: >= {min_factors} 个")
        print(f"因子列表:")
        for name, desc in strategy.FACTOR_DESCRIPTIONS.items():
            print(f"  - {desc}")
        strategy.print_buy_points(buy_points)
    
    if plot:
        strategy.plot_buy_points(symbol, start_date=start_date, end_date=end_date,
                                 buy_points=buy_points, save_path=save_path)
    
    return buy_points


if __name__ == '__main__':
    # 示例：查找 AMD 买点并绘图
    buy_points = find_buy_points('AMD', min_factors=5, plot=True, 
                                 save_path='reports/amd_buy_points.png')
