"""
布林带下轨分析 - 展示价格在布林带下轨以下的日期和程度
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys
sys.path.insert(0, '/Users/windye/PycharmProjects/FUTU_auto')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from src.data.data_fetcher import DataFetcher
from src.data.data_processor import DataProcessor


def analyze_below_lower_band(symbol: str = 'AAPL', 
                              start_date: str = '2020-01-01',
                              end_date: str = '2023-12-31',
                              bb_period: int = 20):
    """
    分析价格在布林带下轨以下的情况
    
    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        bb_period: 布林带周期
    """
    
    # 获取数据
    print(f"获取 {symbol} 数据...")
    fetcher = DataFetcher()
    processor = DataProcessor()
    
    data = fetcher.fetch_stock_data(symbol, start_date, end_date)
    data = processor.add_technical_indicators(data, ['bollinger'])
    
    # 计算价格相对于下轨的位置
    bb_lower_col = f'BB_Lower_{bb_period}'
    bb_upper_col = f'BB_Upper_{bb_period}'
    bb_middle_col = f'BB_Middle_{bb_period}'
    
    # 检查列是否存在
    if bb_lower_col not in data.columns:
        print(f"错误: 找不到 {bb_lower_col} 列")
        print(f"可用列: {[c for c in data.columns if 'BB' in c]}")
        return
    
    # 计算低于下轨的程度 (百分比)
    data['Below_Lower'] = data['Close'] < data[bb_lower_col]
    data['Below_Pct'] = np.where(
        data['Below_Lower'],
        (data[bb_lower_col] - data['Close']) / data[bb_lower_col] * 100,
        0
    )
    
    # 筛选出低于下轨的日期
    below_data = data[data['Below_Lower']].copy()
    
    print(f"\n{'='*60}")
    print(f"布林带下轨分析结果 - {symbol}")
    print(f"{'='*60}")
    print(f"分析期间: {start_date} 至 {end_date}")
    print(f"总交易日数: {len(data)}")
    print(f"低于下轨天数: {len(below_data)}")
    print(f"低于下轨比例: {len(below_data)/len(data)*100:.2f}%")
    
    if len(below_data) > 0:
        print(f"\n最大偏离程度: {below_data['Below_Pct'].max():.2f}%")
        print(f"平均偏离程度: {below_data['Below_Pct'].mean():.2f}%")
        
        # 打印详细信息
        print(f"\n{'='*60}")
        print("低于布林带下轨的日期详情:")
        print(f"{'='*60}")
        print(f"{'日期':<12} {'收盘价':>10} {'下轨':>10} {'偏离程度':>10}")
        print("-" * 45)
        
        for idx, row in below_data.iterrows():
            date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10]
            print(f"{date_str:<12} ${row['Close']:>9.2f} ${row[bb_lower_col]:>9.2f} {row['Below_Pct']:>9.2f}%")
    
    # 创建可视化
    create_visualization(data, below_data, symbol, bb_lower_col, bb_upper_col, bb_middle_col)
    
    return data, below_data


def create_visualization(data, below_data, symbol, bb_lower_col, bb_upper_col, bb_middle_col):
    """创建可视化图表"""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'{symbol} 布林带下轨分析', fontsize=14, fontweight='bold')
    
    # 图1: 价格与布林带
    ax1 = axes[0]
    ax1.plot(data.index, data['Close'], label='收盘价', color='blue', linewidth=1)
    ax1.plot(data.index, data[bb_upper_col], label='上轨', color='gray', linestyle='--', alpha=0.7)
    ax1.plot(data.index, data[bb_middle_col], label='中轨', color='orange', linestyle='-', alpha=0.7)
    ax1.plot(data.index, data[bb_lower_col], label='下轨', color='gray', linestyle='--', alpha=0.7)
    
    # 填充布林带区域
    ax1.fill_between(data.index, data[bb_lower_col], data[bb_upper_col], 
                     alpha=0.1, color='blue', label='布林带区域')
    
    # 标记低于下轨的点
    if len(below_data) > 0:
        ax1.scatter(below_data.index, below_data['Close'], 
                   color='red', s=30, zorder=5, label='低于下轨', marker='v')
    
    ax1.set_ylabel('价格 ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('价格走势与布林带')
    
    # 图2: 低于下轨的程度 (柱状图)
    ax2 = axes[1]
    colors = ['red' if x > 0 else 'lightgray' for x in data['Below_Pct']]
    ax2.bar(data.index, data['Below_Pct'], color=colors, width=1, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 添加平均线
    if len(below_data) > 0:
        avg_below = below_data['Below_Pct'].mean()
        ax2.axhline(y=avg_below, color='orange', linestyle='--', 
                   label=f'平均偏离: {avg_below:.2f}%', linewidth=1.5)
        ax2.legend(loc='upper right')
    
    ax2.set_ylabel('偏离程度 (%)')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('低于布林带下轨的程度 (越高表示偏离越大)')
    
    # 图3: 布林带位置指标
    ax3 = axes[2]
    bb_position_col = f'BB_Position_{bb_lower_col.split("_")[-1]}'
    
    if bb_position_col in data.columns:
        ax3.plot(data.index, data[bb_position_col], color='purple', linewidth=1)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='下轨 (0)')
        ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='中轨 (0.5)')
        ax3.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='上轨 (1)')
        ax3.fill_between(data.index, 0, data[bb_position_col], 
                        where=data[bb_position_col] < 0, 
                        color='red', alpha=0.3, label='低于下轨区域')
    else:
        # 手动计算位置
        bb_width = data[bb_upper_col] - data[bb_lower_col]
        bb_position = (data['Close'] - data[bb_lower_col]) / bb_width
        ax3.plot(data.index, bb_position, color='purple', linewidth=1)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='下轨 (0)')
        ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='中轨 (0.5)')
        ax3.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='上轨 (1)')
        ax3.fill_between(data.index, 0, bb_position, 
                        where=bb_position < 0, 
                        color='red', alpha=0.3, label='低于下轨区域')
    
    ax3.set_ylabel('布林带位置')
    ax3.set_xlabel('日期')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('布林带相对位置 (0=下轨, 0.5=中轨, 1=上轨)')
    
    # 格式化x轴日期
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = '/Users/windye/PycharmProjects/FUTU_auto/reports/bollinger_below_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存至: {output_path}")
    
    plt.show()


if __name__ == '__main__':
    # 运行分析
    data, below_data = analyze_below_lower_band(
        symbol='AAPL',
        start_date='2020-01-01',
        end_date='2023-12-31',
        bb_period=20
    )
