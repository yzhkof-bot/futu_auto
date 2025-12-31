"""
可视化买点特征分析结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('reports/buy_points_analysis.csv')

print(f"总买点数: {len(df)}")
print(f"涉及股票数: {df['symbol'].nunique()}")

# 创建图表
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Nasdaq 100 Buy Points Feature Analysis\n(1M gain≥10%, 2M forward gain≥15%, drawdown≤10%)', fontsize=14)

# 1. 股票买点分布（前15名）
ax1 = axes[0, 0]
symbol_counts = df['symbol'].value_counts().head(15)
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(symbol_counts)))
bars = ax1.barh(symbol_counts.index[::-1], symbol_counts.values[::-1], color=colors[::-1])
ax1.set_xlabel('Number of Buy Points')
ax1.set_title('Top 15 Stocks by Buy Points')
for bar, val in zip(bars, symbol_counts.values[::-1]):
    ax1.text(val + 1, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=9)

# 2. RSI分布
ax2 = axes[0, 1]
ax2.hist(df['RSI_14'].dropna(), bins=30, color='steelblue', edgecolor='white', alpha=0.8)
ax2.axvline(df['RSI_14'].median(), color='red', linestyle='--', label=f'Median: {df["RSI_14"].median():.1f}')
ax2.axvline(70, color='orange', linestyle=':', label='Overbought (70)')
ax2.axvline(30, color='green', linestyle=':', label='Oversold (30)')
ax2.set_xlabel('RSI_14')
ax2.set_ylabel('Frequency')
ax2.set_title('RSI Distribution at Buy Points')
ax2.legend(fontsize=8)

# 3. Momentum分布
ax3 = axes[0, 2]
mom_data = df['Momentum_10'].dropna() * 100
ax3.hist(mom_data, bins=30, color='coral', edgecolor='white', alpha=0.8)
ax3.axvline(mom_data.median(), color='red', linestyle='--', label=f'Median: {mom_data.median():.1f}%')
ax3.set_xlabel('Momentum_10 (%)')
ax3.set_ylabel('Frequency')
ax3.set_title('10-Day Momentum Distribution')
ax3.legend(fontsize=8)

# 4. 布林带位置分布
ax4 = axes[1, 0]
bb_data = df['BB_Position'].dropna()
ax4.hist(bb_data, bins=30, color='mediumpurple', edgecolor='white', alpha=0.8)
ax4.axvline(bb_data.median(), color='red', linestyle='--', label=f'Median: {bb_data.median():.2f}')
ax4.axvline(0.5, color='gray', linestyle=':', label='Middle Band')
ax4.set_xlabel('Bollinger Band Position (0=Lower, 1=Upper)')
ax4.set_ylabel('Frequency')
ax4.set_title('Bollinger Band Position Distribution')
ax4.legend(fontsize=8)

# 5. ADX趋势强度分布
ax5 = axes[1, 1]
adx_data = df['ADX'].dropna()
ax5.hist(adx_data, bins=30, color='seagreen', edgecolor='white', alpha=0.8)
ax5.axvline(adx_data.median(), color='red', linestyle='--', label=f'Median: {adx_data.median():.1f}')
ax5.axvline(25, color='orange', linestyle=':', label='Strong Trend (25)')
ax5.set_xlabel('ADX')
ax5.set_ylabel('Frequency')
ax5.set_title('ADX (Trend Strength) Distribution')
ax5.legend(fontsize=8)

# 6. 波动率分布
ax6 = axes[1, 2]
vol_data = df['Volatility_20'].dropna() * 100
ax6.hist(vol_data, bins=30, color='indianred', edgecolor='white', alpha=0.8)
ax6.axvline(vol_data.median(), color='red', linestyle='--', label=f'Median: {vol_data.median():.1f}%')
ax6.set_xlabel('20-Day Volatility (Annualized %)')
ax6.set_ylabel('Frequency')
ax6.set_title('Volatility Distribution')
ax6.legend(fontsize=8)

plt.tight_layout()
plt.savefig('reports/buy_points_features.png', dpi=150, bbox_inches='tight')
print("\n图表已保存至 reports/buy_points_features.png")

# 创建高频特征汇总图
fig2, ax = plt.subplots(figsize=(12, 6))

# 高频布尔特征
bool_features = {
    'MACD Positive': 95.9,
    'Strong Trend (ADX>25)': 84.1,
    'MACD Bullish (MACD>Signal)': 82.4,
    'Strong Momentum (10D>5%)': 66.3,
    'Stochastic Overbought (>80)': 60.8,
    'MA Bullish Alignment': 60.3,
    'BB Near Upper': 58.7,
    'RSI Overbought (>70)': 52.7,
    'High Volatility': 50.6,
    'Stochastic Bullish (K>D)': 48.5,
}

features = list(bool_features.keys())
frequencies = list(bool_features.values())

colors = ['#2ecc71' if f >= 60 else '#3498db' if f >= 50 else '#95a5a6' for f in frequencies]
bars = ax.barh(features[::-1], frequencies[::-1], color=colors[::-1])

ax.axvline(50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
ax.set_xlabel('Frequency (%)')
ax.set_title('High-Frequency Features at Buy Points\n(Features present in >48% of all buy points)', fontsize=12)
ax.set_xlim(0, 100)

for bar, val in zip(bars, frequencies[::-1]):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('reports/buy_points_high_freq_features.png', dpi=150, bbox_inches='tight')
print("高频特征图表已保存至 reports/buy_points_high_freq_features.png")

# 打印详细统计
print("\n" + "="*70)
print("买点特征因子总结")
print("="*70)

print("\n【核心发现】符合条件的买点具有以下共性特征：\n")

print("1. 趋势确认特征（出现频率>80%）:")
print("   - MACD为正值 (95.9%)")
print("   - ADX>25 强趋势 (84.1%)")
print("   - MACD>Signal 多头排列 (82.4%)")

print("\n2. 动量特征（出现频率>50%）:")
print("   - 10日动量>5% (66.3%)")
print("   - Stochastic>80 超买区 (60.8%)")
print("   - RSI>70 超买区 (52.7%)")

print("\n3. 均线与布林带特征（出现频率>50%）:")
print("   - 均线多头排列 MA5>MA10>MA20>MA50 (60.3%)")
print("   - 价格接近布林带上轨 (58.7%)")

print("\n4. 波动率特征:")
print("   - 高波动率环境 (50.6%)")

print("\n【数值特征典型值】:")
print(f"   - RSI_14: 中位数 70.8 (偏向超买区)")
print(f"   - ADX: 中位数 41.2 (强趋势)")
print(f"   - Momentum_10: 中位数 7.5%")
print(f"   - BB_Position: 中位数 0.83 (接近上轨)")
print(f"   - 52周价格位置: 中位数 74% (偏向高位)")

print("\n【结论】")
print("这些买点的共同特征是：")
print("  → 处于强势上涨趋势中（MACD正值、ADX强趋势、均线多头）")
print("  → 动量强劲（短期涨幅已经很大）")
print("  → 技术指标处于超买区但趋势延续")
print("  → 这是典型的「追涨」信号，适合趋势跟踪策略")

plt.show()
