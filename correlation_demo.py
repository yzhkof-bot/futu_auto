import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 定义你想研究的资产列表
# 例如：苹果, 微软, 谷歌, 亚马逊, 还有 黄金(GLD), 原油(USO)
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'GLD', 'USO']

print("正在下载数据...")
# 2. 下载历史数据 (只取收盘价)
data = yf.download(tickers, start="2023-01-01", end="2023-12-31")['Close']

# 3. 【关键步骤】计算每日收益率 (Percentage Change)
# 我们关心的是波动的同步性，而不是价格绝对值的同步性
returns = data.pct_change().dropna()

# 4. 计算相关性矩阵
# Pandas 自带的 corr() 方法，默认使用 Pearson 相关系数
correlation_matrix = returns.corr()

print("\n--- 相关性矩阵数据 ---")
print(correlation_matrix)

# 5. 可视化：画出热力图 (Heatmap)
# 这是看矩阵最直观的方法，颜色越深/越红代表相关性越高
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, 
            annot=True,         # 在格子里显示数值
            cmap='coolwarm',    # 颜色风格：冷暖色调 (红正蓝负)
            vmin=-1, vmax=1,    # 锁定范围在 -1 到 1
            square=True)        # 让格子是正方形

plt.title('Stock Correlation Matrix (Based on Daily Returns)')
plt.savefig('/Users/windye/PycharmProjects/FUTU_auto/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n热力图已保存到 correlation_heatmap.png")
