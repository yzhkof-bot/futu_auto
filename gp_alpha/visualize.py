"""
因子可视化工具

绘制因子表现、IC 序列、回测曲线等
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_factor_performance(factor_values: np.ndarray,
                            prices: np.ndarray,
                            dates: List,
                            formula: str,
                            forward_days: int = 1,
                            save_path: str = None):
    """
    绘制因子表现图
    
    Args:
        factor_values: 因子值序列
        prices: 价格序列
        dates: 日期列表
        formula: 因子公式
        forward_days: 预测天数
        save_path: 保存路径
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'因子分析: {formula[:60]}...', fontsize=12, fontweight='bold')
    
    # 过滤无效值
    valid = ~(np.isnan(factor_values) | np.isinf(factor_values))
    
    # 图1: 价格走势
    ax1 = axes[0]
    ax1.plot(dates, prices, color='steelblue', linewidth=1)
    ax1.set_ylabel('价格')
    ax1.set_title('价格走势')
    ax1.grid(True, alpha=0.3)
    
    # 图2: 因子值
    ax2 = axes[1]
    factor_plot = np.where(valid, factor_values, np.nan)
    ax2.plot(dates, factor_plot, color='purple', linewidth=0.8, alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax2.set_ylabel('因子值')
    ax2.set_title('因子值序列')
    ax2.grid(True, alpha=0.3)
    
    # 图3: 因子信号与收益
    ax3 = axes[2]
    
    # 计算未来收益
    returns = np.zeros_like(prices)
    returns[:-forward_days] = (prices[forward_days:] - prices[:-forward_days]) / prices[:-forward_days]
    returns[-forward_days:] = np.nan
    
    # 因子方向收益
    factor_normalized = np.where(
        valid,
        (factor_values - np.nanmean(factor_values)) / (np.nanstd(factor_values) + 1e-10),
        0
    )
    strategy_returns = np.sign(factor_normalized) * returns
    strategy_returns = np.where(np.isnan(strategy_returns), 0, strategy_returns)
    
    # 累计收益
    cumulative_strategy = np.cumprod(1 + strategy_returns)
    cumulative_bh = np.cumprod(1 + np.where(np.isnan(returns), 0, returns))
    
    ax3.plot(dates, cumulative_strategy, label='因子策略', color='green', linewidth=1.5)
    ax3.plot(dates, cumulative_bh, label='买入持有', color='gray', linewidth=1, alpha=0.7)
    ax3.axhline(1, color='black', linewidth=0.5, linestyle='--')
    ax3.set_ylabel('累计收益')
    ax3.set_title('策略 vs 买入持有')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 图4: 滚动 IC
    ax4 = axes[3]
    window = 20
    ic_series = np.full(len(factor_values), np.nan)
    
    from scipy import stats
    for i in range(window - 1, len(factor_values)):
        f_window = factor_values[i - window + 1:i + 1]
        r_window = returns[i - window + 1:i + 1]
        valid_window = ~(np.isnan(f_window) | np.isnan(r_window))
        if np.sum(valid_window) > 5:
            try:
                ic, _ = stats.spearmanr(f_window[valid_window], r_window[valid_window])
                ic_series[i] = ic
            except:
                pass
    
    ax4.plot(dates, ic_series, color='orange', linewidth=0.8)
    ax4.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax4.axhline(0.05, color='green', linewidth=0.5, linestyle=':', alpha=0.7)
    ax4.axhline(-0.05, color='red', linewidth=0.5, linestyle=':', alpha=0.7)
    ax4.fill_between(dates, 0, ic_series, where=ic_series > 0, alpha=0.3, color='green')
    ax4.fill_between(dates, 0, ic_series, where=ic_series < 0, alpha=0.3, color='red')
    ax4.set_ylabel('IC')
    ax4.set_xlabel('日期')
    ax4.set_title(f'滚动 IC ({window}日)')
    ax4.set_ylim(-0.5, 0.5)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至 {save_path}")
    
    plt.show()
    return fig


def plot_factor_comparison(factors: List[Dict], save_path: str = None):
    """
    绘制多因子对比图
    
    Args:
        factors: 因子列表
        save_path: 保存路径
    """
    if not factors:
        print("无因子数据")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('因子对比分析', fontsize=14, fontweight='bold')
    
    # 提取数据
    names = [f"#{f['rank']}" for f in factors]
    ics = [f['ic'] for f in factors]
    icirs = [f['icir'] for f in factors]
    sharpes = [f['sharpe'] for f in factors]
    scores = [f['score'] for f in factors]
    lengths = [f['length'] for f in factors]
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(factors)))
    
    # 图1: IC 对比
    ax1 = axes[0, 0]
    bars = ax1.bar(names, ics, color=colors)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axhline(0.03, color='green', linewidth=0.5, linestyle='--', alpha=0.7)
    ax1.axhline(-0.03, color='red', linewidth=0.5, linestyle='--', alpha=0.7)
    ax1.set_ylabel('IC')
    ax1.set_title('信息系数 (IC)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 图2: ICIR 对比
    ax2 = axes[0, 1]
    ax2.bar(names, icirs, color=colors)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axhline(0.5, color='green', linewidth=0.5, linestyle='--', alpha=0.7)
    ax2.set_ylabel('ICIR')
    ax2.set_title('IC 信息比率 (ICIR)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 图3: Sharpe 对比
    ax3 = axes[1, 0]
    ax3.bar(names, sharpes, color=colors)
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.axhline(1, color='green', linewidth=0.5, linestyle='--', alpha=0.7)
    ax3.set_ylabel('Sharpe')
    ax3.set_title('夏普比率')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 图4: 综合得分 vs 复杂度
    ax4 = axes[1, 1]
    scatter = ax4.scatter(lengths, scores, c=range(len(factors)), 
                          cmap='viridis', s=100, alpha=0.8)
    for i, name in enumerate(names):
        ax4.annotate(name, (lengths[i], scores[i]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax4.set_xlabel('公式长度')
    ax4.set_ylabel('综合得分')
    ax4.set_title('得分 vs 复杂度')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至 {save_path}")
    
    plt.show()
    return fig


def plot_evolution_history(history: List[Dict], save_path: str = None):
    """
    绘制进化历史
    
    Args:
        history: 进化历史记录
        save_path: 保存路径
    """
    if not history:
        print("无进化历史")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('进化历史', fontsize=14, fontweight='bold')
    
    generations = [h['generation'] for h in history]
    best_fitness = [h['best_fitness'] for h in history]
    avg_fitness = [h.get('avg_fitness', h['best_fitness']) for h in history]
    
    # 图1: 适应度曲线
    ax1 = axes[0]
    ax1.plot(generations, best_fitness, 'g-', linewidth=2, label='最佳适应度')
    ax1.plot(generations, avg_fitness, 'b--', linewidth=1, alpha=0.7, label='平均适应度')
    ax1.set_ylabel('适应度 (负 IC)')
    ax1.set_title('适应度进化')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 最佳公式长度
    if 'best_length' in history[0]:
        ax2 = axes[1]
        best_lengths = [h['best_length'] for h in history]
        ax2.plot(generations, best_lengths, 'purple', linewidth=1.5)
        ax2.set_ylabel('公式长度')
        ax2.set_xlabel('代数')
        ax2.set_title('公式复杂度进化')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至 {save_path}")
    
    plt.show()
    return fig


if __name__ == '__main__':
    # 测试用例
    import numpy as np
    
    # 模拟数据
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    prices = 100 * np.cumprod(1 + np.random.randn(n) * 0.02)
    factor_values = np.random.randn(n)
    
    plot_factor_performance(
        factor_values, prices, dates,
        formula="div(sub(close, mean10(close)), std10(close))"
    )
