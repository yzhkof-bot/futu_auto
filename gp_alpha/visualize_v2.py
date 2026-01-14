"""
因子可视化工具 V2

工业级可视化:
1. IC 时序图
2. 分层累计收益图
3. 因子衰减图
4. 多因子对比图
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


def plot_ic_analysis(evaluator, save_path: str = None):
    """
    绘制 IC 分析图
    
    Args:
        evaluator: FactorEvaluator 实例
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('IC 分析', fontsize=14, fontweight='bold')
    
    # 计算 IC
    ic_series = evaluator.calc_ic_series()
    ic_stats = evaluator.get_ic_stats()
    
    # 图1: IC 时序
    ax1 = axes[0, 0]
    ax1.bar(ic_series.index, ic_series.values, 
            color=['green' if x > 0 else 'red' for x in ic_series.fillna(0)],
            alpha=0.7, width=1)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axhline(ic_stats['ic_mean'], color='blue', linestyle='--', 
                linewidth=1.5, label=f"均值={ic_stats['ic_mean']:.4f}")
    ax1.set_ylabel('IC')
    ax1.set_title('每日截面 IC')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 图2: IC 滚动均值
    ax2 = axes[0, 1]
    rolling_ic = ic_series.rolling(20, min_periods=5).mean()
    ax2.plot(rolling_ic.index, rolling_ic.values, color='steelblue', linewidth=1.5)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.fill_between(rolling_ic.index, 0, rolling_ic.values,
                     where=rolling_ic > 0, alpha=0.3, color='green')
    ax2.fill_between(rolling_ic.index, 0, rolling_ic.values,
                     where=rolling_ic < 0, alpha=0.3, color='red')
    ax2.set_ylabel('IC (20日滚动均值)')
    ax2.set_title('IC 滚动均值')
    ax2.grid(True, alpha=0.3)
    
    # 图3: IC 分布
    ax3 = axes[1, 0]
    ic_valid = ic_series.dropna()
    ax3.hist(ic_valid, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax3.axvline(ic_stats['ic_mean'], color='red', linestyle='--', 
                linewidth=2, label=f"均值={ic_stats['ic_mean']:.4f}")
    ax3.axvline(0, color='black', linewidth=1)
    ax3.set_xlabel('IC')
    ax3.set_ylabel('频数')
    ax3.set_title(f'IC 分布 (ICIR={ic_stats["ic_ir"]:.2f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 图4: IC 累计
    ax4 = axes[1, 1]
    cumsum_ic = ic_series.fillna(0).cumsum()
    ax4.plot(cumsum_ic.index, cumsum_ic.values, color='purple', linewidth=1.5)
    ax4.fill_between(cumsum_ic.index, 0, cumsum_ic.values, alpha=0.3, color='purple')
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.set_ylabel('累计 IC')
    ax4.set_xlabel('日期')
    ax4.set_title('IC 累计曲线')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至 {save_path}")
    
    plt.show()
    return fig


def plot_group_returns(evaluator, n_groups: int = 5, save_path: str = None):
    """
    绘制分层回测图
    
    Args:
        evaluator: FactorEvaluator 实例
        n_groups: 分组数
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'分层回测 ({n_groups} 组)', fontsize=14, fontweight='bold')
    
    # 计算分层收益
    cumulative = evaluator.calc_cumulative_returns(n_groups)
    group_stats = evaluator.get_group_stats(n_groups)
    
    # 颜色映射
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_groups))
    
    # 图1: 各组累计收益
    ax1 = axes[0, 0]
    for i in range(n_groups):
        col = f'G{i+1}'
        ax1.plot(cumulative.index, cumulative[col].values, 
                 color=colors[i], linewidth=1.5, label=col)
    ax1.axhline(1, color='black', linewidth=0.5, linestyle='--')
    ax1.set_ylabel('累计收益')
    ax1.set_title('各组累计收益')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 图2: 多空累计收益
    ax2 = axes[0, 1]
    ax2.plot(cumulative.index, cumulative['Long-Short'].values,
             color='steelblue', linewidth=2, label='多空组合')
    ax2.axhline(1, color='black', linewidth=0.5, linestyle='--')
    ax2.fill_between(cumulative.index, 1, cumulative['Long-Short'].values,
                     where=cumulative['Long-Short'] > 1, alpha=0.3, color='green')
    ax2.fill_between(cumulative.index, 1, cumulative['Long-Short'].values,
                     where=cumulative['Long-Short'] < 1, alpha=0.3, color='red')
    ax2.set_ylabel('累计收益')
    ax2.set_title(f'多空组合 (年化收益={group_stats["long_short_return"]:.1%}, '
                  f'Sharpe={group_stats["long_short_sharpe"]:.2f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: 各组平均收益柱状图
    ax3 = axes[1, 0]
    group_returns = evaluator._group_returns.mean() * 252  # 年化
    bars = ax3.bar(range(n_groups), group_returns.values[:n_groups], 
                   color=colors, edgecolor='black', linewidth=0.5)
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.set_xticks(range(n_groups))
    ax3.set_xticklabels([f'G{i+1}' for i in range(n_groups)])
    ax3.set_ylabel('年化收益率')
    ax3.set_title(f'各组年化收益 (单调性={group_stats["monotonicity"]:.2f})')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars, group_returns.values[:n_groups]):
        height = bar.get_height()
        ax3.annotate(f'{val:.1%}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    
    # 图4: 多空收益分布
    ax4 = axes[1, 1]
    gr = evaluator._group_returns
    long_short = gr[f'G{n_groups}'] - gr['G1']
    ax4.hist(long_short.dropna() * 100, bins=50, color='steelblue', 
             alpha=0.7, edgecolor='white')
    ax4.axvline(long_short.mean() * 100, color='red', linestyle='--',
                linewidth=2, label=f'均值={long_short.mean()*100:.2f}%')
    ax4.axvline(0, color='black', linewidth=1)
    ax4.set_xlabel('日收益率 (%)')
    ax4.set_ylabel('频数')
    ax4.set_title('多空日收益分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至 {save_path}")
    
    plt.show()
    return fig


def plot_ic_decay(evaluator, max_lag: int = 20, save_path: str = None):
    """
    绘制 IC 衰减图
    
    Args:
        evaluator: FactorEvaluator 实例
        max_lag: 最大滞后期
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ic_decay = evaluator.calc_ic_decay(max_lag)
    decay_stats = evaluator.get_decay_stats(max_lag)
    
    # 柱状图
    colors = ['green' if x > 0 else 'red' for x in ic_decay.fillna(0)]
    ax.bar(ic_decay.index, ic_decay.values, color=colors, alpha=0.7, edgecolor='black')
    
    # 趋势线
    valid = ~ic_decay.isna()
    if valid.sum() > 2:
        z = np.polyfit(ic_decay.index[valid], ic_decay.values[valid], 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(1, max_lag, 100)
        ax.plot(x_smooth, p(x_smooth), 'b--', linewidth=2, alpha=0.7, label='趋势线')
    
    ax.axhline(0, color='black', linewidth=0.5)
    
    # 标注半衰期
    if not np.isnan(decay_stats['half_life']):
        ax.axvline(decay_stats['half_life'], color='orange', linestyle='--',
                   linewidth=2, label=f'半衰期={decay_stats["half_life"]}天')
    
    ax.set_xlabel('滞后天数')
    ax.set_ylabel('IC')
    ax.set_title('因子 IC 衰减分析')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
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
        factors: 因子列表（来自 GPAlphaMinerV2）
        save_path: 保存路径
    """
    if not factors:
        print("无因子数据")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('因子对比分析', fontsize=14, fontweight='bold')
    
    n = len(factors)
    names = [f"#{f['rank']}" for f in factors]
    
    # 提取指标
    train_ic = [f['train_metrics'].get('ic_mean', 0) for f in factors]
    test_ic = [f['test_metrics'].get('ic_mean', 0) for f in factors]
    train_icir = [f['train_metrics'].get('ic_ir', 0) for f in factors]
    test_icir = [f['test_metrics'].get('ic_ir', 0) for f in factors]
    train_sharpe = [f['train_metrics'].get('long_short_sharpe', 0) for f in factors]
    test_sharpe = [f['test_metrics'].get('long_short_sharpe', 0) for f in factors]
    train_score = [f['train_metrics'].get('composite_score', 0) for f in factors]
    test_score = [f['test_metrics'].get('composite_score', 0) for f in factors]
    
    x = np.arange(n)
    width = 0.35
    
    # 图1: IC 对比
    ax1 = axes[0, 0]
    ax1.bar(x - width/2, train_ic, width, label='训练集', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, test_ic, width, label='测试集', color='coral', alpha=0.8)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel('IC')
    ax1.set_title('IC 均值')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 图2: ICIR 对比
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, train_icir, width, label='训练集', color='steelblue', alpha=0.8)
    ax2.bar(x + width/2, test_icir, width, label='测试集', color='coral', alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axhline(0.5, color='green', linewidth=0.5, linestyle='--', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel('ICIR')
    ax2.set_title('IC 信息比率')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 图3: Sharpe 对比
    ax3 = axes[1, 0]
    ax3.bar(x - width/2, train_sharpe, width, label='训练集', color='steelblue', alpha=0.8)
    ax3.bar(x + width/2, test_sharpe, width, label='测试集', color='coral', alpha=0.8)
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.axhline(1, color='green', linewidth=0.5, linestyle='--', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names)
    ax3.set_ylabel('Sharpe')
    ax3.set_title('多空夏普比率')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 图4: 综合得分对比
    ax4 = axes[1, 1]
    ax4.bar(x - width/2, train_score, width, label='训练集', color='steelblue', alpha=0.8)
    ax4.bar(x + width/2, test_score, width, label='测试集', color='coral', alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(names)
    ax4.set_ylabel('Score')
    ax4.set_title('综合得分')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至 {save_path}")
    
    plt.show()
    return fig


def plot_full_report(evaluator, factor_name: str = "Alpha", 
                     n_groups: int = 5, save_path: str = None):
    """
    绘制完整因子报告
    
    Args:
        evaluator: FactorEvaluator 实例
        factor_name: 因子名称
        n_groups: 分组数
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'因子报告: {factor_name}', fontsize=16, fontweight='bold')
    
    # 创建网格
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 计算数据
    ic_series = evaluator.calc_ic_series()
    ic_stats = evaluator.get_ic_stats()
    cumulative = evaluator.calc_cumulative_returns(n_groups)
    group_stats = evaluator.get_group_stats(n_groups)
    ic_decay = evaluator.calc_ic_decay(20)
    turnover = evaluator.calc_turnover()
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_groups))
    
    # 图1: IC 时序 (占据左上两格)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.bar(ic_series.index, ic_series.values,
            color=['green' if x > 0 else 'red' for x in ic_series.fillna(0)],
            alpha=0.7, width=1)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axhline(ic_stats['ic_mean'], color='blue', linestyle='--', linewidth=1.5)
    ax1.set_ylabel('IC')
    ax1.set_title(f'每日 IC (均值={ic_stats["ic_mean"]:.4f}, ICIR={ic_stats["ic_ir"]:.2f})')
    ax1.grid(True, alpha=0.3)
    
    # 图2: IC 分布
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(ic_series.dropna(), bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    ax2.axvline(ic_stats['ic_mean'], color='red', linestyle='--', linewidth=2)
    ax2.axvline(0, color='black', linewidth=1)
    ax2.set_xlabel('IC')
    ax2.set_title('IC 分布')
    ax2.grid(True, alpha=0.3)
    
    # 图3: 分层累计收益
    ax3 = fig.add_subplot(gs[1, :2])
    for i in range(n_groups):
        col = f'G{i+1}'
        ax3.plot(cumulative.index, cumulative[col].values, 
                 color=colors[i], linewidth=1.5, label=col)
    ax3.plot(cumulative.index, cumulative['Long-Short'].values,
             color='black', linewidth=2, linestyle='--', label='多空')
    ax3.axhline(1, color='gray', linewidth=0.5, linestyle=':')
    ax3.set_ylabel('累计收益')
    ax3.set_title(f'分层累计收益 (多空Sharpe={group_stats["long_short_sharpe"]:.2f})')
    ax3.legend(loc='upper left', ncol=n_groups+1, fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 图4: 各组年化收益
    ax4 = fig.add_subplot(gs[1, 2])
    group_returns = evaluator._group_returns.mean() * 252
    ax4.bar(range(n_groups), group_returns.values[:n_groups], 
            color=colors, edgecolor='black', linewidth=0.5)
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.set_xticks(range(n_groups))
    ax4.set_xticklabels([f'G{i+1}' for i in range(n_groups)])
    ax4.set_ylabel('年化收益')
    ax4.set_title(f'各组收益 (单调性={group_stats["monotonicity"]:.2f})')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 图5: IC 衰减
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.bar(ic_decay.index, ic_decay.values,
            color=['green' if x > 0 else 'red' for x in ic_decay.fillna(0)],
            alpha=0.7, edgecolor='black')
    ax5.axhline(0, color='black', linewidth=0.5)
    ax5.set_xlabel('滞后天数')
    ax5.set_ylabel('IC')
    ax5.set_title('IC 衰减')
    ax5.grid(True, alpha=0.3)
    
    # 图6: 换手率
    ax6 = fig.add_subplot(gs[2, 1])
    turnover_valid = turnover.dropna()
    ax6.plot(turnover_valid.index, turnover_valid.values, color='purple', linewidth=1)
    ax6.axhline(turnover_valid.mean(), color='red', linestyle='--', linewidth=1.5)
    ax6.set_ylabel('换手率')
    ax6.set_title(f'日换手率 (均值={turnover_valid.mean():.1%})')
    ax6.grid(True, alpha=0.3)
    
    # 图7: 统计摘要
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    metrics = evaluator.evaluate(verbose=False)
    summary_text = f"""
    【IC 分析】
    IC 均值:     {metrics['ic_mean']:.4f}
    IC 标准差:   {metrics['ic_std']:.4f}
    ICIR:        {metrics['ic_ir']:.4f}
    IC > 0 比例: {metrics['ic_positive_ratio']:.1%}
    
    【分层回测】
    多空收益(年化): {metrics['long_short_return']:.1%}
    多空夏普:       {metrics['long_short_sharpe']:.2f}
    单调性:         {metrics['monotonicity']:.2f}
    
    【其他】
    日均换手率: {metrics['turnover_mean']:.1%}
    半衰期:     {metrics['half_life']} 天
    
    【综合得分】
    {metrics['composite_score']:.4f}
    """
    
    ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至 {save_path}")
    
    plt.show()
    return fig


if __name__ == '__main__':
    # 测试
    from data_manager import PanelDataManager
    from factor_engine import alpha_momentum, preprocess_factor
    from evaluator import FactorEvaluator
    
    # 获取数据
    dm = PanelDataManager()
    dm.fetch(pool_type='nasdaq100', use_cache=True)
    
    # 计算因子
    features = dm.get_feature_panels()
    factor = alpha_momentum(features, d=20)
    factor = preprocess_factor(factor)
    
    # 未来收益
    forward_return = dm.get_forward_return(days=5)
    
    # 评估
    evaluator = FactorEvaluator(factor, forward_return, forward_days=5)
    
    # 绘制完整报告
    plot_full_report(evaluator, factor_name="20日动量", save_path=None)
