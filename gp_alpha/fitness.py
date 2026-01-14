"""
适应度函数 (Fitness Functions)

用于评估 Alpha 因子的有效性
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional


def calculate_ic(factor: np.ndarray, returns: np.ndarray, 
                 method: str = 'spearman') -> float:
    """
    计算信息系数 (Information Coefficient)
    
    IC = corr(factor_t, return_{t+1})
    
    Args:
        factor: 因子值序列
        returns: 未来收益序列（已对齐，即 returns[i] 是 factor[i] 之后的收益）
        method: 'spearman' (秩相关) 或 'pearson'
    
    Returns:
        IC 值（-1 到 1）
    """
    # 过滤无效值
    valid = ~(np.isnan(factor) | np.isnan(returns) | 
              np.isinf(factor) | np.isinf(returns))
    
    if np.sum(valid) < 30:  # 样本太少
        return 0.0
    
    f = factor[valid]
    r = returns[valid]
    
    try:
        if method == 'spearman':
            ic, _ = stats.spearmanr(f, r)
        else:
            ic, _ = stats.pearsonr(f, r)
        
        return ic if not np.isnan(ic) else 0.0
    except:
        return 0.0


def calculate_ic_series(factor: np.ndarray, returns: np.ndarray,
                        window: int = 20, method: str = 'spearman') -> np.ndarray:
    """
    计算滚动 IC 序列
    
    Args:
        factor: 因子值序列
        returns: 未来收益序列
        window: 滚动窗口
        method: 相关系数方法
    
    Returns:
        IC 时间序列
    """
    n = len(factor)
    ic_series = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        f_window = factor[i - window + 1:i + 1]
        r_window = returns[i - window + 1:i + 1]
        ic_series[i] = calculate_ic(f_window, r_window, method)
    
    return ic_series


def calculate_icir(factor: np.ndarray, returns: np.ndarray,
                   window: int = 20) -> float:
    """
    计算 IC 信息比率 (ICIR = mean(IC) / std(IC))
    
    ICIR 越高，因子越稳定有效
    
    Args:
        factor: 因子值序列
        returns: 未来收益序列
        window: 滚动窗口
    
    Returns:
        ICIR 值
    """
    ic_series = calculate_ic_series(factor, returns, window)
    valid_ic = ic_series[~np.isnan(ic_series)]
    
    if len(valid_ic) < 10:
        return 0.0
    
    ic_mean = np.mean(valid_ic)
    ic_std = np.std(valid_ic)
    
    if ic_std < 1e-10:
        return 0.0
    
    return ic_mean / ic_std


def calculate_sharpe(returns: np.ndarray, rf: float = 0.0, 
                     periods_per_year: int = 252) -> float:
    """
    计算夏普比率
    
    Args:
        returns: 收益序列
        rf: 无风险利率（年化）
        periods_per_year: 每年交易日数
    
    Returns:
        年化夏普比率
    """
    valid = ~(np.isnan(returns) | np.isinf(returns))
    r = returns[valid]
    
    if len(r) < 20:
        return 0.0
    
    excess_return = np.mean(r) - rf / periods_per_year
    volatility = np.std(r)
    
    if volatility < 1e-10:
        return 0.0
    
    return excess_return / volatility * np.sqrt(periods_per_year)


def calculate_turnover(factor: np.ndarray, top_pct: float = 0.2) -> float:
    """
    计算因子换手率
    
    换手率过高意味着交易成本高
    
    Args:
        factor: 因子值序列
        top_pct: 选股比例
    
    Returns:
        平均换手率
    """
    valid = ~np.isnan(factor)
    f = factor[valid]
    n = len(f)
    
    if n < 10:
        return 1.0
    
    # 简化：计算因子值变化的绝对值
    changes = np.abs(np.diff(f))
    avg_change = np.mean(changes)
    factor_range = np.max(f) - np.min(f)
    
    if factor_range < 1e-10:
        return 0.0
    
    return avg_change / factor_range


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    计算最大回撤
    
    Args:
        cumulative_returns: 累计收益序列
    
    Returns:
        最大回撤（正数）
    """
    valid = ~np.isnan(cumulative_returns)
    cum_ret = cumulative_returns[valid]
    
    if len(cum_ret) < 2:
        return 0.0
    
    peak = np.maximum.accumulate(cum_ret)
    drawdown = (peak - cum_ret) / (peak + 1e-10)
    
    return np.max(drawdown)


def evaluate_alpha(factor: np.ndarray, prices: np.ndarray,
                   forward_days: int = 1,
                   ic_weight: float = 0.4,
                   icir_weight: float = 0.3,
                   sharpe_weight: float = 0.2,
                   turnover_weight: float = 0.1) -> Tuple[float, dict]:
    """
    综合评估 Alpha 因子
    
    Args:
        factor: 因子值序列
        prices: 价格序列
        forward_days: 预测未来收益的天数
        ic_weight: IC 权重
        icir_weight: ICIR 权重
        sharpe_weight: 夏普比率权重
        turnover_weight: 换手率权重（负向）
    
    Returns:
        (综合得分, 详细指标字典)
    """
    # 计算未来收益
    returns = np.zeros_like(prices)
    returns[:-forward_days] = (prices[forward_days:] - prices[:-forward_days]) / prices[:-forward_days]
    returns[-forward_days:] = np.nan
    
    # 对齐：factor[i] 对应 returns[i]（即 factor 在 t 时刻，收益是 t 到 t+forward_days）
    
    # 计算各项指标
    ic = calculate_ic(factor, returns)
    icir = calculate_icir(factor, returns)
    
    # 基于因子的策略收益（简化：因子值为正时做多）
    factor_normalized = (factor - np.nanmean(factor)) / (np.nanstd(factor) + 1e-10)
    strategy_returns = np.sign(factor_normalized[:-1]) * returns[:-1]
    sharpe = calculate_sharpe(strategy_returns)
    
    # 换手率
    turnover = calculate_turnover(factor)
    
    # 综合得分
    # IC 和 ICIR 取绝对值（因为负 IC 也有预测能力，反向交易即可）
    score = (
        ic_weight * abs(ic) +
        icir_weight * abs(icir) +
        sharpe_weight * max(0, sharpe) / 3.0 +  # 归一化
        turnover_weight * (1 - turnover)  # 换手率越低越好
    )
    
    metrics = {
        'ic': ic,
        'icir': icir,
        'sharpe': sharpe,
        'turnover': turnover,
        'score': score
    }
    
    return score, metrics


def fitness_function(factor: np.ndarray, prices: np.ndarray,
                     forward_days: int = 1) -> float:
    """
    gplearn 适应度函数
    
    返回负值（因为 gplearn 默认最小化）
    
    Args:
        factor: 因子值
        prices: 价格
        forward_days: 预测天数
    
    Returns:
        负的综合得分
    """
    score, _ = evaluate_alpha(factor, prices, forward_days)
    return -score  # gplearn 最小化，所以取负


class AlphaFitness:
    """
    Alpha 因子适应度评估器
    
    用于 gplearn 的自定义适应度
    """
    
    def __init__(self, prices: np.ndarray, forward_days: int = 1):
        """
        Args:
            prices: 价格序列
            forward_days: 预测天数
        """
        self.prices = prices
        self.forward_days = forward_days
        
        # 预计算未来收益
        self.returns = np.zeros_like(prices)
        self.returns[:-forward_days] = (
            (prices[forward_days:] - prices[:-forward_days]) / prices[:-forward_days]
        )
        self.returns[-forward_days:] = np.nan
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 sample_weight: np.ndarray = None) -> float:
        """
        计算适应度（gplearn 接口）
        
        注意：gplearn 传入的 y_pred 就是我们的因子值
        """
        # 计算 IC
        ic = calculate_ic(y_pred, self.returns)
        
        # 返回负 IC（因为 gplearn 最小化）
        # 取绝对值，因为负 IC 也有价值
        return -abs(ic)
