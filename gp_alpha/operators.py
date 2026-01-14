"""
自定义遗传规划算子

定义用于构建 Alpha 因子的基础运算符
"""

import numpy as np
from typing import Callable


def protected_div(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """保护除法，避免除零"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(x2) > 1e-10, x1 / x2, 0.0)
        result = np.clip(result, -1e10, 1e10)
    return result


def protected_log(x: np.ndarray) -> np.ndarray:
    """保护对数，避免负数和零"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(x > 1e-10, np.log(x), 0.0)
        result = np.clip(result, -1e10, 1e10)
    return result


def protected_sqrt(x: np.ndarray) -> np.ndarray:
    """保护平方根，避免负数"""
    return np.sqrt(np.abs(x))


def ts_delay(x: np.ndarray, d: int = 1) -> np.ndarray:
    """时序延迟 (lag)"""
    result = np.roll(x, d)
    result[:d] = np.nan
    return result


def ts_delta(x: np.ndarray, d: int = 1) -> np.ndarray:
    """时序差分"""
    return x - ts_delay(x, d)


def ts_sum(x: np.ndarray, d: int = 5) -> np.ndarray:
    """滚动求和"""
    result = np.full_like(x, np.nan, dtype=float)
    for i in range(d - 1, len(x)):
        result[i] = np.nansum(x[i - d + 1:i + 1])
    return result


def ts_mean(x: np.ndarray, d: int = 5) -> np.ndarray:
    """滚动均值"""
    result = np.full_like(x, np.nan, dtype=float)
    for i in range(d - 1, len(x)):
        result[i] = np.nanmean(x[i - d + 1:i + 1])
    return result


def ts_std(x: np.ndarray, d: int = 5) -> np.ndarray:
    """滚动标准差"""
    result = np.full_like(x, np.nan, dtype=float)
    for i in range(d - 1, len(x)):
        result[i] = np.nanstd(x[i - d + 1:i + 1])
    return result


def ts_max(x: np.ndarray, d: int = 5) -> np.ndarray:
    """滚动最大值"""
    result = np.full_like(x, np.nan, dtype=float)
    for i in range(d - 1, len(x)):
        result[i] = np.nanmax(x[i - d + 1:i + 1])
    return result


def ts_min(x: np.ndarray, d: int = 5) -> np.ndarray:
    """滚动最小值"""
    result = np.full_like(x, np.nan, dtype=float)
    for i in range(d - 1, len(x)):
        result[i] = np.nanmin(x[i - d + 1:i + 1])
    return result


def ts_rank(x: np.ndarray, d: int = 5) -> np.ndarray:
    """滚动排名（百分位）"""
    result = np.full_like(x, np.nan, dtype=float)
    for i in range(d - 1, len(x)):
        window = x[i - d + 1:i + 1]
        valid = ~np.isnan(window)
        if np.sum(valid) > 0:
            rank = np.sum(window[valid] <= x[i]) / np.sum(valid)
            result[i] = rank
    return result


def ts_corr(x: np.ndarray, y: np.ndarray, d: int = 10) -> np.ndarray:
    """滚动相关系数"""
    result = np.full_like(x, np.nan, dtype=float)
    for i in range(d - 1, len(x)):
        x_window = x[i - d + 1:i + 1]
        y_window = y[i - d + 1:i + 1]
        valid = ~(np.isnan(x_window) | np.isnan(y_window))
        if np.sum(valid) > 2:
            corr = np.corrcoef(x_window[valid], y_window[valid])[0, 1]
            result[i] = corr if not np.isnan(corr) else 0.0
    return result


def sign(x: np.ndarray) -> np.ndarray:
    """符号函数"""
    return np.sign(x)


def abs_op(x: np.ndarray) -> np.ndarray:
    """绝对值"""
    return np.abs(x)


def neg(x: np.ndarray) -> np.ndarray:
    """取负"""
    return -x


def rank_cross_sectional(x: np.ndarray) -> np.ndarray:
    """
    截面排名（归一化到0-1）
    注意：这里简化处理，实际应该在多股票截面上计算
    """
    valid = ~np.isnan(x)
    result = np.full_like(x, np.nan, dtype=float)
    if np.sum(valid) > 0:
        ranks = np.argsort(np.argsort(x[valid]))
        result[valid] = ranks / (np.sum(valid) - 1) if np.sum(valid) > 1 else 0.5
    return result


# 用于 gplearn 的函数包装器
def make_ts_function(func: Callable, d: int, name: str):
    """创建带固定窗口的时序函数"""
    def wrapped(x):
        return func(x, d)
    wrapped.__name__ = f"{name}_{d}"
    return wrapped


# 预定义的时序函数（不同窗口）
TS_FUNCTIONS = {
    'ts_delay_1': lambda x: ts_delay(x, 1),
    'ts_delay_5': lambda x: ts_delay(x, 5),
    'ts_delta_1': lambda x: ts_delta(x, 1),
    'ts_delta_5': lambda x: ts_delta(x, 5),
    'ts_mean_5': lambda x: ts_mean(x, 5),
    'ts_mean_10': lambda x: ts_mean(x, 10),
    'ts_mean_20': lambda x: ts_mean(x, 20),
    'ts_std_5': lambda x: ts_std(x, 5),
    'ts_std_10': lambda x: ts_std(x, 10),
    'ts_max_5': lambda x: ts_max(x, 5),
    'ts_max_10': lambda x: ts_max(x, 10),
    'ts_min_5': lambda x: ts_min(x, 5),
    'ts_min_10': lambda x: ts_min(x, 10),
    'ts_rank_5': lambda x: ts_rank(x, 5),
    'ts_rank_10': lambda x: ts_rank(x, 10),
    'ts_sum_5': lambda x: ts_sum(x, 5),
    'ts_sum_10': lambda x: ts_sum(x, 10),
}
