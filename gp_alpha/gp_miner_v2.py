"""
遗传规划 Alpha 因子挖掘器 V2

工业级实现:
1. 使用 Panel 数据结构（日期 × 股票）
2. 截面 IC 作为适应度函数
3. 训练集/测试集切分
4. 完整的因子评估
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Callable
import warnings
import pickle
import os
import random

warnings.filterwarnings('ignore')

# gplearn 导入
try:
    from gplearn.genetic import SymbolicTransformer
    from gplearn.functions import make_function
    from gplearn.fitness import make_fitness
    GPLEARN_AVAILABLE = True
except ImportError:
    GPLEARN_AVAILABLE = False
    print("警告: gplearn 未安装，请运行: pip install gplearn")

from .data_manager import PanelDataManager
from .factor_engine import (
    ts_delay, ts_delta, ts_mean, ts_std, ts_max, ts_min, ts_rank, ts_sum,
    ts_zscore, cs_rank, cs_zscore, preprocess_factor
)
from .evaluator import FactorEvaluator, quick_evaluate


# ============================================================
# 工业级 gplearn 算子库
# ============================================================

# ----------------- 基础运算 -----------------

def _protected_div(x1, x2):
    """保护除法"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(x2) > 1e-10, x1 / x2, 0.0)
        return np.clip(result, -1e6, 1e6)


def _protected_log(x):
    """保护对数"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x > 1e-10, np.log(x), 0.0)


def _protected_sqrt(x):
    """保护平方根"""
    return np.sqrt(np.abs(x))


def _sign(x):
    return np.sign(x)


def _abs(x):
    return np.abs(x)


def _neg(x):
    return -x


def _square(x):
    """平方"""
    return np.clip(x ** 2, -1e10, 1e10)


def _cube(x):
    """立方"""
    return np.clip(x ** 3, -1e10, 1e10)


def _inv(x):
    """倒数"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x) > 1e-10, 1.0 / x, 0.0)


def _sigmoid(x):
    """Sigmoid 函数"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _tanh(x):
    """双曲正切"""
    return np.tanh(np.clip(x, -500, 500))


# ----------------- 比较运算 -----------------

def _max2(x1, x2):
    """两数取大"""
    return np.maximum(x1, x2)


def _min2(x1, x2):
    """两数取小"""
    return np.minimum(x1, x2)


def _gt(x1, x2):
    """大于 (x1 > x2 ? 1 : 0)"""
    return np.where(x1 > x2, 1.0, 0.0)


def _lt(x1, x2):
    """小于 (x1 < x2 ? 1 : 0)"""
    return np.where(x1 < x2, 1.0, 0.0)


# ----------------- 滚动窗口基础函数 -----------------

def _rolling_mean(x, window):
    """滚动均值"""
    result = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        result[i] = np.nanmean(x[start:i+1])
    return result


def _rolling_std(x, window):
    """滚动标准差"""
    result = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        if i - start >= 1:
            result[i] = np.nanstd(x[start:i+1])
        else:
            result[i] = 0
    return result


def _rolling_max(x, window):
    """滚动最大值"""
    result = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        result[i] = np.nanmax(x[start:i+1])
    return result


def _rolling_min(x, window):
    """滚动最小值"""
    result = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        result[i] = np.nanmin(x[start:i+1])
    return result


def _rolling_sum(x, window):
    """滚动求和"""
    result = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        result[i] = np.nansum(x[start:i+1])
    return result


def _rolling_prod(x, window):
    """滚动乘积"""
    result = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        result[i] = np.nanprod(x[start:i+1])
    return np.clip(result, -1e10, 1e10)


def _rolling_rank(x, window):
    """滚动排名（百分位）"""
    result = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        window_data = x[start:i+1]
        valid = ~np.isnan(window_data)
        if np.sum(valid) > 0:
            # 当前值在窗口内的排名百分位
            result[i] = np.sum(window_data[valid] <= x[i]) / np.sum(valid)
    return result


def _rolling_skew(x, window):
    """滚动偏度"""
    from scipy import stats
    result = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        if i - start >= 2:  # 至少3个点
            window_data = x[start:i+1]
            valid = ~np.isnan(window_data)
            if np.sum(valid) >= 3:
                result[i] = stats.skew(window_data[valid])
    return np.nan_to_num(result, nan=0.0)


def _rolling_kurt(x, window):
    """滚动峰度"""
    from scipy import stats
    result = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        if i - start >= 3:  # 至少4个点
            window_data = x[start:i+1]
            valid = ~np.isnan(window_data)
            if np.sum(valid) >= 4:
                result[i] = stats.kurtosis(window_data[valid])
    return np.nan_to_num(result, nan=0.0)


def _rolling_argmax(x, window):
    """滚动最大值位置（距今天数）"""
    result = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        window_data = x[start:i+1]
        valid = ~np.isnan(window_data)
        if np.sum(valid) > 0:
            argmax = np.nanargmax(window_data)
            result[i] = len(window_data) - 1 - argmax  # 距今天数
    return result


def _rolling_argmin(x, window):
    """滚动最小值位置（距今天数）"""
    result = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        window_data = x[start:i+1]
        valid = ~np.isnan(window_data)
        if np.sum(valid) > 0:
            argmin = np.nanargmin(window_data)
            result[i] = len(window_data) - 1 - argmin  # 距今天数
    return result


def _rolling_corr(x1, x2, window):
    """滚动相关系数"""
    result = np.full_like(x1, np.nan, dtype=float)
    for i in range(len(x1)):
        start = max(0, i - window + 1)
        if i - start >= 2:
            w1 = x1[start:i+1]
            w2 = x2[start:i+1]
            valid = ~(np.isnan(w1) | np.isnan(w2))
            if np.sum(valid) >= 3:
                corr = np.corrcoef(w1[valid], w2[valid])[0, 1]
                result[i] = corr if not np.isnan(corr) else 0.0
    return np.nan_to_num(result, nan=0.0)


def _rolling_cov(x1, x2, window):
    """滚动协方差"""
    result = np.full_like(x1, np.nan, dtype=float)
    for i in range(len(x1)):
        start = max(0, i - window + 1)
        if i - start >= 1:
            w1 = x1[start:i+1]
            w2 = x2[start:i+1]
            valid = ~(np.isnan(w1) | np.isnan(w2))
            if np.sum(valid) >= 2:
                result[i] = np.cov(w1[valid], w2[valid])[0, 1]
    return np.nan_to_num(result, nan=0.0)


def _decay_linear(x, window):
    """线性衰减加权均值 (近期权重大)"""
    result = np.full_like(x, np.nan, dtype=float)
    weights = np.arange(1, window + 1, dtype=float)
    weights = weights / weights.sum()
    for i in range(len(x)):
        start = max(0, i - window + 1)
        window_data = x[start:i+1]
        w = weights[-(len(window_data)):]
        w = w / w.sum()
        result[i] = np.nansum(window_data * w)
    return result


def _decay_exp(x, window, halflife=None):
    """指数衰减加权均值"""
    if halflife is None:
        halflife = window / 2
    result = np.full_like(x, np.nan, dtype=float)
    alpha = 1 - np.exp(-np.log(2) / halflife)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        window_data = x[start:i+1]
        n = len(window_data)
        weights = np.array([(1 - alpha) ** j for j in range(n - 1, -1, -1)])
        weights = weights / weights.sum()
        result[i] = np.nansum(window_data * weights)
    return result


# ----------------- 时序算子（固定窗口）-----------------

def _ts_delay_1(x):
    """延迟1天"""
    result = np.roll(x, 1)
    result[0] = x[0]
    return result


def _ts_delay_5(x):
    """延迟5天"""
    result = np.roll(x, 5)
    result[:5] = x[:5]
    return result


def _ts_delay_10(x):
    """延迟10天"""
    result = np.roll(x, 10)
    result[:10] = x[:10]
    return result


def _ts_delay_20(x):
    """延迟20天"""
    result = np.roll(x, 20)
    result[:20] = x[:20]
    return result


def _ts_delta_1(x):
    """1日差分"""
    return x - _ts_delay_1(x)


def _ts_delta_5(x):
    """5日差分"""
    return x - _ts_delay_5(x)


def _ts_delta_10(x):
    """10日差分"""
    return x - _ts_delay_10(x)


def _ts_delta_20(x):
    """20日差分"""
    return x - _ts_delay_20(x)


# 均值
def _ts_mean_3(x):
    return _rolling_mean(x, 3)

def _ts_mean_5(x):
    return _rolling_mean(x, 5)

def _ts_mean_10(x):
    return _rolling_mean(x, 10)

def _ts_mean_20(x):
    return _rolling_mean(x, 20)

def _ts_mean_60(x):
    return _rolling_mean(x, 60)


# 标准差
def _ts_std_5(x):
    return _rolling_std(x, 5)

def _ts_std_10(x):
    return _rolling_std(x, 10)

def _ts_std_20(x):
    return _rolling_std(x, 20)


# 最大最小
def _ts_max_5(x):
    return _rolling_max(x, 5)

def _ts_max_10(x):
    return _rolling_max(x, 10)

def _ts_max_20(x):
    return _rolling_max(x, 20)

def _ts_min_5(x):
    return _rolling_min(x, 5)

def _ts_min_10(x):
    return _rolling_min(x, 10)

def _ts_min_20(x):
    return _rolling_min(x, 20)


# 求和
def _ts_sum_5(x):
    return _rolling_sum(x, 5)

def _ts_sum_10(x):
    return _rolling_sum(x, 10)

def _ts_sum_20(x):
    return _rolling_sum(x, 20)


# 排名
def _ts_rank_5(x):
    return _rolling_rank(x, 5)

def _ts_rank_10(x):
    return _rolling_rank(x, 10)

def _ts_rank_20(x):
    return _rolling_rank(x, 20)


# 偏度峰度
def _ts_skew_20(x):
    return _rolling_skew(x, 20)

def _ts_kurt_20(x):
    return _rolling_kurt(x, 20)


# 最值位置
def _ts_argmax_5(x):
    return _rolling_argmax(x, 5)

def _ts_argmax_10(x):
    return _rolling_argmax(x, 10)

def _ts_argmin_5(x):
    return _rolling_argmin(x, 5)

def _ts_argmin_10(x):
    return _rolling_argmin(x, 10)


# 衰减加权
def _ts_decay_5(x):
    return _decay_linear(x, 5)

def _ts_decay_10(x):
    return _decay_linear(x, 10)

def _ts_decay_20(x):
    return _decay_linear(x, 20)


# ----------------- 双变量时序算子 -----------------

def _ts_corr_10(x1, x2):
    return _rolling_corr(x1, x2, 10)

def _ts_corr_20(x1, x2):
    return _rolling_corr(x1, x2, 20)

def _ts_cov_10(x1, x2):
    return _rolling_cov(x1, x2, 10)

def _ts_cov_20(x1, x2):
    return _rolling_cov(x1, x2, 20)


# ----------------- 复合算子 -----------------

def _ts_zscore_10(x):
    """10日 Z-Score 标准化"""
    mean = _rolling_mean(x, 10)
    std = _rolling_std(x, 10)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(std > 1e-10, (x - mean) / std, 0.0)
    return np.clip(result, -5, 5)


def _ts_zscore_20(x):
    """20日 Z-Score 标准化"""
    mean = _rolling_mean(x, 20)
    std = _rolling_std(x, 20)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(std > 1e-10, (x - mean) / std, 0.0)
    return np.clip(result, -5, 5)


def _ts_pctchange_1(x):
    """1日收益率"""
    prev = _ts_delay_1(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(prev) > 1e-10, (x - prev) / prev, 0.0)
    return np.clip(result, -1, 1)


def _ts_pctchange_5(x):
    """5日收益率"""
    prev = _ts_delay_5(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(prev) > 1e-10, (x - prev) / prev, 0.0)
    return np.clip(result, -1, 1)


def _ts_momentum_10(x):
    """10日动量 (当前值 / 10日前值)"""
    prev = _ts_delay_10(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(prev) > 1e-10, x / prev, 1.0)
    return np.clip(result, 0.1, 10)


def _ts_momentum_20(x):
    """20日动量"""
    prev = _ts_delay_20(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(prev) > 1e-10, x / prev, 1.0)
    return np.clip(result, 0.1, 10)


# ============================================================
# GP Alpha 挖掘器 V2
# ============================================================

class GPAlphaMinerV2:
    """
    遗传规划 Alpha 因子挖掘器 V2
    
    工业级实现，使用截面数据和 IC 评估
    """
    
    def __init__(self,
                 population_size: int = 2000,
                 generations: int = 50,
                 tournament_size: int = 7,
                 p_crossover: float = 0.85,
                 p_subtree_mutation: float = 0.08,
                 p_hoist_mutation: float = 0.03,
                 p_point_mutation: float = 0.04,
                 max_samples: float = 0.9,
                 parsimony_coefficient: float = 0.0003,
                 init_depth: Tuple[int, int] = (3, 8),
                 random_state: int = 42,
                 n_jobs: int = -1,
                 verbose: int = 1):
        """
        初始化（工业级默认参数）
        
        Args:
            population_size: 种群大小 (工业级: 2000-5000)
            generations: 进化代数 (工业级: 50-100)
            tournament_size: 锦标赛大小 (工业级: 5-10，越小多样性越高)
            p_crossover: 交叉概率 (工业级: 0.8-0.9)
            p_subtree_mutation: 子树变异概率
            p_hoist_mutation: 提升变异概率
            p_point_mutation: 点变异概率
            max_samples: 样本采样比例
            parsimony_coefficient: 简洁性惩罚系数 (工业级: 0.0001-0.0005)
            init_depth: 初始树深度范围 (工业级: (3, 8))
            random_state: 随机种子
            n_jobs: 并行数
            verbose: 输出详细程度
        """
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.max_samples = max_samples
        self.parsimony_coefficient = parsimony_coefficient
        self.init_depth = init_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # 数据
        self.data_manager: Optional[PanelDataManager] = None
        self.train_dm: Optional[PanelDataManager] = None
        self.test_dm: Optional[PanelDataManager] = None
        
        # 结果
        self.best_programs = []
        self.best_factors: List[Dict] = []
        self.gp_model = None
        
        # 初始化函数集
        self._init_function_set()
    
    def _init_function_set(self):
        """初始化 gplearn 工业级函数集"""
        if not GPLEARN_AVAILABLE:
            return
        
        # 创建自定义函数
        self.gp_functions = [
            # ==================== 基础运算 (9个) ====================
            'add', 'sub', 'mul',
            make_function(function=_protected_div, name='div', arity=2),
            make_function(function=_protected_log, name='log', arity=1),
            make_function(function=_protected_sqrt, name='sqrt', arity=1),
            make_function(function=_sign, name='sign', arity=1),
            make_function(function=_abs, name='abs', arity=1),
            make_function(function=_neg, name='neg', arity=1),
            
            # ==================== 扩展运算 (6个) ====================
            make_function(function=_square, name='square', arity=1),
            make_function(function=_inv, name='inv', arity=1),
            make_function(function=_sigmoid, name='sigmoid', arity=1),
            make_function(function=_tanh, name='tanh', arity=1),
            make_function(function=_max2, name='max2', arity=2),
            make_function(function=_min2, name='min2', arity=2),
            
            # ==================== 延迟算子 (4个) ====================
            make_function(function=_ts_delay_1, name='delay1', arity=1),
            make_function(function=_ts_delay_5, name='delay5', arity=1),
            make_function(function=_ts_delay_10, name='delay10', arity=1),
            make_function(function=_ts_delay_20, name='delay20', arity=1),
            
            # ==================== 差分算子 (4个) ====================
            make_function(function=_ts_delta_1, name='delta1', arity=1),
            make_function(function=_ts_delta_5, name='delta5', arity=1),
            make_function(function=_ts_delta_10, name='delta10', arity=1),
            make_function(function=_ts_delta_20, name='delta20', arity=1),
            
            # ==================== 均值算子 (5个) ====================
            make_function(function=_ts_mean_3, name='mean3', arity=1),
            make_function(function=_ts_mean_5, name='mean5', arity=1),
            make_function(function=_ts_mean_10, name='mean10', arity=1),
            make_function(function=_ts_mean_20, name='mean20', arity=1),
            make_function(function=_ts_mean_60, name='mean60', arity=1),
            
            # ==================== 标准差算子 (3个) ====================
            make_function(function=_ts_std_5, name='std5', arity=1),
            make_function(function=_ts_std_10, name='std10', arity=1),
            make_function(function=_ts_std_20, name='std20', arity=1),
            
            # ==================== 最大值算子 (3个) ====================
            make_function(function=_ts_max_5, name='max5', arity=1),
            make_function(function=_ts_max_10, name='max10', arity=1),
            make_function(function=_ts_max_20, name='max20', arity=1),
            
            # ==================== 最小值算子 (3个) ====================
            make_function(function=_ts_min_5, name='min5', arity=1),
            make_function(function=_ts_min_10, name='min10', arity=1),
            make_function(function=_ts_min_20, name='min20', arity=1),
            
            # ==================== 求和算子 (3个) ====================
            make_function(function=_ts_sum_5, name='sum5', arity=1),
            make_function(function=_ts_sum_10, name='sum10', arity=1),
            make_function(function=_ts_sum_20, name='sum20', arity=1),
            
            # ==================== 排名算子 (3个) ====================
            make_function(function=_ts_rank_5, name='rank5', arity=1),
            make_function(function=_ts_rank_10, name='rank10', arity=1),
            make_function(function=_ts_rank_20, name='rank20', arity=1),
            
            # ==================== 高阶统计 (2个) ====================
            make_function(function=_ts_skew_20, name='skew20', arity=1),
            make_function(function=_ts_kurt_20, name='kurt20', arity=1),
            
            # ==================== 最值位置 (4个) ====================
            make_function(function=_ts_argmax_5, name='argmax5', arity=1),
            make_function(function=_ts_argmax_10, name='argmax10', arity=1),
            make_function(function=_ts_argmin_5, name='argmin5', arity=1),
            make_function(function=_ts_argmin_10, name='argmin10', arity=1),
            
            # ==================== 衰减加权 (3个) ====================
            make_function(function=_ts_decay_5, name='decay5', arity=1),
            make_function(function=_ts_decay_10, name='decay10', arity=1),
            make_function(function=_ts_decay_20, name='decay20', arity=1),
            
            # ==================== Z-Score 标准化 (2个) ====================
            make_function(function=_ts_zscore_10, name='zscore10', arity=1),
            make_function(function=_ts_zscore_20, name='zscore20', arity=1),
            
            # ==================== 收益率/动量 (4个) ====================
            make_function(function=_ts_pctchange_1, name='pctchg1', arity=1),
            make_function(function=_ts_pctchange_5, name='pctchg5', arity=1),
            make_function(function=_ts_momentum_10, name='mom10', arity=1),
            make_function(function=_ts_momentum_20, name='mom20', arity=1),
            
            # ==================== 双变量时序 (4个) ====================
            make_function(function=_ts_corr_10, name='corr10', arity=2),
            make_function(function=_ts_corr_20, name='corr20', arity=2),
            make_function(function=_ts_cov_10, name='cov10', arity=2),
            make_function(function=_ts_cov_20, name='cov20', arity=2),
        ]
        
        print(f"已加载 {len(self.gp_functions)} 个算子")
    
    def load_data(self,
                  pool_type: str = 'all',
                  start_date: str = None,
                  end_date: str = None,
                  train_ratio: float = 0.7,
                  use_cache: bool = True) -> 'GPAlphaMinerV2':
        """
        加载数据
        
        Args:
            pool_type: 股票池类型
            start_date: 开始日期
            end_date: 结束日期
            train_ratio: 训练集比例
            use_cache: 是否使用缓存
        
        Returns:
            self
        """
        print("=" * 60)
        print("加载数据")
        print("=" * 60)
        
        # 获取数据
        self.data_manager = PanelDataManager()
        self.data_manager.fetch(
            pool_type=pool_type,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache,
            verbose=True
        )
        
        # 切分训练集/测试集
        self.train_dm, self.test_dm = self.data_manager.split_train_test(train_ratio)
        
        print(f"\n训练集: {len(self.train_dm.dates)} 天 ({self.train_dm.start_date} ~ {self.train_dm.end_date})")
        print(f"测试集: {len(self.test_dm.dates)} 天 ({self.test_dm.start_date} ~ {self.test_dm.end_date})")
        
        return self
    
    def _prepare_training_data(self, 
                               dm: PanelDataManager,
                               forward_days: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备训练数据
        
        将 Panel 数据转换为 gplearn 需要的格式：
        - 按股票展开，每只股票的时序数据拼接
        - X: (样本数, 特征数)
        - y: (样本数,) 未来收益
        
        Args:
            dm: 数据管理器
            forward_days: 预测天数
        
        Returns:
            (X, y, feature_names)
        """
        features = dm.get_feature_panels()
        forward_return = dm.get_forward_return(forward_days)
        
        feature_names = list(features.keys())
        
        # 按股票拼接
        X_list = []
        y_list = []
        
        for symbol in dm.symbols:
            # 提取该股票的所有特征
            symbol_features = []
            for fname in feature_names:
                if symbol in features[fname].columns:
                    symbol_features.append(features[fname][symbol].values)
                else:
                    symbol_features.append(np.full(len(dm.dates), np.nan))
            
            X_symbol = np.column_stack(symbol_features)
            
            # 未来收益
            if symbol in forward_return.columns:
                y_symbol = forward_return[symbol].values
            else:
                y_symbol = np.full(len(dm.dates), np.nan)
            
            # 过滤无效行
            valid = ~(np.any(np.isnan(X_symbol), axis=1) | np.isnan(y_symbol))
            
            X_list.append(X_symbol[valid])
            y_list.append(y_symbol[valid])
        
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
        return X, y, feature_names
    
    def _create_ic_fitness(self, y_true: np.ndarray) -> Callable:
        """
        创建 IC 适应度函数
        
        注意：gplearn 的适应度函数需要最小化
        """
        def ic_fitness(y, y_pred, sample_weight):
            """计算负 IC（用于最小化）"""
            from scipy import stats
            
            # 过滤无效值
            valid = ~(np.isnan(y_pred) | np.isinf(y_pred))
            if np.sum(valid) < 50:
                return 1.0  # 惩罚无效因子
            
            try:
                ic, _ = stats.spearmanr(y_pred[valid], y[valid])
                if np.isnan(ic):
                    return 1.0
                return -abs(ic)  # 取绝对值，因为负 IC 也有价值
            except:
                return 1.0
        
        return make_fitness(function=ic_fitness, greater_is_better=False)
    
    def mine(self,
             forward_days: int = 1,
             top_n: int = 10) -> List[Dict]:
        """
        执行因子挖掘
        
        Args:
            forward_days: 预测未来收益天数
            top_n: 返回最佳因子数量
        
        Returns:
            最佳因子列表
        """
        if not GPLEARN_AVAILABLE:
            raise ImportError("请先安装 gplearn: pip install gplearn")
        
        if self.train_dm is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        print("\n" + "=" * 60)
        print("遗传规划因子挖掘 V2")
        print("=" * 60)
        print(f"种群大小: {self.population_size}")
        print(f"进化代数: {self.generations}")
        print(f"预测天数: {forward_days}")
        print(f"返回因子: {top_n}")
        
        # 准备训练数据
        print("\n准备训练数据...")
        X_train, y_train, feature_names = self._prepare_training_data(
            self.train_dm, forward_days
        )
        print(f"训练样本: {len(X_train)}")
        print(f"特征数量: {len(feature_names)}")
        print(f"特征列表: {feature_names}")
        
        # 创建适应度函数
        fitness = self._create_ic_fitness(y_train)
        
        # 创建 GP 模型
        print("\n开始进化...")
        print("-" * 60)
        
        self.gp_model = SymbolicTransformer(
            population_size=self.population_size,
            generations=self.generations,
            tournament_size=self.tournament_size,
            stopping_criteria=-1.0,  # 设为 -1 禁用早停（因为我们的 fitness 是负值）
            p_crossover=self.p_crossover,
            p_subtree_mutation=self.p_subtree_mutation,
            p_hoist_mutation=self.p_hoist_mutation,
            p_point_mutation=self.p_point_mutation,
            max_samples=self.max_samples,
            parsimony_coefficient=self.parsimony_coefficient,
            init_depth=self.init_depth,
            function_set=self.gp_functions,
            feature_names=feature_names,
            metric=fitness,
            n_components=top_n,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        
        # 训练（支持中断保存）
        try:
            self.gp_model.fit(X_train, y_train)
        except KeyboardInterrupt:
            print("\n" + "=" * 60)
            print("⚠️  检测到中断，保存当前最佳结果...")
            print("=" * 60)
        
        # 提取最佳因子（无论是否中断都执行）
        self._extract_and_evaluate_factors(forward_days, feature_names, top_n)
        
        return self.best_factors
    
    def _extract_and_evaluate_factors(self, forward_days: int, feature_names: List[str], top_n: int):
        """提取并评估最佳因子"""
        print("\n" + "=" * 60)
        print(f"评估 Top {top_n} 因子")
        print("=" * 60)
        
        # 检查是否有结果
        if not hasattr(self.gp_model, '_best_programs') or self.gp_model._best_programs is None:
            print("⚠️  没有找到有效因子")
            return
        
        self.best_programs = [p for p in self.gp_model._best_programs if p is not None]
        
        if not self.best_programs:
            print("⚠️  没有找到有效因子")
            return
            
        self.best_factors = []
        
        for i, program in enumerate(self.best_programs):
            print(f"\n[因子 #{i+1}]")
            print(f"  公式: {program}")
            print(f"  复杂度: 长度={program.length_}, 深度={program.depth_}")
            
            # 在训练集上评估
            train_metrics = self._evaluate_program(
                program, self.train_dm, forward_days, feature_names, "训练集"
            )
            
            # 在测试集上评估
            test_metrics = self._evaluate_program(
                program, self.test_dm, forward_days, feature_names, "测试集"
            )
            
            factor_info = {
                'rank': i + 1,
                'formula': str(program),
                'length': program.length_,
                'depth': program.depth_,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'program': program,
            }
            
            self.best_factors.append(factor_info)
        
        # 按测试集得分排序
        self.best_factors.sort(
            key=lambda x: x['test_metrics'].get('composite_score', 0),
            reverse=True
        )
        
        # 更新排名
        for i, f in enumerate(self.best_factors):
            f['rank'] = i + 1
    
    def _evaluate_program(self,
                          program,
                          dm: PanelDataManager,
                          forward_days: int,
                          feature_names: List[str],
                          dataset_name: str) -> Dict:
        """
        评估单个因子程序
        
        Args:
            program: gplearn 程序
            dm: 数据管理器
            forward_days: 预测天数
            feature_names: 特征名列表
            dataset_name: 数据集名称
        
        Returns:
            评估指标字典
        """
        features = dm.get_feature_panels()
        forward_return = dm.get_forward_return(forward_days)
        
        # 计算因子面板
        factor_panel = pd.DataFrame(index=dm.dates, columns=dm.symbols, dtype=float)
        
        for symbol in dm.symbols:
            # 提取该股票的特征
            symbol_features = []
            for fname in feature_names:
                if symbol in features[fname].columns:
                    symbol_features.append(features[fname][symbol].values)
                else:
                    symbol_features.append(np.full(len(dm.dates), np.nan))
            
            X_symbol = np.column_stack(symbol_features)
            
            # 过滤无效行
            valid = ~np.any(np.isnan(X_symbol), axis=1)
            
            # 计算因子值
            factor_values = np.full(len(dm.dates), np.nan)
            if np.sum(valid) > 0:
                try:
                    factor_values[valid] = program.execute(X_symbol[valid])
                except:
                    pass
            
            factor_panel[symbol] = factor_values
        
        # 预处理因子
        factor_panel = preprocess_factor(factor_panel)
        
        # 评估
        try:
            evaluator = FactorEvaluator(factor_panel, forward_return, forward_days)
            metrics = evaluator.evaluate(verbose=False)
            
            print(f"  {dataset_name}: IC={metrics['ic_mean']:.4f}, "
                  f"ICIR={metrics['ic_ir']:.4f}, "
                  f"Sharpe={metrics['long_short_sharpe']:.2f}, "
                  f"Score={metrics['composite_score']:.4f}")
            
            return metrics
        except Exception as e:
            print(f"  {dataset_name}: 评估失败 - {e}")
            return {'composite_score': 0}
    
    def get_factor_panel(self, 
                         factor_idx: int = 0,
                         dm: PanelDataManager = None) -> pd.DataFrame:
        """
        获取因子面板
        
        Args:
            factor_idx: 因子索引（0 为最佳）
            dm: 数据管理器，默认使用全量数据
        
        Returns:
            因子面板 DataFrame
        """
        if not self.best_factors:
            raise ValueError("请先调用 mine() 挖掘因子")
        
        if dm is None:
            dm = self.data_manager
        
        program = self.best_factors[factor_idx]['program']
        features = dm.get_feature_panels()
        feature_names = list(features.keys())
        
        factor_panel = pd.DataFrame(index=dm.dates, columns=dm.symbols, dtype=float)
        
        for symbol in dm.symbols:
            symbol_features = []
            for fname in feature_names:
                if symbol in features[fname].columns:
                    symbol_features.append(features[fname][symbol].values)
                else:
                    symbol_features.append(np.full(len(dm.dates), np.nan))
            
            X_symbol = np.column_stack(symbol_features)
            valid = ~np.any(np.isnan(X_symbol), axis=1)
            
            factor_values = np.full(len(dm.dates), np.nan)
            if np.sum(valid) > 0:
                try:
                    factor_values[valid] = program.execute(X_symbol[valid])
                except:
                    pass
            
            factor_panel[symbol] = factor_values
        
        return preprocess_factor(factor_panel)
    
    def print_summary(self, top_n: int = 5):
        """打印结果摘要"""
        if not self.best_factors:
            print("无因子结果")
            return
        
        print("\n" + "=" * 70)
        print("🏆 最佳因子汇总（按测试集得分排序）")
        print("=" * 70)
        
        for f in self.best_factors[:top_n]:
            train = f['train_metrics']
            test = f['test_metrics']
            
            print(f"\n[#{f['rank']}] {f['formula']}")
            print(f"    复杂度: 长度={f['length']}, 深度={f['depth']}")
            print(f"    训练集: IC={train.get('ic_mean', 0):.4f}, "
                  f"ICIR={train.get('ic_ir', 0):.4f}, "
                  f"Sharpe={train.get('long_short_sharpe', 0):.2f}")
            print(f"    测试集: IC={test.get('ic_mean', 0):.4f}, "
                  f"ICIR={test.get('ic_ir', 0):.4f}, "
                  f"Sharpe={test.get('long_short_sharpe', 0):.2f}")
            print(f"    综合得分: 训练={train.get('composite_score', 0):.4f}, "
                  f"测试={test.get('composite_score', 0):.4f}")
    
    def save(self, filepath: str):
        """保存模型"""
        save_data = {
            'best_factors': [
                {k: v for k, v in f.items() if k != 'program'}
                for f in self.best_factors
            ],
            'params': {
                'population_size': self.population_size,
                'generations': self.generations,
            }
        }
        
        # 保存程序（单独处理）
        if self.gp_model is not None:
            save_data['gp_model'] = self.gp_model
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"模型已保存至 {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.best_factors = save_data.get('best_factors', [])
        self.gp_model = save_data.get('gp_model', None)
        
        if self.gp_model is not None:
            self.best_programs = [p for p in self.gp_model._best_programs if p is not None]
            
            # 恢复 program 引用
            for i, f in enumerate(self.best_factors):
                if i < len(self.best_programs):
                    f['program'] = self.best_programs[i]
        
        print(f"模型已加载: {len(self.best_factors)} 个因子")


def quick_mine(pool_type: str = 'nasdaq100',
               population_size: int = 300,
               generations: int = 15,
               forward_days: int = 5,
               top_n: int = 5) -> List[Dict]:
    """
    快速挖掘入口
    
    Args:
        pool_type: 股票池类型
        population_size: 种群大小
        generations: 进化代数
        forward_days: 预测天数
        top_n: 返回因子数
    
    Returns:
        最佳因子列表
    """
    miner = GPAlphaMinerV2(
        population_size=population_size,
        generations=generations,
        verbose=1
    )
    
    miner.load_data(pool_type=pool_type, train_ratio=0.7)
    factors = miner.mine(forward_days=forward_days, top_n=top_n)
    miner.print_summary(top_n)
    
    return factors


if __name__ == '__main__':
    # 测试
    factors = quick_mine(
        pool_type='nasdaq100',
        population_size=200,
        generations=10,
        forward_days=5,
        top_n=5
    )
