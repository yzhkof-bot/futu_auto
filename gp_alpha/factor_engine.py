"""
截面因子计算引擎

支持:
1. 时序算子 - 单股票时间维度计算
2. 截面算子 - 同一天跨股票计算
3. 因子预处理 - 去极值、标准化、中性化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Union
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# 时序算子 (Time-Series Operators)
# 对每只股票的时间序列独立计算
# ============================================================

def ts_delay(panel: pd.DataFrame, d: int = 1) -> pd.DataFrame:
    """时序延迟 (lag)"""
    return panel.shift(d)


def ts_delta(panel: pd.DataFrame, d: int = 1) -> pd.DataFrame:
    """时序差分"""
    return panel - panel.shift(d)


def ts_pct_change(panel: pd.DataFrame, d: int = 1) -> pd.DataFrame:
    """时序百分比变化"""
    return panel.pct_change(d)


def ts_sum(panel: pd.DataFrame, d: int = 5) -> pd.DataFrame:
    """滚动求和"""
    return panel.rolling(window=d, min_periods=1).sum()


def ts_mean(panel: pd.DataFrame, d: int = 5) -> pd.DataFrame:
    """滚动均值"""
    return panel.rolling(window=d, min_periods=1).mean()


def ts_std(panel: pd.DataFrame, d: int = 5) -> pd.DataFrame:
    """滚动标准差"""
    return panel.rolling(window=d, min_periods=2).std()


def ts_max(panel: pd.DataFrame, d: int = 5) -> pd.DataFrame:
    """滚动最大值"""
    return panel.rolling(window=d, min_periods=1).max()


def ts_min(panel: pd.DataFrame, d: int = 5) -> pd.DataFrame:
    """滚动最小值"""
    return panel.rolling(window=d, min_periods=1).min()


def ts_rank(panel: pd.DataFrame, d: int = 5) -> pd.DataFrame:
    """
    滚动时序排名（百分位）
    当前值在过去 d 天中的排名位置
    """
    def rank_pct(x):
        if len(x) < 2:
            return 0.5
        return stats.rankdata(x)[-1] / len(x)
    
    return panel.rolling(window=d, min_periods=2).apply(rank_pct, raw=True)


def ts_corr(panel1: pd.DataFrame, panel2: pd.DataFrame, d: int = 10) -> pd.DataFrame:
    """滚动相关系数"""
    return panel1.rolling(window=d, min_periods=5).corr(panel2)


def ts_cov(panel1: pd.DataFrame, panel2: pd.DataFrame, d: int = 10) -> pd.DataFrame:
    """滚动协方差"""
    return panel1.rolling(window=d, min_periods=5).cov(panel2)


def ts_zscore(panel: pd.DataFrame, d: int = 20) -> pd.DataFrame:
    """滚动 Z-score 标准化"""
    mean = ts_mean(panel, d)
    std = ts_std(panel, d)
    return (panel - mean) / (std + 1e-10)


def ts_returns_volatility(panel: pd.DataFrame, d: int = 20) -> pd.DataFrame:
    """滚动收益率波动率"""
    returns = panel.pct_change()
    return returns.rolling(window=d, min_periods=5).std()


def ts_skew(panel: pd.DataFrame, d: int = 20) -> pd.DataFrame:
    """滚动偏度"""
    return panel.rolling(window=d, min_periods=10).skew()


def ts_kurt(panel: pd.DataFrame, d: int = 20) -> pd.DataFrame:
    """滚动峰度"""
    return panel.rolling(window=d, min_periods=10).kurt()


# ============================================================
# 截面算子 (Cross-Sectional Operators)
# 同一天跨所有股票计算
# ============================================================

def cs_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """
    截面排名（百分位）
    每天对所有股票排名，归一化到 [0, 1]
    """
    return panel.rank(axis=1, pct=True)


def cs_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """
    截面 Z-score 标准化
    每天对所有股票标准化
    """
    mean = panel.mean(axis=1)
    std = panel.std(axis=1)
    return panel.sub(mean, axis=0).div(std + 1e-10, axis=0)


def cs_demean(panel: pd.DataFrame) -> pd.DataFrame:
    """截面去均值"""
    mean = panel.mean(axis=1)
    return panel.sub(mean, axis=0)


def cs_scale(panel: pd.DataFrame, target: float = 1.0) -> pd.DataFrame:
    """
    截面缩放
    使每天的因子绝对值之和等于 target
    """
    abs_sum = panel.abs().sum(axis=1)
    return panel.div(abs_sum + 1e-10, axis=0) * target


# ============================================================
# 因子预处理 (Factor Preprocessing)
# ============================================================

def winsorize(panel: pd.DataFrame, 
              lower_pct: float = 0.01, 
              upper_pct: float = 0.99,
              axis: int = 1) -> pd.DataFrame:
    """
    去极值（Winsorize）
    
    Args:
        panel: 因子面板
        lower_pct: 下界百分位
        upper_pct: 上界百分位
        axis: 0=时序方向, 1=截面方向
    
    Returns:
        去极值后的面板
    """
    if axis == 1:
        # 截面去极值（每天独立）
        lower = panel.quantile(lower_pct, axis=1)
        upper = panel.quantile(upper_pct, axis=1)
        return panel.clip(lower=lower, upper=upper, axis=0)
    else:
        # 时序去极值（每只股票独立）
        lower = panel.quantile(lower_pct, axis=0)
        upper = panel.quantile(upper_pct, axis=0)
        return panel.clip(lower=lower, upper=upper, axis=1)


def mad_winsorize(panel: pd.DataFrame, n_mad: float = 5.0) -> pd.DataFrame:
    """
    MAD 去极值
    使用中位数绝对偏差，更稳健
    
    Args:
        panel: 因子面板
        n_mad: MAD 倍数阈值
    
    Returns:
        去极值后的面板
    """
    median = panel.median(axis=1)
    mad = (panel.sub(median, axis=0)).abs().median(axis=1)
    
    lower = median - n_mad * mad
    upper = median + n_mad * mad
    
    return panel.clip(lower=lower, upper=upper, axis=0)


def neutralize(factor_panel: pd.DataFrame,
               neutralize_panels: List[pd.DataFrame],
               method: str = 'regression') -> pd.DataFrame:
    """
    因子中性化
    
    Args:
        factor_panel: 待中性化的因子面板
        neutralize_panels: 用于中性化的因子面板列表（如市值、行业）
        method: 'regression' 或 'demean'
    
    Returns:
        中性化后的因子面板
    """
    if method == 'demean':
        # 简单去均值（用于分组中性化）
        result = factor_panel.copy()
        for np_panel in neutralize_panels:
            group_mean = factor_panel.groupby(np_panel, axis=1).transform('mean')
            result = result - group_mean
        return result
    
    elif method == 'regression':
        # 回归中性化
        result = pd.DataFrame(index=factor_panel.index, columns=factor_panel.columns)
        
        for date in factor_panel.index:
            y = factor_panel.loc[date].values
            X = np.column_stack([np.loc[date].values for np in neutralize_panels])
            
            # 过滤 NaN
            valid = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
            if np.sum(valid) < 10:
                result.loc[date] = np.nan
                continue
            
            # OLS 回归
            X_valid = np.column_stack([np.ones(np.sum(valid)), X[valid]])
            y_valid = y[valid]
            
            try:
                beta = np.linalg.lstsq(X_valid, y_valid, rcond=None)[0]
                residuals = y_valid - X_valid @ beta
                result.loc[date, valid] = residuals
            except:
                result.loc[date] = np.nan
        
        return result.astype(float)
    
    else:
        raise ValueError(f"未知方法: {method}")


def preprocess_factor(panel: pd.DataFrame,
                      winsorize_pct: float = 0.01,
                      standardize: bool = True,
                      demean: bool = True) -> pd.DataFrame:
    """
    标准因子预处理流程
    
    1. 去极值 (Winsorize)
    2. 截面标准化 (Z-score)
    
    Args:
        panel: 原始因子面板
        winsorize_pct: 去极值百分位
        standardize: 是否标准化
        demean: 是否去均值
    
    Returns:
        预处理后的因子面板
    """
    result = panel.copy()
    
    # 1. 去极值
    if winsorize_pct > 0:
        result = winsorize(result, winsorize_pct, 1 - winsorize_pct)
    
    # 2. 标准化
    if standardize:
        result = cs_zscore(result)
    elif demean:
        result = cs_demean(result)
    
    return result


# ============================================================
# 因子表达式计算引擎
# ============================================================

class FactorEngine:
    """
    因子计算引擎
    
    支持从特征面板计算复杂因子表达式
    """
    
    # 可用的时序算子
    TS_OPERATORS = {
        'delay1': lambda x: ts_delay(x, 1),
        'delay5': lambda x: ts_delay(x, 5),
        'delay10': lambda x: ts_delay(x, 10),
        'delta1': lambda x: ts_delta(x, 1),
        'delta5': lambda x: ts_delta(x, 5),
        'delta10': lambda x: ts_delta(x, 10),
        'mean5': lambda x: ts_mean(x, 5),
        'mean10': lambda x: ts_mean(x, 10),
        'mean20': lambda x: ts_mean(x, 20),
        'mean60': lambda x: ts_mean(x, 60),
        'std5': lambda x: ts_std(x, 5),
        'std10': lambda x: ts_std(x, 10),
        'std20': lambda x: ts_std(x, 20),
        'max5': lambda x: ts_max(x, 5),
        'max10': lambda x: ts_max(x, 10),
        'min5': lambda x: ts_min(x, 5),
        'min10': lambda x: ts_min(x, 10),
        'rank5': lambda x: ts_rank(x, 5),
        'rank10': lambda x: ts_rank(x, 10),
        'sum5': lambda x: ts_sum(x, 5),
        'sum10': lambda x: ts_sum(x, 10),
        'zscore20': lambda x: ts_zscore(x, 20),
        'vol20': lambda x: ts_returns_volatility(x, 20),
        'skew20': lambda x: ts_skew(x, 20),
    }
    
    # 可用的截面算子
    CS_OPERATORS = {
        'cs_rank': cs_rank,
        'cs_zscore': cs_zscore,
        'cs_demean': cs_demean,
        'cs_scale': cs_scale,
    }
    
    # 可用的基础运算
    BASIC_OPERATORS = {
        'add': lambda x, y: x + y,
        'sub': lambda x, y: x - y,
        'mul': lambda x, y: x * y,
        'div': lambda x, y: x / (y + 1e-10),
        'log': lambda x: np.log(np.abs(x) + 1e-10),
        'sqrt': lambda x: np.sqrt(np.abs(x)),
        'abs': lambda x: np.abs(x),
        'neg': lambda x: -x,
        'sign': lambda x: np.sign(x),
        'pow2': lambda x: x ** 2,
        'pow3': lambda x: x ** 3,
    }
    
    def __init__(self, feature_panels: Dict[str, pd.DataFrame]):
        """
        初始化
        
        Args:
            feature_panels: 特征名 -> DataFrame 的字典
        """
        self.features = feature_panels
        self.feature_names = list(feature_panels.keys())
    
    def get_feature(self, name: str) -> pd.DataFrame:
        """获取特征面板"""
        if name not in self.features:
            raise ValueError(f"未知特征: {name}，可用: {self.feature_names}")
        return self.features[name].copy()
    
    def apply_ts_operator(self, panel: pd.DataFrame, op_name: str) -> pd.DataFrame:
        """应用时序算子"""
        if op_name not in self.TS_OPERATORS:
            raise ValueError(f"未知时序算子: {op_name}")
        return self.TS_OPERATORS[op_name](panel)
    
    def apply_cs_operator(self, panel: pd.DataFrame, op_name: str) -> pd.DataFrame:
        """应用截面算子"""
        if op_name not in self.CS_OPERATORS:
            raise ValueError(f"未知截面算子: {op_name}")
        return self.CS_OPERATORS[op_name](panel)
    
    def compute_alpha(self, 
                      expression: Callable[[Dict[str, pd.DataFrame]], pd.DataFrame],
                      preprocess: bool = True) -> pd.DataFrame:
        """
        计算 Alpha 因子
        
        Args:
            expression: 因子表达式函数，接收特征字典，返回因子面板
            preprocess: 是否预处理
        
        Returns:
            因子面板
        """
        # 计算原始因子
        factor = expression(self.features)
        
        # 预处理
        if preprocess:
            factor = preprocess_factor(factor)
        
        return factor
    
    def list_operators(self) -> Dict[str, List[str]]:
        """列出所有可用算子"""
        return {
            'time_series': list(self.TS_OPERATORS.keys()),
            'cross_sectional': list(self.CS_OPERATORS.keys()),
            'basic': list(self.BASIC_OPERATORS.keys()),
            'features': self.feature_names,
        }


# ============================================================
# 预定义因子模板
# ============================================================

def alpha_momentum(features: Dict[str, pd.DataFrame], d: int = 20) -> pd.DataFrame:
    """动量因子: 过去 d 日收益率"""
    return features['close'].pct_change(d)


def alpha_reversal(features: Dict[str, pd.DataFrame], d: int = 5) -> pd.DataFrame:
    """反转因子: 过去 d 日收益率的负值"""
    return -features['close'].pct_change(d)


def alpha_volatility(features: Dict[str, pd.DataFrame], d: int = 20) -> pd.DataFrame:
    """波动率因子: 过去 d 日收益率标准差"""
    return features['close'].pct_change().rolling(d).std()


def alpha_volume_price(features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """量价因子: 成交量变化 / 价格变化"""
    vol_change = features['volume'].pct_change(5)
    price_change = features['close'].pct_change(5)
    return vol_change / (price_change.abs() + 1e-10)


def alpha_price_range(features: Dict[str, pd.DataFrame], d: int = 10) -> pd.DataFrame:
    """价格区间因子: (收盘 - 最低) / (最高 - 最低)"""
    high_d = ts_max(features['high'], d)
    low_d = ts_min(features['low'], d)
    return (features['close'] - low_d) / (high_d - low_d + 1e-10)


def alpha_overnight_gap(features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """隔夜跳空因子: 开盘价 / 昨收盘价 - 1"""
    return features['open'] / features['close'].shift(1) - 1


def alpha_intraday_return(features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """日内收益因子: 收盘价 / 开盘价 - 1"""
    return features['close'] / features['open'] - 1


if __name__ == '__main__':
    # 测试
    from data_manager import PanelDataManager
    
    dm = PanelDataManager()
    dm.fetch(pool_type='nasdaq100', use_cache=True)
    
    features = dm.get_feature_panels()
    engine = FactorEngine(features)
    
    print("可用算子:")
    for category, ops in engine.list_operators().items():
        print(f"  {category}: {len(ops)} 个")
    
    # 计算动量因子
    momentum = alpha_momentum(features, d=20)
    momentum = preprocess_factor(momentum)
    
    print("\n动量因子 (前5行, 前5列):")
    print(momentum.iloc[:5, :5])
