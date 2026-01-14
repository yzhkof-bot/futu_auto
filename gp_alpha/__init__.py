"""
遗传规划 (Genetic Programming) Alpha 因子挖掘模块 V2

工业级实现:
- Panel 数据结构（日期 × 股票）
- 截面 IC 评估
- 分层回测
- 训练/测试集切分

使用方法:
    from gp_alpha import GPAlphaMinerV2, quick_mine
    
    # 快速挖掘
    factors = quick_mine(pool_type='nasdaq100', generations=15)
    
    # 完整流程
    miner = GPAlphaMinerV2(population_size=500, generations=20)
    miner.load_data(pool_type='all')
    factors = miner.mine(forward_days=5, top_n=10)
    miner.print_summary()
"""

# V2 版本（工业级）
from .data_manager import PanelDataManager
from .factor_engine import (
    FactorEngine,
    ts_delay, ts_delta, ts_mean, ts_std, ts_max, ts_min, ts_rank, ts_sum,
    cs_rank, cs_zscore, cs_demean,
    winsorize, mad_winsorize, neutralize, preprocess_factor,
    alpha_momentum, alpha_reversal, alpha_volatility
)
from .evaluator import FactorEvaluator, quick_evaluate
from .gp_miner_v2 import GPAlphaMinerV2, quick_mine
from .visualize_v2 import (
    plot_ic_analysis, plot_group_returns, plot_ic_decay,
    plot_factor_comparison, plot_full_report
)

# V1 版本（保留兼容）
from .gp_alpha_miner import GPAlphaMiner
from .fitness import calculate_ic, calculate_icir, evaluate_alpha

__all__ = [
    # V2 核心
    'GPAlphaMinerV2',
    'quick_mine',
    'PanelDataManager',
    'FactorEngine',
    'FactorEvaluator',
    'quick_evaluate',
    
    # 因子算子
    'ts_delay', 'ts_delta', 'ts_mean', 'ts_std', 'ts_max', 'ts_min', 'ts_rank', 'ts_sum',
    'cs_rank', 'cs_zscore', 'cs_demean',
    'winsorize', 'mad_winsorize', 'neutralize', 'preprocess_factor',
    
    # 预定义因子
    'alpha_momentum', 'alpha_reversal', 'alpha_volatility',
    
    # 可视化
    'plot_ic_analysis', 'plot_group_returns', 'plot_ic_decay',
    'plot_factor_comparison', 'plot_full_report',
    
    # V1 兼容
    'GPAlphaMiner',
    'calculate_ic', 'calculate_icir', 'evaluate_alpha',
]
