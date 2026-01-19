"""
遗传规划 Alpha 因子挖掘器

使用 gplearn 进行因子公式的自动进化
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Callable
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

# gplearn 导入
try:
    from gplearn.genetic import SymbolicRegressor, SymbolicTransformer
    from gplearn.functions import make_function
    from gplearn.fitness import make_fitness
    GPLEARN_AVAILABLE = True
except ImportError:
    GPLEARN_AVAILABLE = False
    print("警告: gplearn 未安装，请运行: pip install gplearn")

from .operators import (
    protected_div, protected_log, protected_sqrt,
    ts_delay, ts_delta, ts_mean, ts_std, ts_max, ts_min, ts_rank, ts_sum,
    abs_op, neg  # sign 已移除
)
from .fitness import calculate_ic, calculate_icir, evaluate_alpha, AlphaFitness

# 导入统一股票池
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.stock_pool import get_stock_pool, ALL_STOCKS


class GPAlphaMiner:
    """
    遗传规划 Alpha 因子挖掘器
    
    使用进化算法自动发现有效的交易因子公式
    """
    
    def __init__(self, 
                 population_size: int = 500,
                 generations: int = 20,
                 tournament_size: int = 20,
                 stopping_criteria: float = 0.0,
                 p_crossover: float = 0.7,
                 p_subtree_mutation: float = 0.1,
                 p_hoist_mutation: float = 0.05,
                 p_point_mutation: float = 0.1,
                 max_samples: float = 0.9,
                 parsimony_coefficient: float = 0.001,
                 init_depth: Tuple[int, int] = (2, 6),
                 random_state: int = 42,
                 n_jobs: int = -1,
                 verbose: int = 1):
        """
        初始化遗传规划挖掘器
        
        Args:
            population_size: 种群大小（每代公式数量）
            generations: 进化代数
            tournament_size: 锦标赛选择大小
            stopping_criteria: 停止条件（适应度阈值）
            p_crossover: 交叉概率
            p_subtree_mutation: 子树变异概率
            p_hoist_mutation: 提升变异概率
            p_point_mutation: 点变异概率
            max_samples: 每代使用的样本比例
            parsimony_coefficient: 简洁性系数（惩罚复杂公式）
            init_depth: 初始树深度范围
            random_state: 随机种子
            n_jobs: 并行任务数（-1 使用所有核心）
            verbose: 输出详细程度
        """
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.stopping_criteria = stopping_criteria
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
        
        # 存储结果
        self.best_programs = []
        self.evolution_history = []
        self.data_cache = {}
        
        # 初始化自定义函数
        self._init_custom_functions()
    
    def _init_custom_functions(self):
        """初始化 gplearn 自定义函数"""
        if not GPLEARN_AVAILABLE:
            return
        
        # 基础运算
        self.gp_div = make_function(
            function=protected_div,
            name='div',
            arity=2
        )
        
        self.gp_log = make_function(
            function=protected_log,
            name='log',
            arity=1
        )
        
        self.gp_sqrt = make_function(
            function=protected_sqrt,
            name='sqrt',
            arity=1
        )
        
        # sign 已移除：会导致因子值只有 -1/0/+1，截面无法分层，Sharpe=nan
        
        self.gp_abs = make_function(
            function=abs_op,
            name='abs',
            arity=1
        )
        
        self.gp_neg = make_function(
            function=neg,
            name='neg',
            arity=1
        )
        
        # 时序函数（固定窗口）
        self.gp_ts_delay_1 = make_function(
            function=lambda x: ts_delay(x, 1),
            name='delay1',
            arity=1
        )
        
        self.gp_ts_delay_5 = make_function(
            function=lambda x: ts_delay(x, 5),
            name='delay5',
            arity=1
        )
        
        self.gp_ts_delta_1 = make_function(
            function=lambda x: ts_delta(x, 1),
            name='delta1',
            arity=1
        )
        
        self.gp_ts_delta_5 = make_function(
            function=lambda x: ts_delta(x, 5),
            name='delta5',
            arity=1
        )
        
        self.gp_ts_mean_5 = make_function(
            function=lambda x: ts_mean(x, 5),
            name='mean5',
            arity=1
        )
        
        self.gp_ts_mean_10 = make_function(
            function=lambda x: ts_mean(x, 10),
            name='mean10',
            arity=1
        )
        
        self.gp_ts_mean_20 = make_function(
            function=lambda x: ts_mean(x, 20),
            name='mean20',
            arity=1
        )
        
        self.gp_ts_std_5 = make_function(
            function=lambda x: ts_std(x, 5),
            name='std5',
            arity=1
        )
        
        self.gp_ts_std_10 = make_function(
            function=lambda x: ts_std(x, 10),
            name='std10',
            arity=1
        )
        
        self.gp_ts_max_5 = make_function(
            function=lambda x: ts_max(x, 5),
            name='max5',
            arity=1
        )
        
        self.gp_ts_min_5 = make_function(
            function=lambda x: ts_min(x, 5),
            name='min5',
            arity=1
        )
        
        self.gp_ts_rank_5 = make_function(
            function=lambda x: ts_rank(x, 5),
            name='rank5',
            arity=1
        )
        
        # 函数集合
        self.function_set = [
            'add', 'sub', 'mul',  # 基础运算
            self.gp_div,
            self.gp_log,
            self.gp_sqrt,
            # self.gp_sign,  # 已移除：因子值离散化导致 Sharpe=nan
            self.gp_abs,
            self.gp_neg,
            self.gp_ts_delay_1,
            self.gp_ts_delay_5,
            self.gp_ts_delta_1,
            self.gp_ts_delta_5,
            self.gp_ts_mean_5,
            self.gp_ts_mean_10,
            self.gp_ts_mean_20,
            self.gp_ts_std_5,
            self.gp_ts_std_10,
            self.gp_ts_max_5,
            self.gp_ts_min_5,
            self.gp_ts_rank_5,
        ]
    
    def fetch_data(self, symbols: List[str], 
                   start_date: str = None, 
                   end_date: str = None,
                   use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        获取股票数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
        
        Returns:
            {symbol: DataFrame} 字典
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        
        data = {}
        
        for symbol in symbols:
            cache_key = f"{symbol}_{start_date}_{end_date}"
            
            if use_cache and cache_key in self.data_cache:
                data[symbol] = self.data_cache[cache_key]
                continue
            
            try:
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if not df.empty and len(df) >= 100:
                    data[symbol] = df
                    if use_cache:
                        self.data_cache[cache_key] = df
                    
                    if self.verbose > 0:
                        print(f"  ✓ {symbol}: {len(df)} 天数据")
            except Exception as e:
                if self.verbose > 0:
                    print(f"  ✗ {symbol}: 获取失败 - {e}")
        
        return data
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        准备特征矩阵
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            (特征矩阵, 特征名称列表)
        """
        features = {}
        
        # 基础价格特征
        features['open'] = df['Open'].values
        features['high'] = df['High'].values
        features['low'] = df['Low'].values
        features['close'] = df['Close'].values
        features['volume'] = df['Volume'].values
        
        # 价格比率
        features['hl_ratio'] = (df['High'] / df['Low']).values
        features['co_ratio'] = (df['Close'] / df['Open']).values
        features['oc_range'] = ((df['Close'] - df['Open']) / df['Open']).values
        
        # 成交量相关
        features['vol_price'] = (df['Volume'] * df['Close']).values
        
        # 简单收益
        features['return_1'] = df['Close'].pct_change(1).values
        features['return_5'] = df['Close'].pct_change(5).values
        features['return_10'] = df['Close'].pct_change(10).values
        
        # VWAP 近似
        features['vwap'] = ((df['High'] + df['Low'] + df['Close']) / 3).values
        
        # 构建特征矩阵
        feature_names = list(features.keys())
        X = np.column_stack([features[name] for name in feature_names])
        
        return X, feature_names
    
    def _create_fitness(self, prices: np.ndarray, forward_days: int = 1):
        """创建适应度函数"""
        # 计算未来收益
        returns = np.zeros_like(prices)
        returns[:-forward_days] = (
            (prices[forward_days:] - prices[:-forward_days]) / prices[:-forward_days]
        )
        returns[-forward_days:] = np.nan
        
        def ic_fitness(y, y_pred, sample_weight):
            """IC 适应度（gplearn 接口）"""
            ic = calculate_ic(y_pred, returns)
            return -abs(ic)  # 最小化负 IC
        
        return make_fitness(function=ic_fitness, greater_is_better=False)
    
    def mine(self, 
             symbols: List[str] = None,
             n_symbols: int = 20,
             start_date: str = None,
             end_date: str = None,
             forward_days: int = 1,
             top_n: int = 10) -> List[Dict]:
        """
        执行遗传规划挖掘
        
        Args:
            symbols: 股票代码列表（None 则随机选择）
            n_symbols: 使用的股票数量
            start_date: 开始日期
            end_date: 结束日期
            forward_days: 预测未来收益天数
            top_n: 返回最佳因子数量
        
        Returns:
            最佳因子列表
        """
        if not GPLEARN_AVAILABLE:
            raise ImportError("请先安装 gplearn: pip install gplearn")
        
        print("=" * 70)
        print("遗传规划 Alpha 因子挖掘")
        print("=" * 70)
        
        # 选择股票（从统一股票池获取）
        if symbols is None:
            import random
            stock_pool = get_stock_pool('all')  # 使用全部股票池
            symbols = random.sample(stock_pool, min(n_symbols, len(stock_pool)))
        
        print(f"\n股票池: {len(symbols)} 只")
        print(f"进化参数: 种群={self.population_size}, 代数={self.generations}")
        print(f"预测目标: {forward_days} 日收益")
        
        # 获取数据
        print(f"\n获取数据...")
        data = self.fetch_data(symbols, start_date, end_date)
        
        if not data:
            print("无有效数据")
            return []
        
        # 合并所有股票数据进行训练
        print(f"\n准备训练数据...")
        all_X = []
        all_y = []
        all_prices = []
        
        for symbol, df in data.items():
            X, feature_names = self.prepare_features(df)
            prices = df['Close'].values
            
            # 未来收益作为目标
            y = np.zeros(len(prices))
            y[:-forward_days] = (
                (prices[forward_days:] - prices[:-forward_days]) / prices[:-forward_days]
            )
            y[-forward_days:] = np.nan
            
            # 过滤无效行
            valid = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
            
            all_X.append(X[valid])
            all_y.append(y[valid])
            all_prices.append(prices[valid])
        
        X_train = np.vstack(all_X)
        y_train = np.concatenate(all_y)
        prices_train = np.concatenate(all_prices)
        
        print(f"训练样本: {len(X_train)}")
        print(f"特征数量: {len(feature_names)}")
        print(f"特征列表: {feature_names}")
        
        # 创建适应度函数
        fitness = self._create_fitness(prices_train, forward_days)
        
        # 创建 GP 模型
        print(f"\n开始进化...")
        print("-" * 70)
        
        gp = SymbolicTransformer(
            population_size=self.population_size,
            generations=self.generations,
            tournament_size=self.tournament_size,
            stopping_criteria=self.stopping_criteria,
            p_crossover=self.p_crossover,
            p_subtree_mutation=self.p_subtree_mutation,
            p_hoist_mutation=self.p_hoist_mutation,
            p_point_mutation=self.p_point_mutation,
            max_samples=self.max_samples,
            parsimony_coefficient=self.parsimony_coefficient,
            init_depth=self.init_depth,
            function_set=self.function_set,
            feature_names=feature_names,
            metric=fitness,
            n_components=top_n,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        
        # 训练
        gp.fit(X_train, y_train)
        
        # 提取最佳因子
        print("\n" + "=" * 70)
        print(f"进化完成！提取 Top {top_n} 因子")
        print("=" * 70)
        
        best_factors = []
        
        for i, program in enumerate(gp._best_programs):
            if program is None:
                continue
            
            # 计算因子值
            factor_values = program.execute(X_train)
            
            # 评估
            score, metrics = evaluate_alpha(factor_values, prices_train, forward_days)
            
            factor_info = {
                'rank': i + 1,
                'formula': str(program),
                'fitness': program.fitness_,
                'length': program.length_,
                'depth': program.depth_,
                'ic': metrics['ic'],
                'icir': metrics['icir'],
                'sharpe': metrics['sharpe'],
                'turnover': metrics['turnover'],
                'score': score
            }
            
            best_factors.append(factor_info)
            
            print(f"\n[因子 #{i+1}]")
            print(f"  公式: {program}")
            print(f"  IC: {metrics['ic']:.4f} | ICIR: {metrics['icir']:.4f}")
            print(f"  Sharpe: {metrics['sharpe']:.4f} | 换手率: {metrics['turnover']:.4f}")
            print(f"  综合得分: {score:.4f}")
            print(f"  复杂度: 长度={program.length_}, 深度={program.depth_}")
        
        # 按得分排序
        best_factors.sort(key=lambda x: x['score'], reverse=True)
        
        self.best_programs = gp._best_programs
        
        return best_factors
    
    def backtest_factor(self, 
                        factor_formula: str,
                        symbol: str,
                        start_date: str = None,
                        end_date: str = None,
                        forward_days: int = 1) -> Dict:
        """
        回测单个因子
        
        Args:
            factor_formula: 因子公式字符串
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            forward_days: 预测天数
        
        Returns:
            回测结果
        """
        # 获取数据
        data = self.fetch_data([symbol], start_date, end_date)
        
        if symbol not in data:
            return {'error': f'无法获取 {symbol} 数据'}
        
        df = data[symbol]
        X, feature_names = self.prepare_features(df)
        prices = df['Close'].values
        
        # 找到对应的程序
        program = None
        for p in self.best_programs:
            if p is not None and str(p) == factor_formula:
                program = p
                break
        
        if program is None:
            return {'error': '未找到对应的因子程序'}
        
        # 计算因子值
        factor_values = program.execute(X)
        
        # 评估
        score, metrics = evaluate_alpha(factor_values, prices, forward_days)
        
        return {
            'symbol': symbol,
            'formula': factor_formula,
            'metrics': metrics,
            'factor_values': factor_values,
            'prices': prices,
            'dates': df.index.tolist()
        }
    
    def save(self, filepath: str):
        """保存模型"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'best_programs': self.best_programs,
                'evolution_history': self.evolution_history,
                'params': {
                    'population_size': self.population_size,
                    'generations': self.generations,
                }
            }, f)
        print(f"模型已保存至 {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.best_programs = data['best_programs']
        self.evolution_history = data.get('evolution_history', [])
        print(f"模型已加载: {len(self.best_programs)} 个因子")


def quick_mine(n_symbols: int = 20,
               population_size: int = 300,
               generations: int = 15,
               top_n: int = 5) -> List[Dict]:
    """
    快速挖掘入口
    
    Args:
        n_symbols: 使用股票数量
        population_size: 种群大小
        generations: 进化代数
        top_n: 返回因子数量
    
    Returns:
        最佳因子列表
    """
    miner = GPAlphaMiner(
        population_size=population_size,
        generations=generations,
        verbose=1
    )
    
    return miner.mine(n_symbols=n_symbols, top_n=top_n)


if __name__ == '__main__':
    # 快速测试
    factors = quick_mine(n_symbols=10, population_size=200, generations=10, top_n=5)
    
    print("\n" + "=" * 70)
    print("最终结果汇总")
    print("=" * 70)
    
    for f in factors:
        print(f"\n#{f['rank']}: IC={f['ic']:.4f}, Score={f['score']:.4f}")
        print(f"   {f['formula']}")
