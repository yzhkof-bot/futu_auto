"""
截面因子评估器

工业级因子评估指标:
1. 截面 IC / Rank IC - 每日因子值与未来收益的相关性
2. IC_IR - IC 的信息比率 (mean/std)
3. 分层回测 - Top/Bottom 分组收益差
4. 换手率分析
5. 因子衰减分析
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class FactorEvaluator:
    """
    截面因子评估器
    
    评估因子的预测能力和投资价值
    """
    
    def __init__(self, 
                 factor_panel: pd.DataFrame,
                 return_panel: pd.DataFrame,
                 forward_days: int = 1):
        """
        初始化
        
        Args:
            factor_panel: 因子面板 (日期 × 股票)
            return_panel: 未来收益面板 (日期 × 股票)，已对齐
            forward_days: 预测天数（用于标注）
        """
        self.factor = factor_panel
        self.returns = return_panel
        self.forward_days = forward_days
        
        # 确保索引对齐
        common_dates = self.factor.index.intersection(self.returns.index)
        common_symbols = self.factor.columns.intersection(self.returns.columns)
        
        self.factor = self.factor.loc[common_dates, common_symbols]
        self.returns = self.returns.loc[common_dates, common_symbols]
        
        self.dates = common_dates
        self.symbols = common_symbols
        
        # 缓存计算结果
        self._ic_series: Optional[pd.Series] = None
        self._rank_ic_series: Optional[pd.Series] = None
        self._group_returns: Optional[pd.DataFrame] = None
    
    # ============================================================
    # IC 相关指标
    # ============================================================
    
    def calc_ic_series(self, method: str = 'spearman') -> pd.Series:
        """
        计算每日截面 IC
        
        Args:
            method: 'spearman' (秩相关) 或 'pearson'
        
        Returns:
            IC 时间序列
        """
        ic_list = []
        
        for date in self.dates:
            f = self.factor.loc[date]
            r = self.returns.loc[date]
            
            # 过滤无效值
            valid = ~(f.isna() | r.isna())
            if valid.sum() < 20:
                ic_list.append(np.nan)
                continue
            
            f_valid = f[valid]
            r_valid = r[valid]
            
            try:
                if method == 'spearman':
                    ic, _ = stats.spearmanr(f_valid, r_valid)
                else:
                    ic, _ = stats.pearsonr(f_valid, r_valid)
                ic_list.append(ic)
            except:
                ic_list.append(np.nan)
        
        self._ic_series = pd.Series(ic_list, index=self.dates, name='IC')
        return self._ic_series
    
    def calc_rank_ic_series(self) -> pd.Series:
        """计算 Rank IC (Spearman)"""
        return self.calc_ic_series(method='spearman')
    
    def get_ic_stats(self) -> Dict[str, float]:
        """
        获取 IC 统计指标
        
        Returns:
            {
                'ic_mean': IC 均值,
                'ic_std': IC 标准差,
                'ic_ir': IC 信息比率 (IC_mean / IC_std),
                'ic_positive_ratio': IC > 0 的比例,
                'ic_abs_mean': |IC| 均值,
                't_stat': t 统计量,
                'p_value': p 值,
            }
        """
        if self._ic_series is None:
            self.calc_ic_series()
        
        ic = self._ic_series.dropna()
        
        if len(ic) < 10:
            return {
                'ic_mean': np.nan,
                'ic_std': np.nan,
                'ic_ir': np.nan,
                'ic_positive_ratio': np.nan,
                'ic_abs_mean': np.nan,
                't_stat': np.nan,
                'p_value': np.nan,
            }
        
        ic_mean = ic.mean()
        ic_std = ic.std()
        ic_ir = ic_mean / ic_std if ic_std > 1e-10 else 0
        
        # t 检验
        t_stat, p_value = stats.ttest_1samp(ic, 0)
        
        return {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'ic_positive_ratio': (ic > 0).mean(),
            'ic_abs_mean': ic.abs().mean(),
            't_stat': t_stat,
            'p_value': p_value,
        }
    
    # ============================================================
    # 分层回测
    # ============================================================
    
    def calc_group_returns(self, n_groups: int = 5) -> pd.DataFrame:
        """
        分层回测
        
        每天按因子值分成 n_groups 组，计算各组平均收益
        
        Args:
            n_groups: 分组数量
        
        Returns:
            DataFrame (日期 × 组别) 各组日收益
        """
        group_returns = pd.DataFrame(index=self.dates, 
                                     columns=[f'G{i+1}' for i in range(n_groups)])
        
        for date in self.dates:
            f = self.factor.loc[date]
            r = self.returns.loc[date]
            
            # 过滤无效值
            valid = ~(f.isna() | r.isna())
            if valid.sum() < n_groups * 5:
                continue
            
            f_valid = f[valid]
            r_valid = r[valid]
            
            # 按因子值分组
            try:
                groups = pd.qcut(f_valid, n_groups, labels=False, duplicates='drop')
                
                for g in range(n_groups):
                    mask = groups == g
                    if mask.sum() > 0:
                        group_returns.loc[date, f'G{g+1}'] = r_valid[mask].mean()
            except:
                continue
        
        self._group_returns = group_returns.astype(float)
        return self._group_returns
    
    def get_group_stats(self, n_groups: int = 5) -> Dict[str, float]:
        """
        获取分层统计
        
        Returns:
            {
                'long_short_return': 多空收益 (年化),
                'long_return': 多头收益 (年化),
                'short_return': 空头收益 (年化),
                'long_short_sharpe': 多空夏普,
                'monotonicity': 单调性得分,
            }
        """
        if self._group_returns is None:
            self.calc_group_returns(n_groups)
        
        gr = self._group_returns.dropna()
        
        if len(gr) < 20:
            return {
                'long_short_return': np.nan,
                'long_return': np.nan,
                'short_return': np.nan,
                'long_short_sharpe': np.nan,
                'monotonicity': np.nan,
            }
        
        # 多头 (最高组) 和空头 (最低组)
        long_ret = gr[f'G{n_groups}']
        short_ret = gr['G1']
        long_short_ret = long_ret - short_ret
        
        # 年化
        annual_factor = 252 / self.forward_days
        
        long_annual = long_ret.mean() * annual_factor
        short_annual = short_ret.mean() * annual_factor
        ls_annual = long_short_ret.mean() * annual_factor
        
        # 多空夏普
        ls_sharpe = (long_short_ret.mean() / long_short_ret.std() * 
                     np.sqrt(annual_factor)) if long_short_ret.std() > 1e-10 else 0
        
        # 单调性：各组平均收益的排名相关性
        group_means = gr.mean()
        expected_rank = np.arange(1, n_groups + 1)
        actual_rank = group_means.rank().values
        monotonicity, _ = stats.spearmanr(expected_rank, actual_rank)
        
        return {
            'long_short_return': ls_annual,
            'long_return': long_annual,
            'short_return': short_annual,
            'long_short_sharpe': ls_sharpe,
            'monotonicity': monotonicity,
        }
    
    def calc_cumulative_returns(self, n_groups: int = 5) -> pd.DataFrame:
        """
        计算各组累计收益
        
        Returns:
            DataFrame (日期 × 组别) 累计收益
        """
        if self._group_returns is None:
            self.calc_group_returns(n_groups)
        
        gr = self._group_returns.fillna(0)
        cumulative = (1 + gr).cumprod()
        
        # 添加多空组合
        long_short = gr[f'G{n_groups}'] - gr['G1']
        cumulative['Long-Short'] = (1 + long_short).cumprod()
        
        return cumulative
    
    # ============================================================
    # 换手率分析
    # ============================================================
    
    def calc_turnover(self, top_pct: float = 0.2) -> pd.Series:
        """
        计算换手率
        
        每天选取因子值最高的 top_pct 股票，计算相邻两天的换手比例
        
        Args:
            top_pct: 选股比例
        
        Returns:
            换手率时间序列
        """
        turnover_list = []
        prev_selected = None
        
        for date in self.dates:
            f = self.factor.loc[date].dropna()
            
            if len(f) < 10:
                turnover_list.append(np.nan)
                continue
            
            # 选取 top_pct 股票
            n_select = max(1, int(len(f) * top_pct))
            selected = set(f.nlargest(n_select).index)
            
            if prev_selected is not None:
                # 换手率 = 变化的股票数 / 总选股数
                changed = len(selected.symmetric_difference(prev_selected))
                turnover = changed / (2 * n_select)
                turnover_list.append(turnover)
            else:
                turnover_list.append(np.nan)
            
            prev_selected = selected
        
        return pd.Series(turnover_list, index=self.dates, name='Turnover')
    
    def get_turnover_stats(self, top_pct: float = 0.2) -> Dict[str, float]:
        """
        获取换手率统计
        
        Returns:
            {
                'turnover_mean': 平均换手率,
                'turnover_std': 换手率标准差,
                'annual_turnover': 年化换手率,
            }
        """
        turnover = self.calc_turnover(top_pct).dropna()
        
        if len(turnover) < 10:
            return {
                'turnover_mean': np.nan,
                'turnover_std': np.nan,
                'annual_turnover': np.nan,
            }
        
        return {
            'turnover_mean': turnover.mean(),
            'turnover_std': turnover.std(),
            'annual_turnover': turnover.mean() * 252 / self.forward_days,
        }
    
    # ============================================================
    # 因子衰减分析
    # ============================================================
    
    def calc_ic_decay(self, max_lag: int = 20) -> pd.Series:
        """
        计算 IC 衰减
        
        因子值与不同滞后期收益的 IC
        
        Args:
            max_lag: 最大滞后期
        
        Returns:
            IC 衰减序列 (lag -> IC)
        """
        ic_decay = []
        
        for lag in range(1, max_lag + 1):
            # 计算 lag 日后的收益
            lagged_returns = self.returns.shift(-lag + self.forward_days)
            
            # 计算 IC
            ic_list = []
            for date in self.dates[:-lag]:
                f = self.factor.loc[date]
                r = lagged_returns.loc[date]
                
                valid = ~(f.isna() | r.isna())
                if valid.sum() < 20:
                    continue
                
                try:
                    ic, _ = stats.spearmanr(f[valid], r[valid])
                    ic_list.append(ic)
                except:
                    continue
            
            ic_decay.append(np.mean(ic_list) if ic_list else np.nan)
        
        return pd.Series(ic_decay, index=range(1, max_lag + 1), name='IC_Decay')
    
    def get_decay_stats(self, max_lag: int = 20) -> Dict[str, float]:
        """
        获取衰减统计
        
        Returns:
            {
                'half_life': IC 衰减到一半的天数,
                'decay_rate': 衰减速率,
            }
        """
        ic_decay = self.calc_ic_decay(max_lag)
        
        if ic_decay.isna().all():
            return {'half_life': np.nan, 'decay_rate': np.nan}
        
        ic_decay = ic_decay.dropna()
        initial_ic = ic_decay.iloc[0]
        
        if abs(initial_ic) < 0.01:
            return {'half_life': np.nan, 'decay_rate': np.nan}
        
        # 找到 IC 衰减到一半的位置
        half_ic = initial_ic / 2
        half_life = np.nan
        
        for lag, ic in ic_decay.items():
            if abs(ic) <= abs(half_ic):
                half_life = lag
                break
        
        # 衰减速率（指数拟合）
        try:
            log_ic = np.log(np.abs(ic_decay.values) + 1e-10)
            lags = np.arange(1, len(ic_decay) + 1)
            slope, _ = np.polyfit(lags, log_ic, 1)
            decay_rate = -slope
        except:
            decay_rate = np.nan
        
        return {
            'half_life': half_life,
            'decay_rate': decay_rate,
        }
    
    # ============================================================
    # 综合评估
    # ============================================================
    
    def evaluate(self, n_groups: int = 5, verbose: bool = True) -> Dict[str, float]:
        """
        综合评估因子
        
        Args:
            n_groups: 分组数量
            verbose: 是否打印结果
        
        Returns:
            所有评估指标的字典
        """
        # 计算各项指标
        ic_stats = self.get_ic_stats()
        group_stats = self.get_group_stats(n_groups)
        turnover_stats = self.get_turnover_stats()
        decay_stats = self.get_decay_stats()
        
        # 合并结果
        result = {
            **ic_stats,
            **group_stats,
            **turnover_stats,
            **decay_stats,
        }
        
        # 计算综合得分
        score = self._calc_composite_score(result)
        result['composite_score'] = score
        
        if verbose:
            self._print_report(result, n_groups)
        
        return result
    
    def _calc_composite_score(self, metrics: Dict[str, float]) -> float:
        """
        计算综合得分
        
        权重:
        - IC_IR: 40%
        - 多空夏普: 30%
        - 单调性: 20%
        - 换手率（负向）: 10%
        """
        ic_ir = metrics.get('ic_ir', 0) or 0
        ls_sharpe = metrics.get('long_short_sharpe', 0) or 0
        monotonicity = metrics.get('monotonicity', 0) or 0
        turnover = metrics.get('turnover_mean', 1) or 1
        
        # 归一化
        ic_ir_score = min(abs(ic_ir) / 0.5, 1.0)  # ICIR > 0.5 满分
        sharpe_score = min(abs(ls_sharpe) / 2.0, 1.0)  # Sharpe > 2 满分
        mono_score = (monotonicity + 1) / 2  # [-1, 1] -> [0, 1]
        turnover_score = max(0, 1 - turnover)  # 换手率越低越好
        
        score = (
            0.40 * ic_ir_score +
            0.30 * sharpe_score +
            0.20 * mono_score +
            0.10 * turnover_score
        )
        
        return score
    
    def _print_report(self, metrics: Dict[str, float], n_groups: int):
        """打印评估报告"""
        print("\n" + "=" * 60)
        print("因子评估报告")
        print("=" * 60)
        
        print(f"\n【IC 分析】")
        print(f"  IC 均值:       {metrics['ic_mean']:.4f}")
        print(f"  IC 标准差:     {metrics['ic_std']:.4f}")
        print(f"  IC_IR:         {metrics['ic_ir']:.4f}")
        print(f"  IC > 0 比例:   {metrics['ic_positive_ratio']:.2%}")
        print(f"  t 统计量:      {metrics['t_stat']:.2f} (p={metrics['p_value']:.4f})")
        
        print(f"\n【分层回测】({n_groups} 组)")
        print(f"  多头收益(年化):   {metrics['long_return']:.2%}")
        print(f"  空头收益(年化):   {metrics['short_return']:.2%}")
        print(f"  多空收益(年化):   {metrics['long_short_return']:.2%}")
        print(f"  多空夏普:         {metrics['long_short_sharpe']:.2f}")
        print(f"  单调性:           {metrics['monotonicity']:.2f}")
        
        print(f"\n【换手率】")
        print(f"  日均换手率:    {metrics['turnover_mean']:.2%}")
        print(f"  年化换手率:    {metrics['annual_turnover']:.1f}x")
        
        print(f"\n【因子衰减】")
        print(f"  半衰期:        {metrics['half_life']} 天")
        print(f"  衰减速率:      {metrics['decay_rate']:.4f}")
        
        print(f"\n【综合得分】")
        print(f"  {metrics['composite_score']:.4f}")
        print("=" * 60)


def quick_evaluate(factor_panel: pd.DataFrame,
                   return_panel: pd.DataFrame,
                   forward_days: int = 1,
                   verbose: bool = True) -> Dict[str, float]:
    """
    快速评估因子
    
    Args:
        factor_panel: 因子面板
        return_panel: 未来收益面板
        forward_days: 预测天数
        verbose: 是否打印
    
    Returns:
        评估指标字典
    """
    evaluator = FactorEvaluator(factor_panel, return_panel, forward_days)
    return evaluator.evaluate(verbose=verbose)


if __name__ == '__main__':
    # 测试
    from data_manager import PanelDataManager
    from factor_engine import alpha_momentum, preprocess_factor
    
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
    result = quick_evaluate(factor, forward_return, forward_days=5)
