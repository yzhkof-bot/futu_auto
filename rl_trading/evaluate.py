#!/usr/bin/env python3
"""
评估脚本 - 评估训练好的模型并生成报告
"""

import argparse
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rl_trading.config import EnvConfig, PPOConfig, FeatureConfig
from rl_trading.env import MultiStockEnv
from rl_trading.agent import PPOAgent
from rl_trading.features import FeatureEngineer
from src.data.data_fetcher import DataFetcher


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(
        self,
        model_path: str,
        env_config: Optional[EnvConfig] = None,
        ppo_config: Optional[PPOConfig] = None
    ):
        """
        初始化评估器
        
        Args:
            model_path: 模型文件路径
            env_config: 环境配置
            ppo_config: PPO 配置
        """
        self.model_path = Path(model_path)
        self.env_config = env_config or EnvConfig()
        self.ppo_config = ppo_config or PPOConfig()
        
        self.data_fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer()
        
        self.env: Optional[MultiStockEnv] = None
        self.agent: Optional[PPOAgent] = None
        
        # 评估结果
        self.results: Dict = {}
    
    def load_model(self) -> None:
        """加载模型"""
        print(f"加载模型: {self.model_path}")
        
        # 需要先创建环境来获取状态/动作维度
        if self.env is None:
            self.prepare_environment()
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.ppo_config.hidden_dims,
            device=self.ppo_config.device
        )
        
        self.agent.load(str(self.model_path))
        self.agent.set_training_mode(False)
    
    def prepare_environment(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> None:
        """准备评估环境"""
        start_date = start_date or self.env_config.start_date
        end_date = end_date or self.env_config.end_date
        
        print(f"准备评估数据: {start_date} ~ {end_date}")
        
        # 获取数据
        all_data = {}
        for symbol in self.env_config.symbols:
            try:
                df = self.data_fetcher.fetch_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=True
                )
                if len(df) > 0:
                    all_data[symbol] = df
            except Exception as e:
                print(f"  {symbol} 获取失败: {e}")
        
        # 计算特征
        processed_data = {}
        for symbol, df in all_data.items():
            processed_df = self.feature_engineer.compute_features(df, fit_scaler=True)
            processed_data[symbol] = processed_df
        
        # 对齐数据
        min_length = min(len(df) for df in processed_data.values())
        for symbol in processed_data:
            processed_data[symbol] = processed_data[symbol].iloc[-min_length:].reset_index(drop=True)
        
        # 创建环境
        feature_columns = self.feature_engineer.get_feature_importance_columns()
        
        self.env = MultiStockEnv(
            dfs=processed_data,
            feature_columns=feature_columns,
            initial_balance=self.env_config.initial_balance,
            transaction_cost=self.env_config.transaction_cost,
            slippage=self.env_config.slippage,
            max_position_per_stock=self.env_config.max_position_per_stock
        )
        
        self.processed_data = processed_data
    
    def evaluate(self, n_episodes: int = 1) -> Dict:
        """
        评估模型
        
        Args:
            n_episodes: 评估回合数
            
        Returns:
            评估结果
        """
        if self.agent is None:
            self.load_model()
        
        print(f"\n开始评估 ({n_episodes} 回合)...")
        
        all_portfolio_values = []
        all_positions_history = []
        all_actions_history = []
        all_stats = []
        
        for ep in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            
            portfolio_values = [self.env_config.initial_balance]
            positions_history = []
            actions_history = []
            
            while not done:
                action, _, _ = self.agent.select_action(state, deterministic=True)
                state, _, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                portfolio_values.append(info['portfolio_value'])
                positions_history.append(info['positions'].copy())
                actions_history.append(action.copy())
            
            stats = self.env.get_episode_stats()
            all_stats.append(stats)
            all_portfolio_values.append(portfolio_values)
            all_positions_history.append(positions_history)
            all_actions_history.append(actions_history)
            
            print(f"  Episode {ep+1}: Return={stats['total_return']:.2%}, "
                  f"Sharpe={stats['sharpe_ratio']:.2f}, MaxDD={stats['max_drawdown']:.2%}")
        
        # 汇总结果
        self.results = {
            'n_episodes': n_episodes,
            'portfolio_values': all_portfolio_values,
            'positions_history': all_positions_history,
            'actions_history': all_actions_history,
            'stats': all_stats,
            'avg_return': np.mean([s['total_return'] for s in all_stats]),
            'avg_sharpe': np.mean([s['sharpe_ratio'] for s in all_stats]),
            'avg_max_dd': np.mean([s['max_drawdown'] for s in all_stats]),
            'final_value': np.mean([s['final_value'] for s in all_stats])
        }
        
        return self.results
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """绘制评估结果"""
        if not self.results:
            print("请先运行 evaluate()")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 资产曲线
        ax1 = axes[0, 0]
        for i, values in enumerate(self.results['portfolio_values']):
            ax1.plot(values, alpha=0.7, label=f'Episode {i+1}')
        ax1.axhline(y=self.env_config.initial_balance, color='gray', linestyle='--', label='Initial')
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 收益分布
        ax2 = axes[0, 1]
        returns = [s['total_return'] for s in self.results['stats']]
        ax2.hist(returns, bins=20, edgecolor='black', alpha=0.7)
        ax2.axvline(x=np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.2%}')
        ax2.set_title('Return Distribution')
        ax2.set_xlabel('Total Return')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 仓位分布（最后一个 episode）
        ax3 = axes[1, 0]
        if self.results['positions_history']:
            positions_df = pd.DataFrame(self.results['positions_history'][-1])
            positions_df.plot(ax=ax3, alpha=0.7)
            ax3.set_title('Position Allocation (Last Episode)')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Position Ratio')
            ax3.legend(loc='upper right', fontsize=8)
            ax3.grid(True, alpha=0.3)
        
        # 4. 性能指标
        ax4 = axes[1, 1]
        metrics = ['Total Return', 'Sharpe Ratio', 'Max Drawdown']
        values = [
            self.results['avg_return'] * 100,
            self.results['avg_sharpe'],
            self.results['avg_max_dd'] * 100
        ]
        colors = ['green' if v > 0 else 'red' for v in values]
        colors[2] = 'red'  # Max DD 总是负面的
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_title('Performance Metrics')
        ax4.set_ylabel('Value')
        ax4.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.annotate(f'{val:.2f}%' if 'Return' in metrics[bars.index(bar)] or 'Drawdown' in metrics[bars.index(bar)] else f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存到 {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """生成评估报告"""
        if not self.results:
            return "请先运行 evaluate()"
        
        report = f"""
================================================================================
                         RL 交易策略评估报告
================================================================================

评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
模型路径: {self.model_path}

--------------------------------------------------------------------------------
                              基本配置
--------------------------------------------------------------------------------
股票池: {', '.join(self.env_config.symbols)}
初始资金: ${self.env_config.initial_balance:,.0f}
单股最大仓位: {self.env_config.max_position_per_stock:.0%}
交易成本: {self.env_config.transaction_cost:.2%}
滑点: {self.env_config.slippage:.2%}

--------------------------------------------------------------------------------
                              评估结果
--------------------------------------------------------------------------------
评估回合数: {self.results['n_episodes']}

【收益指标】
  平均总收益率: {self.results['avg_return']:.2%}
  最终资产均值: ${self.results['final_value']:,.0f}
  收益金额: ${self.results['final_value'] - self.env_config.initial_balance:,.0f}

【风险指标】
  平均夏普比率: {self.results['avg_sharpe']:.2f}
  平均最大回撤: {self.results['avg_max_dd']:.2%}

【各回合详情】
"""
        for i, stats in enumerate(self.results['stats']):
            report += f"""
  Episode {i+1}:
    总收益率: {stats['total_return']:.2%}
    夏普比率: {stats['sharpe_ratio']:.2f}
    最大回撤: {stats['max_drawdown']:.2%}
    最终资产: ${stats['final_value']:,.0f}
"""
        
        report += """
================================================================================
"""
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"报告已保存到 {save_path}")
        
        return report
    
    def compare_with_benchmark(
        self,
        benchmark: str = 'SPY'
    ) -> Dict:
        """
        与基准进行比较
        
        Args:
            benchmark: 基准代码
            
        Returns:
            比较结果
        """
        if not self.results:
            print("请先运行 evaluate()")
            return {}
        
        # 获取基准数据
        benchmark_data = self.data_fetcher.fetch_stock_data(
            symbol=benchmark,
            start_date=self.env_config.start_date,
            end_date=self.env_config.end_date
        )
        
        # 计算基准收益
        benchmark_return = (benchmark_data['Close'].iloc[-1] / benchmark_data['Close'].iloc[0]) - 1
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        benchmark_sharpe = benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252)
        
        # 计算基准最大回撤
        cumulative = (1 + benchmark_returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        benchmark_max_dd = drawdown.min()
        
        comparison = {
            'strategy_return': self.results['avg_return'],
            'benchmark_return': benchmark_return,
            'strategy_sharpe': self.results['avg_sharpe'],
            'benchmark_sharpe': benchmark_sharpe,
            'strategy_max_dd': self.results['avg_max_dd'],
            'benchmark_max_dd': benchmark_max_dd,
            'alpha': self.results['avg_return'] - benchmark_return,
            'information_ratio': (self.results['avg_return'] - benchmark_return) / np.std([s['total_return'] for s in self.results['stats']])
        }
        
        print("\n" + "=" * 50)
        print(f"策略 vs {benchmark} 基准比较")
        print("=" * 50)
        print(f"{'指标':<20} {'策略':>12} {benchmark:>12}")
        print("-" * 50)
        print(f"{'总收益率':<20} {comparison['strategy_return']:>11.2%} {comparison['benchmark_return']:>11.2%}")
        print(f"{'夏普比率':<20} {comparison['strategy_sharpe']:>12.2f} {comparison['benchmark_sharpe']:>12.2f}")
        print(f"{'最大回撤':<20} {comparison['strategy_max_dd']:>11.2%} {comparison['benchmark_max_dd']:>11.2%}")
        print("-" * 50)
        print(f"{'Alpha':<20} {comparison['alpha']:>11.2%}")
        print(f"{'信息比率':<20} {comparison['information_ratio']:>12.2f}")
        
        return comparison


def parse_args():
    parser = argparse.ArgumentParser(description='评估 RL 交易策略')
    
    parser.add_argument('--model', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--n-episodes', type=int, default=1,
                       help='评估回合数')
    parser.add_argument('--start-date', type=str, default=None,
                       help='评估数据开始日期')
    parser.add_argument('--end-date', type=str, default=None,
                       help='评估数据结束日期')
    parser.add_argument('--benchmark', type=str, default='SPY',
                       help='基准代码')
    parser.add_argument('--output-dir', type=str, default='rl_trading/reports',
                       help='输出目录')
    parser.add_argument('--no-plot', action='store_true',
                       help='不显示图表')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建评估器
    evaluator = ModelEvaluator(model_path=args.model)
    
    # 准备环境
    if args.start_date or args.end_date:
        evaluator.prepare_environment(
            start_date=args.start_date,
            end_date=args.end_date
        )
    
    # 评估
    results = evaluator.evaluate(n_episodes=args.n_episodes)
    
    # 生成报告
    report = evaluator.generate_report(
        save_path=str(output_dir / 'evaluation_report.txt')
    )
    print(report)
    
    # 与基准比较
    evaluator.compare_with_benchmark(benchmark=args.benchmark)
    
    # 绘制图表
    if not args.no_plot:
        evaluator.plot_results(
            save_path=str(output_dir / 'evaluation_plots.png')
        )


if __name__ == '__main__':
    main()
