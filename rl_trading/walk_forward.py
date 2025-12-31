"""
Walk-Forward 验证训练器

滚动窗口训练：
- 用 N 年数据训练
- 用下一年数据测试
- 窗口向后滑动，重复

这样可以模拟真实的"未知未来"场景，避免过拟合。
"""

import os
import sys
import time
import json
import functools
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

# 强制刷新输出
print = functools.partial(print, flush=True)

from .config import EnvConfig, PPOConfig, TrainConfig, FeatureConfig
from .env import SingleStockEnv
from .agent import PPOAgent
from .features import FeatureEngineer

# 导入数据获取模块
sys.path.append(str(Path(__file__).parent.parent))
from src.data.data_fetcher import DataFetcher


@dataclass
class WalkForwardConfig:
    """Walk-Forward 配置"""
    # 股票
    symbols: List[str] = None
    
    # 时间窗口
    train_years: int = 3          # 训练窗口（年）
    test_years: int = 1           # 测试窗口（年）
    start_year: int = 2015        # 起始年份
    end_year: int = 2024          # 结束年份
    
    # 训练参数
    timesteps_per_fold: int = 200_000  # 每个 fold 的训练步数
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['NVDA']


class WalkForwardTrainer:
    """
    Walk-Forward 验证训练器
    
    滚动窗口训练，模拟真实的时间序列预测场景
    """
    
    def __init__(
        self,
        wf_config: Optional[WalkForwardConfig] = None,
        ppo_config: Optional[PPOConfig] = None,
        feature_config: Optional[FeatureConfig] = None
    ):
        self.wf_config = wf_config or WalkForwardConfig()
        self.ppo_config = ppo_config or PPOConfig()
        self.feature_config = feature_config or FeatureConfig()
        
        # 初始化组件
        self.data_fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer(
            normalize=self.feature_config.normalize_features,
            clip_value=self.feature_config.clip_features
        )
        
        # 结果记录
        self.fold_results = []
        
        # 创建目录
        self.output_dir = Path("rl_trading/walk_forward_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        self._set_seed(42)
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        import torch
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def _fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """获取所有股票的完整数据"""
        print("=" * 60)
        print("获取数据...")
        print("=" * 60)
        
        all_data = {}
        start_date = f"{self.wf_config.start_year - 1}-01-01"  # 多取一年用于特征计算
        end_date = f"{self.wf_config.end_year + 1}-01-01"
        
        for symbol in self.wf_config.symbols:
            print(f"\n获取 {symbol} 数据...")
            try:
                df = self.data_fetcher.fetch_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=True
                )
                
                if len(df) > 0:
                    # 保存日期信息（在特征计算前）
                    if isinstance(df.index, pd.DatetimeIndex):
                        dates = df.index.to_series().reset_index(drop=True)
                        df = df.reset_index(drop=True)  # 删除日期索引，避免被归一化
                    elif 'Date' in df.columns:
                        dates = df['Date'].copy()
                        df = df.drop(columns=['Date'])
                    else:
                        dates = None
                    
                    # 计算特征（不包含 Date 列）
                    processed_df = self.feature_engineer.compute_features(
                        df, fit_scaler=(symbol == self.wf_config.symbols[0])
                    )
                    
                    # 恢复 Date 列（对齐长度，因为特征计算会删除 warmup 行）
                    if dates is not None:
                        # 特征计算删除了前 60 行（warmup_period）
                        warmup = len(dates) - len(processed_df)
                        processed_df['Date'] = dates.iloc[warmup:].reset_index(drop=True)
                    
                    all_data[symbol] = processed_df
                    print(f"  {symbol}: {len(processed_df)} 条数据")
                    
            except Exception as e:
                print(f"  {symbol} 获取失败: {e}")
                import traceback
                traceback.print_exc()
        
        return all_data
    
    def _split_by_year(
        self, 
        data: Dict[str, pd.DataFrame], 
        train_start_year: int,
        train_end_year: int,
        test_end_year: int
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """按年份划分训练集和测试集"""
        train_data = {}
        test_data = {}
        
        for symbol, df in data.items():
            df = df.copy()
            
            # 从 Date 列获取年份
            if 'Date' in df.columns:
                df['_year'] = pd.to_datetime(df['Date']).dt.year
            elif isinstance(df.index, pd.DatetimeIndex):
                df['_year'] = df.index.year
            else:
                raise ValueError(f"{symbol} 没有日期信息，columns: {df.columns.tolist()[:10]}")
            
            # 划分
            train_mask = (df['_year'] >= train_start_year) & (df['_year'] < train_end_year)
            test_mask = (df['_year'] >= train_end_year) & (df['_year'] < test_end_year)
            
            train_df = df[train_mask].drop(columns=['_year']).reset_index(drop=True)
            test_df = df[test_mask].drop(columns=['_year']).reset_index(drop=True)
            
            if len(train_df) > 0 and len(test_df) > 0:
                train_data[symbol] = train_df
                test_data[symbol] = test_df
        
        return train_data, test_data
    
    def _create_env(
        self, 
        data: Dict[str, pd.DataFrame], 
        training_mode: bool = True
    ) -> SingleStockEnv:
        """创建环境"""
        feature_columns = self.feature_engineer.get_core_features()
        
        env = SingleStockEnv(
            dfs=data,
            feature_columns=feature_columns,
            initial_balance=1_000_000.0,
            transaction_cost=0.001,
            slippage=0.0005,
            max_position=1.0,
            lookback_window=60,
            training_mode=training_mode,
            reward_scaling=100.0,
            trade_penalty=0.1,
            inactivity_penalty=0.005,
            use_lstm=self.ppo_config.use_lstm,
            episode_length=500,
            discrete_action=self.ppo_config.discrete_action
        )
        
        return env
    
    def _create_agent(self, env: SingleStockEnv) -> PPOAgent:
        """创建 Agent"""
        if self.ppo_config.use_lstm:
            state_dim = env.observation_space.shape[1]
        else:
            state_dim = env.observation_space.shape[0]
        
        action_dim = self.ppo_config.action_dim if self.ppo_config.discrete_action else env.action_space.shape[0]
        
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.ppo_config.hidden_dims,
            critic_hidden_dims=self.ppo_config.critic_hidden_dims,
            learning_rate=self.ppo_config.learning_rate,
            gamma=self.ppo_config.gamma,
            gae_lambda=self.ppo_config.gae_lambda,
            clip_epsilon=self.ppo_config.clip_epsilon,
            value_coef=self.ppo_config.value_coef,
            entropy_coef=self.ppo_config.entropy_coef,
            max_grad_norm=self.ppo_config.max_grad_norm,
            n_epochs=self.ppo_config.n_epochs,
            batch_size=self.ppo_config.batch_size,
            use_lstm=self.ppo_config.use_lstm,
            lstm_hidden_size=self.ppo_config.lstm_hidden_size,
            num_lstm_layers=self.ppo_config.num_lstm_layers,
            device=self.ppo_config.device,
            lr_schedule='cosine',
            min_lr=1e-6,
            total_timesteps=self.wf_config.timesteps_per_fold,
            discrete_action=self.ppo_config.discrete_action
        )
        
        return agent
    
    def _train_fold(
        self, 
        train_env: SingleStockEnv, 
        agent: PPOAgent,
        fold_name: str
    ) -> Dict:
        """训练单个 fold"""
        print(f"\n训练 {fold_name}...")
        
        total_timesteps = self.wf_config.timesteps_per_fold
        n_steps = self.ppo_config.n_steps
        
        timesteps = 0
        episode = 0
        best_reward = -np.inf
        
        state, _ = train_env.reset()
        episode_reward = 0
        
        while timesteps < total_timesteps:
            # 收集 n_steps 步经验
            for _ in range(n_steps):
                action, log_prob, value = agent.select_action(state)
                next_state, reward, terminated, truncated, info = train_env.step(action)
                done = terminated or truncated
                
                agent.store_transition(state, action, reward, done, log_prob, value)
                
                state = next_state
                episode_reward += reward
                timesteps += 1
                
                if done:
                    episode += 1
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                    episode_reward = 0
                    state, _ = train_env.reset()
                
                if timesteps >= total_timesteps:
                    break
            
            # 更新策略
            _, _, last_value = agent.select_action(state)
            agent.update(last_value, n_steps=n_steps, last_done=done)
            
            # 打印进度
            if timesteps % 50000 < n_steps:
                print(f"  {fold_name}: {timesteps}/{total_timesteps} ({100*timesteps/total_timesteps:.0f}%)")
        
        return {'best_reward': best_reward, 'episodes': episode}
    
    def _evaluate_fold(
        self, 
        test_env: SingleStockEnv, 
        agent: PPOAgent,
        fold_name: str
    ) -> Dict:
        """评估单个 fold"""
        print(f"评估 {fold_name}...")
        
        state, _ = test_env.reset()
        done = False
        total_trades = 0
        
        while not done:
            action, _, _ = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
        
        stats = test_env.get_episode_stats()
        
        # 计算基准收益
        prices = test_env.current_prices
        start_idx = test_env.lookback_window
        end_idx = test_env.current_step
        if end_idx > start_idx:
            benchmark_return = (prices[end_idx] - prices[start_idx]) / prices[start_idx]
        else:
            benchmark_return = 0.0
        
        return {
            'total_return': stats['total_return'],
            'benchmark_return': benchmark_return,
            'excess_return': stats['total_return'] - benchmark_return,
            'sharpe_ratio': stats['sharpe_ratio'],
            'max_drawdown': stats['max_drawdown'],
            'total_trades': stats['total_trades']
        }
    
    def run(self) -> pd.DataFrame:
        """
        运行 Walk-Forward 验证
        
        Returns:
            包含所有 fold 结果的 DataFrame
        """
        print("=" * 60)
        print("Walk-Forward 验证")
        print("=" * 60)
        print(f"股票: {self.wf_config.symbols}")
        print(f"训练窗口: {self.wf_config.train_years} 年")
        print(f"测试窗口: {self.wf_config.test_years} 年")
        print(f"时间范围: {self.wf_config.start_year} - {self.wf_config.end_year}")
        print(f"每 fold 训练步数: {self.wf_config.timesteps_per_fold:,}")
        
        # 获取所有数据
        all_data = self._fetch_all_data()
        
        if not all_data:
            raise ValueError("无法获取数据")
        
        # 计算 fold 数量
        train_years = self.wf_config.train_years
        test_years = self.wf_config.test_years
        start_year = self.wf_config.start_year
        end_year = self.wf_config.end_year
        
        # 生成所有 fold
        folds = []
        current_train_start = start_year
        
        while current_train_start + train_years + test_years <= end_year + 1:
            train_end = current_train_start + train_years
            test_end = train_end + test_years
            
            folds.append({
                'train_start': current_train_start,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end
            })
            
            current_train_start += test_years  # 滑动一个测试窗口
        
        print(f"\n共 {len(folds)} 个 fold:")
        for i, fold in enumerate(folds):
            print(f"  Fold {i+1}: 训练 {fold['train_start']}-{fold['train_end']-1}, "
                  f"测试 {fold['test_start']}-{fold['test_end']-1}")
        
        # 运行每个 fold
        results = []
        
        for i, fold in enumerate(folds):
            fold_name = f"Fold {i+1}/{len(folds)}"
            print(f"\n{'='*60}")
            print(f"{fold_name}: 训练 {fold['train_start']}-{fold['train_end']-1}, "
                  f"测试 {fold['test_start']}-{fold['test_end']-1}")
            print("=" * 60)
            
            # 划分数据
            train_data, test_data = self._split_by_year(
                all_data,
                fold['train_start'],
                fold['train_end'],
                fold['test_end']
            )
            
            if not train_data or not test_data:
                print(f"  跳过：数据不足")
                continue
            
            print(f"  训练数据: {sum(len(df) for df in train_data.values())} 条")
            print(f"  测试数据: {sum(len(df) for df in test_data.values())} 条")
            
            # 创建环境
            train_env = self._create_env(train_data, training_mode=True)
            test_env = self._create_env(test_data, training_mode=False)
            
            # 创建新 Agent（每个 fold 从头训练）
            agent = self._create_agent(train_env)
            
            # 训练
            train_stats = self._train_fold(train_env, agent, fold_name)
            
            # 评估
            eval_stats = self._evaluate_fold(test_env, agent, fold_name)
            
            # 保存模型
            model_path = self.output_dir / f"model_fold_{i+1}.pt"
            agent.save(str(model_path))
            
            # 记录结果
            result = {
                'fold': i + 1,
                'train_period': f"{fold['train_start']}-{fold['train_end']-1}",
                'test_period': f"{fold['test_start']}-{fold['test_end']-1}",
                **eval_stats
            }
            results.append(result)
            
            # 打印结果
            print(f"\n{fold_name} 测试结果:")
            print(f"  策略收益: {eval_stats['total_return']*100:.2f}%")
            print(f"  基准收益: {eval_stats['benchmark_return']*100:.2f}%")
            print(f"  超额收益: {eval_stats['excess_return']*100:.2f}%")
            print(f"  夏普比率: {eval_stats['sharpe_ratio']:.2f}")
            print(f"  最大回撤: {eval_stats['max_drawdown']*100:.2f}%")
            print(f"  交易次数: {eval_stats['total_trades']}")
            
            # 清理
            train_env.close()
            test_env.close()
        
        # 汇总结果
        results_df = pd.DataFrame(results)
        
        print("\n" + "=" * 60)
        print("Walk-Forward 验证汇总")
        print("=" * 60)
        print(results_df.to_string(index=False))
        
        # 计算平均指标
        print(f"\n平均指标:")
        print(f"  平均策略收益: {results_df['total_return'].mean()*100:.2f}%")
        print(f"  平均基准收益: {results_df['benchmark_return'].mean()*100:.2f}%")
        print(f"  平均超额收益: {results_df['excess_return'].mean()*100:.2f}%")
        print(f"  平均夏普比率: {results_df['sharpe_ratio'].mean():.2f}")
        print(f"  跑赢基准次数: {(results_df['excess_return'] > 0).sum()}/{len(results_df)}")
        
        # 保存结果
        results_path = self.output_dir / "walk_forward_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n结果已保存到: {results_path}")
        
        return results_df


def main():
    """主函数"""
    # 配置
    wf_config = WalkForwardConfig(
        symbols=['NVDA'],           # 可以改成 MAG7
        train_years=3,              # 用 3 年数据训练
        test_years=1,               # 用 1 年数据测试
        start_year=2015,            # 从 2015 年开始
        end_year=2024,              # 到 2024 年结束
        timesteps_per_fold=200_000  # 每个 fold 训练 20 万步
    )
    
    ppo_config = PPOConfig()
    feature_config = FeatureConfig()
    
    # 运行
    trainer = WalkForwardTrainer(
        wf_config=wf_config,
        ppo_config=ppo_config,
        feature_config=feature_config
    )
    
    results = trainer.run()
    
    return results


if __name__ == "__main__":
    main()
