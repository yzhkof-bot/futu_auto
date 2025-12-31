"""
训练器 - 管理整个训练流程
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

# 强制刷新输出
print = functools.partial(print, flush=True)

from .config import EnvConfig, PPOConfig, TrainConfig, FeatureConfig
from .env import TradingEnv, MultiStockEnv, SingleStockEnv
from .agent import PPOAgent
from .features import FeatureEngineer
from .visualizer import TrainingVisualizer

# 导入数据获取模块
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.data.data_fetcher import DataFetcher


class Trainer:
    """
    RL 交易策略训练器
    """
    
    def __init__(
        self,
        env_config: Optional[EnvConfig] = None,
        ppo_config: Optional[PPOConfig] = None,
        train_config: Optional[TrainConfig] = None,
        feature_config: Optional[FeatureConfig] = None
    ):
        """
        初始化训练器
        
        Args:
            env_config: 环境配置
            ppo_config: PPO 配置
            train_config: 训练配置
            feature_config: 特征配置
        """
        self.env_config = env_config or EnvConfig()
        self.ppo_config = ppo_config or PPOConfig()
        self.train_config = train_config or TrainConfig()
        self.feature_config = feature_config or FeatureConfig()
        
        # 设置随机种子
        self._set_seed(self.train_config.seed)
        
        # 初始化组件
        self.data_fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer(
            normalize=self.feature_config.normalize_features,
            clip_value=self.feature_config.clip_features
        )
        
        # 数据和环境
        self.train_data: Dict[str, pd.DataFrame] = {}
        self.test_data: Dict[str, pd.DataFrame] = {}
        self.train_env: Optional[MultiStockEnv] = None
        self.test_env: Optional[MultiStockEnv] = None
        
        # Agent
        self.agent: Optional[PPOAgent] = None
        
        # 训练记录
        self.training_history = {
            'episode_rewards': [],
            'episode_returns': [],
            'episode_sharpe': [],
            'episode_max_dd': [],
            'eval_returns': [],
            'eval_sharpe': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': []
        }
        
        # 可视化器
        self.visualizer = TrainingVisualizer(save_dir=self.train_config.log_dir)
        
        # 创建目录
        Path(self.train_config.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.train_config.log_dir).mkdir(parents=True, exist_ok=True)
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        import torch
        import random
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def prepare_data(self) -> None:
        """准备训练和测试数据"""
        print("=" * 60)
        print("准备数据...")
        print("=" * 60)
        
        all_data = {}
        
        # 获取所有股票数据
        for symbol in self.env_config.symbols:
            print(f"\n获取 {symbol} 数据...")
            try:
                df = self.data_fetcher.fetch_stock_data(
                    symbol=symbol,
                    start_date=self.env_config.start_date,
                    end_date=self.env_config.end_date,
                    use_cache=True
                )
                
                if len(df) > 0:
                    all_data[symbol] = df
                    print(f"  {symbol}: {len(df)} 条数据")
                else:
                    print(f"  {symbol}: 无数据")
                    
            except Exception as e:
                print(f"  {symbol} 获取失败: {e}")
        
        if not all_data:
            raise ValueError("无法获取任何股票数据")
        
        # 计算特征
        print("\n计算技术指标特征...")
        processed_data = {}
        
        for symbol, df in all_data.items():
            processed_df = self.feature_engineer.compute_features(
                df, 
                fit_scaler=(symbol == list(all_data.keys())[0])  # 只用第一只股票拟合 scaler
            )
            processed_data[symbol] = processed_df
            print(f"  {symbol}: {len(self.feature_engineer.get_feature_columns())} 个特征")
        
        # 对齐数据长度
        min_length = min(len(df) for df in processed_data.values())
        for symbol in processed_data:
            processed_data[symbol] = processed_data[symbol].iloc[-min_length:].reset_index(drop=True)
        
        # 划分训练集和测试集
        split_idx = int(min_length * self.env_config.train_ratio)
        
        for symbol, df in processed_data.items():
            self.train_data[symbol] = df.iloc[:split_idx].reset_index(drop=True)
            self.test_data[symbol] = df.iloc[split_idx:].reset_index(drop=True)
        
        print(f"\n数据划分完成:")
        print(f"  训练集: {split_idx} 条")
        print(f"  测试集: {min_length - split_idx} 条")
        print(f"  股票数: {len(processed_data)}")
    
    def create_environments(self) -> None:
        """创建训练和测试环境"""
        print("\n创建交易环境...")
        
        # 使用精简版核心特征（12个），降低状态空间复杂度
        feature_columns = self.feature_engineer.get_core_features()
        print(f"使用 {len(feature_columns)} 个核心特征: {feature_columns}")
        
        # 判断是否使用序列模型（LSTM 或 Transformer）
        model_type = getattr(self.ppo_config, 'model_type', 'lstm')
        use_sequence_model = model_type in ['lstm', 'transformer']
        
        # 保存环境参数
        self._env_kwargs = {
            'dfs': self.train_data,
            'feature_columns': feature_columns,
            'initial_balance': self.env_config.initial_balance,
            'transaction_cost': self.env_config.transaction_cost,
            'slippage': self.env_config.slippage,
            'max_position': self.env_config.max_position_per_stock,
            'lookback_window': self.env_config.lookback_window,
            'training_mode': True,
            'reward_scaling': 100.0,       # 1% 收益 = 1.0 奖励
            'trade_penalty': 0.05,         # 降低交易惩罚，鼓励交易
            'inactivity_penalty': 0.0,     # 取消空仓惩罚，由超额收益自然惩罚
            'use_lstm': use_sequence_model,  # LSTM/Transformer 都需要序列输入
            'episode_length': 500,         # 每个 episode 500 天
            'discrete_action': getattr(self.ppo_config, 'discrete_action', False),  # 离散动作模式
            'use_excess_return': True      # 使用超额收益作为奖励
        }
        
        # 创建单个环境用于获取空间信息
        sample_env = SingleStockEnv(**self._env_kwargs)
        self.observation_space = sample_env.observation_space
        self.action_space = sample_env.action_space
        
        # 创建训练环境
        self.train_env = SingleStockEnv(**self._env_kwargs)
        
        # 测试环境参数（推理模式，走完整数据）
        test_env_kwargs = {
            'dfs': self.test_data,
            'feature_columns': feature_columns,
            'initial_balance': self.env_config.initial_balance,
            'transaction_cost': self.env_config.transaction_cost,
            'slippage': self.env_config.slippage,
            'max_position': self.env_config.max_position_per_stock,
            'lookback_window': self.env_config.lookback_window,
            'training_mode': False,        # 推理模式：从头走到尾
            'reward_scaling': 100.0,
            'trade_penalty': 0.05,
            'inactivity_penalty': 0.0,
            'use_lstm': use_sequence_model,  # LSTM/Transformer 都需要序列输入
            'discrete_action': getattr(self.ppo_config, 'discrete_action', False),  # 离散动作模式
            'use_excess_return': True      # 使用超额收益作为奖励
        }
        
        # 创建测试环境（单个即可）
        self.test_env = SingleStockEnv(**test_env_kwargs)
        
        sample_env.close()
        
        # 打印环境信息
        discrete_mode = getattr(self.ppo_config, 'discrete_action', False)
        action_dim = getattr(self.ppo_config, 'action_dim', 3)
        print(f"  模型类型: {model_type}")
        print(f"  状态空间维度: {self.observation_space.shape}")
        if discrete_mode:
            if action_dim == 2:
                print(f"  动作空间: 离散 2动作 (0=空仓, 1=满仓)")
            else:
                print(f"  动作空间: 离散 {action_dim}动作")
        else:
            print(f"  动作空间维度: {self.action_space.shape}")
        print(f"  训练股票池: {list(self.train_data.keys())}")
        print(f"  测试环境步数: {self.test_env.n_steps}")
        print(f"  奖励模式: 超额收益 (策略收益 - 基准收益)")
        print(f"  交易惩罚: {self.train_env.trade_penalty}")
    
    def create_agent(self) -> None:
        """创建 PPO Agent"""
        print("\n创建 PPO Agent...")
        
        # 获取模型类型
        model_type = getattr(self.ppo_config, 'model_type', 'lstm')
        use_sequence_model = model_type in ['lstm', 'transformer']
        
        # LSTM/Transformer 模式下，state_dim 是每个时间步的特征数
        if use_sequence_model:
            # observation_space.shape = (lookback_window, n_features + 3)
            state_dim = self.observation_space.shape[1]  # 每个时间步的特征数
        else:
            # MLP 模式下是压扁的向量
            state_dim = self.observation_space.shape[0]
        
        # 离散动作模式
        discrete_action = getattr(self.ppo_config, 'discrete_action', False)
        if discrete_action:
            action_dim = getattr(self.ppo_config, 'action_dim', 3)  # 默认 3 个动作
        else:
            action_dim = self.action_space.shape[0]
        
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.ppo_config.hidden_dims,
            critic_hidden_dims=getattr(self.ppo_config, 'critic_hidden_dims', None),
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
            lr_schedule=getattr(self.ppo_config, 'lr_schedule', 'linear'),
            min_lr=getattr(self.ppo_config, 'min_lr', 1e-6),
            total_timesteps=self.train_config.total_timesteps,
            discrete_action=discrete_action,
            # Transformer 配置
            model_type=model_type,
            d_model=getattr(self.ppo_config, 'd_model', 128),
            nhead=getattr(self.ppo_config, 'nhead', 4),
            num_transformer_layers=getattr(self.ppo_config, 'num_transformer_layers', 2),
            dim_feedforward=getattr(self.ppo_config, 'dim_feedforward', 256),
            dropout=getattr(self.ppo_config, 'dropout', 0.1)
        )
        
        print(self.agent.get_network_summary())
    
    def train(self) -> Dict:
        """
        执行训练
        
        Returns:
            训练结果统计
        """
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
        # 准备工作
        if not self.train_data:
            self.prepare_data()
        
        if self.train_env is None:
            self.create_environments()
        
        if self.agent is None:
            self.create_agent()
        
        # 训练参数
        total_timesteps = self.train_config.total_timesteps
        n_steps = self.ppo_config.n_steps
        
        # 生成训练 ID（用于区分不同训练）
        self.train_id = time.strftime("%Y%m%d_%H%M%S")
        print(f"训练 ID: {self.train_id}")
        
        # 训练循环
        timesteps = 0
        episode = 0
        best_eval_return = -np.inf
        no_improve_count = 0
        
        start_time = time.time()
        
        # 初始化状态
        self._current_state = None
        
        while timesteps < total_timesteps:
            episode += 1
            
            # 收集经验（返回统计数据和最后的 terminated 状态）
            episode_reward, episode_steps, episode_stats, last_terminated = self._collect_rollout()
            timesteps += episode_steps
            
            # 获取最后状态的价值
            _, _, last_value = self.agent.select_action(self._current_state)
            
            # 更新策略（传入步数和 last_terminated 用于正确的 GAE 计算）
            update_stats = self.agent.update(last_value, n_steps=episode_steps, last_done=last_terminated)
            
            # 记录
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_returns'].append(episode_stats['total_return'])
            self.training_history['episode_sharpe'].append(episode_stats['sharpe_ratio'])
            self.training_history['episode_max_dd'].append(episode_stats['max_drawdown'])
            self.training_history['policy_loss'].append(update_stats['policy_loss'])
            self.training_history['value_loss'].append(update_stats['value_loss'])
            self.training_history['entropy'].append(update_stats['entropy'])
            
            # 打印进度
            if timesteps % self.train_config.log_freq < n_steps:
                elapsed = time.time() - start_time
                fps = timesteps / elapsed
                
                print(f"\n[Episode {episode}] Steps: {timesteps}/{total_timesteps} ({100*timesteps/total_timesteps:.1f}%)")
                print(f"  Episode Reward (episode): {episode_reward:.2f}")
                if 'rollout_reward_sum' in episode_stats:
                    print(f"  Rollout Reward (2048 steps): {episode_stats['rollout_reward_sum']:.2f}")
                print(f"  Total Return: {episode_stats['total_return']:.2%}")
                print(f"  Sharpe Ratio: {episode_stats['sharpe_ratio']:.2f}")
                print(f"  Max Drawdown: {episode_stats['max_drawdown']:.2%}")
                print(f"  Trades: {episode_stats.get('total_trades', 0)}")
                # 调试：奖励分解
                if 'dbg_reward_return_sum' in episode_stats:
                    print(f"  Reward Return Sum: {episode_stats['dbg_reward_return_sum']:.2f}")
                    print(f"  Reward Penalty Sum: {episode_stats['dbg_reward_penalty_sum']:.2f}")
                    print(f"  Reward Total Sum: {episode_stats['dbg_reward_total_sum']:.2f}")
                print(f"  Policy Loss: {update_stats['policy_loss']:.4f}")
                print(f"  Value Loss: {update_stats['value_loss']:.4f}")
                print(f"  Entropy: {update_stats['entropy']:.4f}")
                print(f"  Learning Rate: {update_stats['learning_rate']:.2e}")
                print(f"  FPS: {fps:.0f}")
            
            # 评估
            if timesteps % self.train_config.eval_freq < n_steps:
                eval_stats = self.evaluate()
                self.training_history['eval_returns'].append(eval_stats['total_return'])
                self.training_history['eval_sharpe'].append(eval_stats['sharpe_ratio'])
                
                print(f"\n[Evaluation]")
                print(f"  Test Return: {eval_stats['total_return']:.2%}")
                print(f"  Test Sharpe: {eval_stats['sharpe_ratio']:.2f}")
                print(f"  Test Max DD: {eval_stats['max_drawdown']:.2%}")
                
                # 保存最佳模型
                if eval_stats['total_return'] > best_eval_return:
                    best_eval_return = eval_stats['total_return']
                    self.save_model('best_model.pt')
                    no_improve_count = 0
                    print(f"  New best model saved!")
                else:
                    no_improve_count += 1
                
                # 更新可视化（使用训练 ID 命名）
                save_path = f"{self.train_config.log_dir}/training_curves_{self.train_id}.png"
                self.visualizer.plot_training_curves(self.training_history, save_path=save_path)
                
                # 早停检查
                if no_improve_count >= self.train_config.early_stop_patience:
                    print(f"\n早停：{no_improve_count} 次评估未改善")
                    break
            
            # 定期保存
            if timesteps % self.train_config.save_freq < n_steps:
                self.save_model(f'checkpoint_{timesteps}.pt')
        
        # 训练完成
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)
        print(f"总时间: {total_time/60:.1f} 分钟")
        print(f"总步数: {timesteps}")
        print(f"总回合: {episode}")
        print(f"最佳测试收益: {best_eval_return:.2%}")
        
        # 保存最终模型
        self.save_model('final_model.pt')
        
        # 保存训练历史
        self._save_training_history()
        
        return {
            'total_timesteps': timesteps,
            'total_episodes': episode,
            'total_time': total_time,
            'best_eval_return': best_eval_return,
            'final_train_return': self.training_history['episode_returns'][-1] if self.training_history['episode_returns'] else 0
        }
    
    def _collect_rollout(self) -> Tuple[float, int, Dict, bool]:
        """
        收集一个 rollout 的经验（支持并行环境）
        
        Returns:
            (episode_reward, episode_steps, episode_stats, last_terminated)
        """
        # 初始化状态
        if not hasattr(self, '_current_state') or self._current_state is None:
            self._current_state, _ = self.train_env.reset()
        
        state = self._current_state
        total_reward = 0
        total_steps = 0
        episode_stats = None
        last_terminated = False  # 记录最后一步是否为 terminated（破产）
        
        self.agent.set_training_mode(True)
        
        n_steps = self.ppo_config.n_steps
        
        for _ in range(n_steps):
            # 选择动作
            action, log_prob, value = self.agent.select_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = self.train_env.step(action)
            done = terminated or truncated
            
            # 存储经验（存储 terminated 而非 done，用于正确的 GAE 计算）
            # 注意：这里仍然存储 done，但在 GAE 计算时会特殊处理最后一步
            self.agent.store_transition(
                state, action, reward, value, log_prob, terminated
            )
            total_reward += reward
            total_steps += 1
            
            # 记录最后一步的 terminated 状态
            last_terminated = terminated
            
            # 获取完成的 episode 统计
            if done and 'episode_stats' in info:
                episode_stats = info['episode_stats']
            
            state = next_state
            
            # 如果 episode 结束，重置环境和 LSTM 隐藏状态
            if done:
                state, _ = self.train_env.reset()
                # 重置 LSTM 隐藏状态，避免跨 episode 信息泄露
                if self.ppo_config.use_lstm:
                    self.agent.lstm_hidden = None
        
        # 更新当前状态
        self._current_state = state
        
        # 如果没有完成的 episode，获取当前统计
        if episode_stats is None:
            episode_stats = self.train_env.get_episode_stats()
        
        # 区分：rollout 奖励（2048步累计） vs 该 episode 的奖励（dbg_reward_total_sum）
        episode_stats['rollout_reward_sum'] = total_reward
        episode_reward = episode_stats.get('dbg_reward_total_sum', total_reward)
        
        return episode_reward, total_steps, episode_stats, last_terminated
    
    def evaluate(self, n_episodes: int = 1) -> Dict:
        """
        评估当前策略
        
        Args:
            n_episodes: 评估回合数
            
        Returns:
            评估统计
        """
        self.agent.set_training_mode(False)
        
        all_returns = []
        all_sharpe = []
        all_max_dd = []
        
        print(f"  [Eval] Starting evaluation, test_env.n_steps={self.test_env.n_steps}")
        
        for ep in range(n_episodes):
            state, _ = self.test_env.reset()
            done = False
            steps = 0
            
            while not done:
                # 评估时使用随机模式，因为确定性模式下初期模型动作太小
                action, _, _ = self.agent.select_action(state, deterministic=False)
                state, _, terminated, truncated, info = self.test_env.step(action)
                done = terminated or truncated
                steps += 1
            
            stats = self.test_env.get_episode_stats()
            all_returns.append(stats['total_return'])
            all_sharpe.append(stats['sharpe_ratio'])
            all_max_dd.append(stats['max_drawdown'])
            
            print(f"  [Eval] Episode {ep+1}: steps={steps}, return={stats['total_return']:.2%}, trades={stats.get('total_trades', 0)}")
        
        return {
            'total_return': np.mean(all_returns),
            'sharpe_ratio': np.mean(all_sharpe),
            'max_drawdown': np.mean(all_max_dd)
        }
    
    def save_model(self, filename: str) -> None:
        """保存模型"""
        path = Path(self.train_config.model_dir) / filename
        self.agent.save(str(path))
    
    def load_model(self, filename: str) -> None:
        """加载模型"""
        path = Path(self.train_config.model_dir) / filename
        self.agent.load(str(path))
    
    def _save_training_history(self) -> None:
        """保存训练历史"""
        history_path = Path(self.train_config.log_dir) / 'training_history.json'
        
        # 转换为可序列化格式
        history = {}
        for key, values in self.training_history.items():
            history[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"训练历史已保存到 {history_path}")
    
    def get_training_summary(self) -> str:
        """获取训练摘要"""
        if not self.training_history['episode_returns']:
            return "尚未开始训练"
        
        returns = self.training_history['episode_returns']
        sharpe = self.training_history['episode_sharpe']
        
        summary = f"""
训练摘要
========
总回合数: {len(returns)}
平均收益率: {np.mean(returns):.2%}
最佳收益率: {np.max(returns):.2%}
最差收益率: {np.min(returns):.2%}
平均夏普比率: {np.mean(sharpe):.2f}
最佳夏普比率: {np.max(sharpe):.2f}

最近 10 回合:
  平均收益率: {np.mean(returns[-10:]):.2%}
  平均夏普比率: {np.mean(sharpe[-10:]):.2f}
"""
        return summary
