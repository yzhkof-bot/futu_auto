"""
PPO (Proximal Policy Optimization) Agent
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path

from .networks import (
    ActorCritic, 
    LSTMActorCritic, 
    DiscreteLSTMActorCritic,
    TransformerActorCritic,
    DiscreteTransformerActorCritic
)


class RolloutBuffer:
    """
    经验回放缓冲区
    存储一个 rollout 周期的数据
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = None
        self.returns = None
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """添加一条经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        last_done: bool = False
    ):
        """
        计算 GAE (Generalized Advantage Estimation) 和回报
        
        Args:
            last_value: 最后状态的价值估计
            gamma: 折扣因子
            gae_lambda: GAE lambda
            last_done: 最后一步是否为 terminated（破产），用于区分 truncated
        """
        n_steps = len(self.rewards)
        
        advantages = np.zeros(n_steps, dtype=np.float32)
        last_gae = 0
        
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                # 最后一步：如果是 truncated（时间到）而非 terminated（破产），
                # 应该 bootstrap 未来价值
                next_value = last_value
                # 只有真正的 terminated 才截断未来奖励
                next_non_terminal = 0.0 if last_done else 1.0
            else:
                next_value = self.values[t + 1]
                # 中间步骤：如果 done 是因为 episode 结束（可能是 truncated），
                # 我们仍然需要正确处理
                next_non_terminal = 1.0 - float(self.dones[t])
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        
        self.advantages = advantages
        self.returns = advantages + np.array(self.values)
    
    def get_batches(
        self,
        batch_size: int,
        device: torch.device
    ) -> DataLoader:
        """获取批次数据"""
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.FloatTensor(np.array(self.actions)).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        advantages = torch.FloatTensor(self.advantages).to(device)
        returns = torch.FloatTensor(self.returns).to(device)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset = TensorDataset(states, actions, old_log_probs, advantages, returns)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = None
        self.returns = None


class PPOAgent:
    """
    PPO Agent
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 128],
        critic_hidden_dims: Optional[List[int]] = None,  # Critic 独立容量
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        use_lstm: bool = False,
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 2,
        device: str = "cpu",
        lr_schedule: str = "linear",
        min_lr: float = 1e-6,
        total_timesteps: int = 200_000,
        discrete_action: bool = False,  # 是否使用离散动作空间
        # Transformer 配置
        model_type: str = "lstm",  # "lstm" 或 "transformer"
        d_model: int = 128,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        """
        初始化 PPO Agent
        
        Args:
            state_dim: 状态维度（LSTM/Transformer 模式下是每个时间步的特征数）
            action_dim: 动作维度
            hidden_dims: 隐藏层维度 (Actor)
            critic_hidden_dims: Critic 隐藏层维度（None 则与 Actor 相同）
            learning_rate: 学习率
            gamma: 折扣因子
            gae_lambda: GAE lambda
            clip_epsilon: PPO clip 参数
            value_coef: 价值损失系数
            entropy_coef: 熵正则化系数
            max_grad_norm: 梯度裁剪
            n_epochs: 每次更新的 epoch 数
            batch_size: 批次大小
            use_lstm: 是否使用 LSTM（兼容旧代码）
            lstm_hidden_size: LSTM 隐藏层大小
            num_lstm_layers: LSTM 层数
            device: 计算设备
            lr_schedule: 学习率调度方式 (linear, cosine, constant)
            min_lr: 最小学习率
            total_timesteps: 总训练步数（用于学习率调度）
            discrete_action: 是否使用离散动作空间（买入/持有/卖出）
            model_type: 模型类型 "lstm" 或 "transformer"
            d_model: Transformer 隐藏维度
            nhead: 注意力头数
            num_transformer_layers: Transformer 层数
            dim_feedforward: FFN 隐藏维度
            dropout: Dropout 比率
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.initial_entropy_coef = entropy_coef  # 保存初始熵系数
        self.entropy_coef = entropy_coef
        self.min_entropy_coef = 0.0001  # 最小熵系数（保持少量探索）
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.discrete_action = discrete_action
        self.device = torch.device(device)
        
        # 模型类型：优先使用 model_type，兼容旧的 use_lstm
        self.model_type = model_type
        if use_lstm and model_type == "lstm":
            self.model_type = "lstm"
        self.use_lstm = (self.model_type == "lstm")  # 兼容旧代码
        
        # 学习率调度参数
        self.initial_lr = learning_rate
        self.min_lr = min_lr
        self.lr_schedule = lr_schedule
        self.total_timesteps = total_timesteps
        self.current_timesteps = 0
        
        # 创建网络
        if self.model_type == "transformer":
            if discrete_action:
                self.network = DiscreteTransformerActorCritic(
                    input_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dims=hidden_dims,
                    critic_hidden_dims=critic_hidden_dims,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_transformer_layers,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout
                ).to(self.device)
            else:
                self.network = TransformerActorCritic(
                    input_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dims=hidden_dims,
                    critic_hidden_dims=critic_hidden_dims,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_transformer_layers,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout
                ).to(self.device)
        elif self.model_type == "lstm":
            if discrete_action:
                # 离散动作空间的 LSTM 网络
                self.network = DiscreteLSTMActorCritic(
                    input_dim=state_dim,
                    action_dim=action_dim,  # 默认 3: 卖出/持有/买入
                    hidden_dims=hidden_dims,
                    critic_hidden_dims=critic_hidden_dims,
                    lstm_hidden_size=lstm_hidden_size,
                    num_lstm_layers=num_lstm_layers
                ).to(self.device)
            else:
                # 连续动作空间的 LSTM 网络
                self.network = LSTMActorCritic(
                    input_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dims=hidden_dims,
                    critic_hidden_dims=critic_hidden_dims,
                    lstm_hidden_size=lstm_hidden_size,
                    num_lstm_layers=num_lstm_layers
                ).to(self.device)
        else:
            self.network = ActorCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims
            ).to(self.device)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.network.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5  # 添加权重衰减
        )
        
        # 经验缓冲区
        self.buffer = RolloutBuffer()
        
        # LSTM 隐藏状态（Transformer 不需要）
        self.lstm_hidden = None
        
        # 训练统计
        self.train_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'approx_kl': []
        }
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        选择动作
        
        Args:
            state: 状态
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 动作
            log_prob: 对数概率
            value: 状态价值
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if self.model_type in ["lstm", "transformer"]:
                # LSTM 和 Transformer 使用相同接口
                # Transformer 返回 None 作为 hidden，但接口兼容
                action, log_prob, value, _, self.lstm_hidden = self.network.get_action(
                    state_tensor, self.lstm_hidden, deterministic
                )
            else:
                action, log_prob, value, _ = self.network.get_action(
                    state_tensor, deterministic
                )
            
            action = action.cpu().numpy().squeeze()
            log_prob = log_prob.cpu().numpy().squeeze()
            value = value.cpu().numpy().squeeze()
        
        return action, log_prob, value
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """存储转移"""
        self.buffer.add(state, action, reward, value, log_prob, done)
    
    def update(self, last_value: float, n_steps: int = 0, last_done: bool = False) -> Dict[str, float]:
        """
        更新策略
        
        Args:
            last_value: 最后状态的价值估计
            n_steps: 本次收集的步数（用于学习率调度）
            last_done: 最后一步是否为 terminated（破产）
            
        Returns:
            训练统计信息
        """
        # 更新当前步数
        self.current_timesteps += n_steps
        
        # 更新学习率
        self._update_learning_rate()
        
        # 计算回报和优势
        self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda, last_done
        )
        
        # 获取批次数据
        dataloader = self.buffer.get_batches(self.batch_size, self.device)
        
        # 训练多个 epoch
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        n_updates = 0
        
        for epoch in range(self.n_epochs):
            for batch in dataloader:
                states, actions, old_log_probs, advantages, returns = batch
                
                # 确保维度正确
                if old_log_probs.dim() == 1:
                    old_log_probs = old_log_probs.unsqueeze(-1)
                if advantages.dim() == 1:
                    advantages = advantages.unsqueeze(-1)
                if returns.dim() == 1:
                    returns = returns.unsqueeze(-1)
                
                # 评估动作
                log_probs, values, entropy = self.network.evaluate_actions(states, actions)
                
                # 计算比率，限制范围防止爆炸
                log_ratio = log_probs - old_log_probs
                log_ratio = torch.clamp(log_ratio, -10.0, 10.0)  # 防止 exp 爆炸
                ratio = torch.exp(log_ratio)
                
                # PPO clip 损失
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = nn.MSELoss()(values, returns)
                
                # 熵奖励
                entropy_loss = -entropy.mean()
                
                # 检查 loss 是否有效
                if torch.isnan(policy_loss) or torch.isinf(policy_loss):
                    print("Warning: Invalid policy loss, skipping batch")
                    continue
                if torch.isnan(value_loss) or torch.isinf(value_loss):
                    print("Warning: Invalid value loss, skipping batch")
                    continue
                
                # 总损失
                loss = (
                    policy_loss 
                    + self.value_coef * value_loss 
                    + self.entropy_coef * entropy_loss
                )
                
                # 再次检查总损失
                if torch.isnan(loss) or torch.isinf(loss):
                    print("Warning: Invalid total loss, skipping batch")
                    continue
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # 计算近似 KL 散度
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                total_approx_kl += approx_kl
                n_updates += 1
        
        # 清空缓冲区
        self.buffer.clear()
        
        # 重置 LSTM 隐藏状态（Transformer 不需要，但保持兼容）
        if self.model_type == "lstm":
            self.lstm_hidden = None
        
        # 返回统计信息
        if n_updates == 0:
            n_updates = 1  # 防止除零
        
        stats = {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'approx_kl': total_approx_kl / n_updates,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        # 记录统计
        for key, value in stats.items():
            if key in self.train_stats:
                self.train_stats[key].append(value)
        
        return stats
    
    def _update_learning_rate(self):
        """根据调度策略更新学习率"""
        progress = min(1.0, self.current_timesteps / self.total_timesteps)
        
        if self.lr_schedule == "linear":
            # 线性衰减
            lr = self.initial_lr * (1 - progress) + self.min_lr * progress
        elif self.lr_schedule == "cosine":
            # 余弦退火
            import math
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        else:
            # 常数学习率
            lr = self.initial_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # 同时衰减熵系数（在前 80% 的训练中线性衰减到 0）
        entropy_decay_progress = min(1.0, progress / 0.8)
        self.entropy_coef = self.initial_entropy_coef * (1 - entropy_decay_progress) + self.min_entropy_coef * entropy_decay_progress
    
    def save(self, path: str):
        """保存模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_timesteps': self.current_timesteps,
            'train_stats': self.train_stats,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'use_lstm': self.use_lstm,
                'model_type': self.model_type,
                'discrete_action': self.discrete_action,
                'initial_lr': self.initial_lr,
                'min_lr': self.min_lr,
                'lr_schedule': self.lr_schedule,
                'total_timesteps': self.total_timesteps
            }
        }, path)
        
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_timesteps = checkpoint.get('current_timesteps', 0)
        self.train_stats = checkpoint.get('train_stats', self.train_stats)
        
        print(f"Model loaded from {path}")
    
    def set_training_mode(self, training: bool = True):
        """设置训练/评估模式"""
        self.network.train(training)
    
    def get_network_summary(self) -> str:
        """获取网络结构摘要"""
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        
        summary = f"""
Network Summary:
================
Model Type: {self.model_type}
State Dimension: {self.state_dim}
Action Dimension: {self.action_dim}
Discrete Action: {self.discrete_action}
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Device: {self.device}
"""
        return summary
