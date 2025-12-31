"""
神经网络架构 - Actor-Critic 网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import List, Tuple, Optional
import numpy as np


class ActorNetwork(nn.Module):
    """
    Actor 网络 - 输出动作的均值和标准差
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 128],
        log_std_min: float = -20,
        log_std_max: float = 0.5  # 限制最大标准差约1.65
    ):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 构建隐藏层
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # 均值输出层
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        
        # 独立的可学习 log_std 参数（不依赖状态）
        # 初始化为0，让 sigmoid(0)=0.5，处于中间位置
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
        # 输出层使用较小的初始化
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        # log_std 已经是 Parameter，初始化为 0
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 状态张量
            
        Returns:
            mean: 动作均值
            log_std: 动作对数标准差
        """
        features = self.shared_layers(state)
        
        mean = self.mean_layer(features)
        
        # 使用 sigmoid 映射到固定范围 [std_min, std_max]，梯度永不消失
        # std 范围 [0.05, 0.8]，既保证探索，又防止 tanh 饱和
        std_min, std_max = 0.05, 0.8
        std = std_min + (std_max - std_min) * torch.sigmoid(self.log_std)
        log_std = torch.log(std)
        log_std = log_std.expand(state.size(0), -1)
        
        return mean, log_std
    
    def get_action(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作
        
        Args:
            state: 状态张量
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 动作（tanh 压缩后）
            log_prob: 动作的对数概率
            mean: 动作均值
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros_like(action)
        else:
            # 重参数化采样
            normal = Normal(mean, std)
            x = normal.rsample()  # 重参数化
            action = torch.tanh(x)
            
            # 计算对数概率（考虑 tanh 变换的雅可比行列式）
            log_prob = normal.log_prob(x)
            # tanh 变换的修正
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob, mean


class CriticNetwork(nn.Module):
    """
    Critic 网络 - 输出状态价值
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [256, 256, 128]
    ):
        super().__init__()
        
        # 构建网络
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态张量
            
        Returns:
            value: 状态价值
        """
        return self.network(state)


class ActorCritic(nn.Module):
    """
    Actor-Critic 联合网络
    共享部分底层特征提取
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 128],
        share_features: bool = True,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()
        
        self.share_features = share_features
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        if share_features:
            # 共享特征提取层
            shared_layers = []
            prev_dim = state_dim
            for i, hidden_dim in enumerate(hidden_dims[:-1]):
                shared_layers.append(nn.Linear(prev_dim, hidden_dim))
                shared_layers.append(nn.LayerNorm(hidden_dim))
                shared_layers.append(nn.ReLU())
                prev_dim = hidden_dim
            
            self.shared_net = nn.Sequential(*shared_layers)
            
            # Actor 头
            self.actor_head = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.LayerNorm(hidden_dims[-1]),
                nn.ReLU()
            )
            self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
            self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
            
            # Critic 头
            self.critic_head = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.LayerNorm(hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], 1)
            )
        else:
            # 独立的 Actor 和 Critic
            self.actor = ActorNetwork(state_dim, action_dim, hidden_dims, log_std_min, log_std_max)
            self.critic = CriticNetwork(state_dim, hidden_dims)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
        if self.share_features:
            nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
            nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
    
    def forward(
        self, 
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            mean: 动作均值
            log_std: 动作对数标准差
            value: 状态价值
        """
        # 处理输入中的 NaN
        state = torch.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)
        
        if self.share_features:
            shared_features = self.shared_net(state)
            
            # Actor
            actor_features = self.actor_head(shared_features)
            mean = self.mean_layer(actor_features)
            log_std = self.log_std_layer(actor_features)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            
            # Critic
            value = self.critic_head(shared_features)
        else:
            mean, log_std = self.actor(state)
            value = self.critic(state)
        
        # 确保输出不含 NaN
        mean = torch.nan_to_num(mean, nan=0.0)
        log_std = torch.nan_to_num(log_std, nan=0.0)
        value = torch.nan_to_num(value, nan=0.0)
        
        return mean, log_std, value
    
    def get_action(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作和价值
        
        Returns:
            action: 动作
            log_prob: 对数概率
            value: 状态价值
            mean: 动作均值
        """
        mean, log_std, value = self.forward(state)
        std = log_std.exp()
        
        # 限制 std 范围，防止数值不稳定
        std = torch.clamp(std, min=1e-6, max=10.0)
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros(action.shape[0], 1, device=action.device)
        else:
            normal = Normal(mean, std)
            x = normal.rsample()
            action = torch.tanh(x)
            
            # 计算 log_prob，使用更稳定的方式
            log_prob = normal.log_prob(x)
            # 限制 action 范围以避免 log(0)
            action_safe = torch.clamp(action, -0.9999, 0.9999)
            log_prob = log_prob - torch.log(1 - action_safe.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
            # 限制 log_prob 范围
            log_prob = torch.clamp(log_prob, min=-20.0, max=2.0)
        
        return action, log_prob, value, mean
    
    def evaluate_actions(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定状态-动作对
        
        Returns:
            log_prob: 动作的对数概率
            value: 状态价值
            entropy: 策略熵
        """
        mean, log_std, value = self.forward(state)
        std = log_std.exp()
        
        # 限制 std 范围
        std = torch.clamp(std, min=1e-6, max=10.0)
        
        normal = Normal(mean, std)
        
        # 反向计算 tanh 前的值，限制范围以保证数值稳定
        action_clipped = torch.clamp(action, -0.9999, 0.9999)
        x = torch.atanh(action_clipped)
        
        log_prob = normal.log_prob(x)
        log_prob = log_prob - torch.log(1 - action_clipped.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # 限制 log_prob 范围
        log_prob = torch.clamp(log_prob, min=-20.0, max=2.0)
        
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        # 限制 entropy 范围
        entropy = torch.clamp(entropy, min=-10.0, max=10.0)
        
        return log_prob, value, entropy
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """获取状态价值"""
        if self.share_features:
            shared_features = self.shared_net(state)
            return self.critic_head(shared_features)
        else:
            return self.critic(state)


class LSTMActorCritic(nn.Module):
    """
    带 LSTM 的 Actor-Critic 网络
    用于捕捉时序依赖
    
    输入格式: (batch, seq_len, n_features)
    - seq_len: lookback_window (如 60)
    - n_features: 特征数 + 账户状态 (如 39 + 3 = 42)
    """
    
    def __init__(
        self,
        input_dim: int,           # 每个时间步的特征数 (n_features + 3)
        action_dim: int,
        hidden_dims: List[int] = [256, 128],  # Actor 的 MLP 层
        critic_hidden_dims: Optional[List[int]] = None,  # Critic 独立的 MLP 层（None 则与 Actor 相同）
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 2,
        log_std_min: float = -20,
        log_std_max: float = 0.5,  # 降低上限，限制最大 std≈1.65
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Critic 使用独立配置，默认与 Actor 相同
        if critic_hidden_dims is None:
            critic_hidden_dims = hidden_dims
        
        # 输入特征预处理层
        self.input_norm = nn.LayerNorm(input_dim)
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # 共享的后处理层（可选，如果 hidden_dims 长度 > 1）
        shared_layers = []
        prev_dim = lstm_hidden_size
        for hidden_dim in hidden_dims[:-1] if len(hidden_dims) > 1 else []:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.LayerNorm(hidden_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.shared_net = nn.Sequential(*shared_layers) if shared_layers else nn.Identity()
        
        # Actor 头
        actor_hidden = hidden_dims[-1] if hidden_dims else lstm_hidden_size
        self.actor_head = nn.Sequential(
            nn.Linear(prev_dim, actor_hidden),
            nn.LayerNorm(actor_hidden),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(actor_hidden, action_dim)
        # 使用独立的可学习 log_std 参数，初始化为 0 (std≈0.425，适度探索)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic 头 - 独立的更大容量网络
        critic_layers = []
        critic_prev_dim = lstm_hidden_size  # Critic 从 LSTM 输出开始，不共享 shared_net
        for hidden_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(critic_prev_dim, hidden_dim))
            critic_layers.append(nn.LayerNorm(hidden_dim))
            critic_layers.append(nn.ReLU())
            critic_layers.append(nn.Dropout(dropout))
            critic_prev_dim = hidden_dim
        critic_layers.append(nn.Linear(critic_prev_dim, 1))
        self.critic_net = nn.Sequential(*critic_layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
        
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        # log_std 已经初始化为 0，不需要额外初始化
    
    def forward(
        self, 
        state: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """
        前向传播
        
        Args:
            state: 状态张量 (batch, seq_len, input_dim) 或 (seq_len, input_dim)
            hidden: LSTM 隐藏状态 (可选)
            
        Returns:
            mean, log_std, value, new_hidden
        """
        # 处理输入中的 NaN
        state = torch.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # 确保输入是 3D: (batch, seq_len, input_dim)
        if state.dim() == 2:
            state = state.unsqueeze(0)  # (1, seq_len, input_dim)
        
        batch_size = state.shape[0]
        
        # 输入归一化
        state = self.input_norm(state)
        
        # LSTM
        if hidden is None:
            lstm_out, new_hidden = self.lstm(state)
        else:
            lstm_out, new_hidden = self.lstm(state, hidden)
        
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]  # (batch, lstm_hidden_size)
        
        # Actor 分支：共享层 + Actor 头
        actor_features = self.shared_net(lstm_out)
        actor_features = self.actor_head(actor_features)
        mean = self.mean_layer(actor_features)
        
        # 使用 sigmoid 映射到固定范围 [std_min, std_max]，梯度永不消失
        std_min, std_max = 0.05, 0.8
        std = std_min + (std_max - std_min) * torch.sigmoid(self.log_std)
        log_std = torch.log(std)
        log_std = log_std.expand(mean.shape[0], -1)
        
        # Critic 分支：独立网络，直接从 LSTM 输出
        value = self.critic_net(lstm_out)
        
        # 确保输出不含 NaN
        mean = torch.nan_to_num(mean, nan=0.0)
        log_std = torch.nan_to_num(log_std, nan=0.0)
        value = torch.nan_to_num(value, nan=0.0)
        
        return mean, log_std, value, new_hidden
    
    def _init_hidden(
        self, 
        batch_size: int, 
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """初始化 LSTM 隐藏状态"""
        h = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        c = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        return (h, c)
    
    def get_action(
        self, 
        state: torch.Tensor,
        hidden: Optional[Tuple] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """获取动作"""
        mean, log_std, value, new_hidden = self.forward(state, hidden)
        std = log_std.exp()
        std = torch.clamp(std, min=1e-6, max=10.0)
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros(action.shape[0], 1, device=action.device)
        else:
            normal = Normal(mean, std)
            x = normal.rsample()
            action = torch.tanh(x)
            
            log_prob = normal.log_prob(x)
            action_safe = torch.clamp(action, -0.9999, 0.9999)
            log_prob = log_prob - torch.log(1 - action_safe.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            log_prob = torch.clamp(log_prob, min=-20.0, max=2.0)
        
        return action, log_prob, value, mean, new_hidden
    
    def evaluate_actions(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定状态-动作对
        
        Returns:
            log_prob: 动作的对数概率
            value: 状态价值
            entropy: 策略熵
        """
        mean, log_std, value, _ = self.forward(state)
        std = log_std.exp()
        std = torch.clamp(std, min=1e-6, max=10.0)
        
        normal = Normal(mean, std)
        
        # 反向计算 tanh 前的值
        action_clipped = torch.clamp(action, -0.9999, 0.9999)
        x = torch.atanh(action_clipped)
        
        log_prob = normal.log_prob(x)
        log_prob = log_prob - torch.log(1 - action_clipped.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob = torch.clamp(log_prob, min=-20.0, max=2.0)
        
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        entropy = torch.clamp(entropy, min=-10.0, max=10.0)
        
        return log_prob, value, entropy
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """获取状态价值"""
        # 处理输入
        state = torch.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)
        if state.dim() == 2:
            state = state.unsqueeze(0)
        
        state = self.input_norm(state)
        lstm_out, _ = self.lstm(state)
        lstm_out = lstm_out[:, -1, :]
        
        return self.critic_net(lstm_out)


class DiscreteLSTMActorCritic(nn.Module):
    """
    离散动作空间的 LSTM Actor-Critic 网络
    
    动作空间: 3 个离散动作
    - 0: 卖出清仓 (0%)
    - 1: 持有不变
    - 2: 买入满仓 (100%)
    
    输入格式: (batch, seq_len, n_features)
    """
    
    def __init__(
        self,
        input_dim: int,           # 每个时间步的特征数 (n_features + 3)
        action_dim: int = 3,      # 离散动作数量：卖出/持有/买入
        hidden_dims: List[int] = [256, 128],  # Actor 的 MLP 层
        critic_hidden_dims: Optional[List[int]] = None,  # Critic 独立的 MLP 层
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        
        # Critic 使用独立配置，默认与 Actor 相同
        if critic_hidden_dims is None:
            critic_hidden_dims = hidden_dims
        
        # 输入特征预处理层
        self.input_norm = nn.LayerNorm(input_dim)
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # 共享的后处理层
        shared_layers = []
        prev_dim = lstm_hidden_size
        for hidden_dim in hidden_dims[:-1] if len(hidden_dims) > 1 else []:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.LayerNorm(hidden_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.shared_net = nn.Sequential(*shared_layers) if shared_layers else nn.Identity()
        
        # Actor 头 - 输出动作概率 logits
        actor_hidden = hidden_dims[-1] if hidden_dims else lstm_hidden_size
        self.actor_head = nn.Sequential(
            nn.Linear(prev_dim, actor_hidden),
            nn.LayerNorm(actor_hidden),
            nn.ReLU(),
            nn.Linear(actor_hidden, action_dim)  # 输出 3 个动作的 logits
        )
        
        # Critic 头 - 独立的更大容量网络
        critic_layers = []
        critic_prev_dim = lstm_hidden_size  # Critic 从 LSTM 输出开始
        for hidden_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(critic_prev_dim, hidden_dim))
            critic_layers.append(nn.LayerNorm(hidden_dim))
            critic_layers.append(nn.ReLU())
            critic_layers.append(nn.Dropout(dropout))
            critic_prev_dim = hidden_dim
        critic_layers.append(nn.Linear(critic_prev_dim, 1))
        self.critic_net = nn.Sequential(*critic_layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(
        self, 
        state: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """
        前向传播
        
        Args:
            state: 状态张量 (batch, seq_len, input_dim)
            hidden: LSTM 隐藏状态 (可选)
            
        Returns:
            action_logits: 动作 logits (batch, action_dim)
            value: 状态价值 (batch, 1)
            new_hidden: 新的 LSTM 隐藏状态
        """
        # 处理输入中的 NaN
        state = torch.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # 确保输入是 3D: (batch, seq_len, input_dim)
        if state.dim() == 2:
            state = state.unsqueeze(0)
        
        # 输入归一化
        state = self.input_norm(state)
        
        # LSTM
        if hidden is None:
            lstm_out, new_hidden = self.lstm(state)
        else:
            lstm_out, new_hidden = self.lstm(state, hidden)
        
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]  # (batch, lstm_hidden_size)
        
        # Actor 分支：输出动作 logits
        actor_features = self.shared_net(lstm_out)
        action_logits = self.actor_head(actor_features)
        
        # Critic 分支：独立网络
        value = self.critic_net(lstm_out)
        
        # 确保输出不含 NaN
        action_logits = torch.nan_to_num(action_logits, nan=0.0)
        value = torch.nan_to_num(value, nan=0.0)
        
        return action_logits, value, new_hidden
    
    def get_action(
        self, 
        state: torch.Tensor,
        hidden: Optional[Tuple] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """
        获取动作
        
        Returns:
            action: 离散动作 (batch,) 值为 0, 1, 2
            log_prob: 动作的对数概率 (batch, 1)
            value: 状态价值 (batch, 1)
            action_probs: 动作概率分布 (batch, action_dim)
            new_hidden: LSTM 隐藏状态
        """
        action_logits, value, new_hidden = self.forward(state, hidden)
        
        # 创建 Categorical 分布
        dist = Categorical(logits=action_logits)
        
        if deterministic:
            # 确定性策略：选择概率最高的动作
            action = torch.argmax(action_logits, dim=-1)
            log_prob = dist.log_prob(action).unsqueeze(-1)
        else:
            # 随机策略：从分布中采样
            action = dist.sample()
            log_prob = dist.log_prob(action).unsqueeze(-1)
        
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action, log_prob, value, action_probs, new_hidden
    
    def evaluate_actions(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定状态-动作对
        
        Args:
            state: 状态张量
            action: 离散动作 (batch,) 或 (batch, 1)
            
        Returns:
            log_prob: 动作的对数概率 (batch, 1)
            value: 状态价值 (batch, 1)
            entropy: 策略熵 (batch, 1)
        """
        action_logits, value, _ = self.forward(state)
        
        # 确保 action 是 1D
        if action.dim() > 1:
            action = action.squeeze(-1)
        action = action.long()
        
        # 创建 Categorical 分布
        dist = Categorical(logits=action_logits)
        
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        
        return log_prob, value, entropy
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """获取状态价值"""
        state = torch.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)
        if state.dim() == 2:
            state = state.unsqueeze(0)
        
        state = self.input_norm(state)
        lstm_out, _ = self.lstm(state)
        lstm_out = lstm_out[:, -1, :]
        
        return self.critic_net(lstm_out)


# ==================== Transformer 架构 ====================

class PositionalEncoding(nn.Module):
    """
    位置编码 - 为序列添加位置信息
    """
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerActorCritic(nn.Module):
    """
    Transformer Actor-Critic 网络（连续动作空间）
    
    使用 Transformer Encoder 捕捉时序依赖
    
    输入格式: (batch, seq_len, n_features)
    - seq_len: lookback_window (如 60)
    - n_features: 特征数 + 账户状态 (如 39 + 3 = 42)
    """
    
    def __init__(
        self,
        input_dim: int,           # 每个时间步的特征数
        action_dim: int,
        hidden_dims: List[int] = [256, 128],  # Actor 的 MLP 层
        critic_hidden_dims: Optional[List[int]] = None,
        d_model: int = 128,       # Transformer 隐藏维度
        nhead: int = 4,           # 注意力头数
        num_layers: int = 2,      # Transformer 层数
        dim_feedforward: int = 256,  # FFN 隐藏维度
        log_std_min: float = -20,
        log_std_max: float = 0.5,
        dropout: float = 0.1,
        max_seq_len: int = 200
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        if critic_hidden_dims is None:
            critic_hidden_dims = hidden_dims
        
        # 输入投影层：将 input_dim 映射到 d_model
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 共享的后处理层
        shared_layers = []
        prev_dim = d_model
        for hidden_dim in hidden_dims[:-1] if len(hidden_dims) > 1 else []:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.LayerNorm(hidden_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.shared_net = nn.Sequential(*shared_layers) if shared_layers else nn.Identity()
        
        # Actor 头
        actor_hidden = hidden_dims[-1] if hidden_dims else d_model
        self.actor_head = nn.Sequential(
            nn.Linear(prev_dim, actor_hidden),
            nn.LayerNorm(actor_hidden),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(actor_hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic 头 - 独立网络
        critic_layers = []
        critic_prev_dim = d_model
        for hidden_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(critic_prev_dim, hidden_dim))
            critic_layers.append(nn.LayerNorm(hidden_dim))
            critic_layers.append(nn.ReLU())
            critic_layers.append(nn.Dropout(dropout))
            critic_prev_dim = hidden_dim
        critic_layers.append(nn.Linear(critic_prev_dim, 1))
        self.critic_net = nn.Sequential(*critic_layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        nn.init.xavier_uniform_(self.mean_layer.weight, gain=0.01)
    
    def forward(
        self, 
        state: torch.Tensor,
        hidden: Optional[Tuple] = None  # 保留接口兼容性，Transformer 不需要 hidden
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        前向传播
        
        Args:
            state: 状态张量 (batch, seq_len, input_dim)
            hidden: 保留接口兼容性，不使用
            
        Returns:
            mean, log_std, value, None
        """
        state = torch.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)
        
        if state.dim() == 2:
            state = state.unsqueeze(0)
        
        batch_size = state.shape[0]
        
        # 输入投影
        x = self.input_projection(state)  # (batch, seq_len, d_model)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # 取最后一个时间步（或可以用 mean pooling）
        x = x[:, -1, :]  # (batch, d_model)
        
        # Actor 分支
        actor_features = self.shared_net(x)
        actor_features = self.actor_head(actor_features)
        mean = self.mean_layer(actor_features)
        
        # 使用 sigmoid 映射 std
        std_min, std_max = 0.05, 0.8
        std = std_min + (std_max - std_min) * torch.sigmoid(self.log_std)
        log_std = torch.log(std)
        log_std = log_std.expand(mean.shape[0], -1)
        
        # Critic 分支
        value = self.critic_net(x)
        
        mean = torch.nan_to_num(mean, nan=0.0)
        log_std = torch.nan_to_num(log_std, nan=0.0)
        value = torch.nan_to_num(value, nan=0.0)
        
        return mean, log_std, value, None
    
    def get_action(
        self, 
        state: torch.Tensor,
        hidden: Optional[Tuple] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """获取动作"""
        mean, log_std, value, _ = self.forward(state, hidden)
        std = log_std.exp()
        std = torch.clamp(std, min=1e-6, max=10.0)
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros(action.shape[0], 1, device=action.device)
        else:
            normal = Normal(mean, std)
            x = normal.rsample()
            action = torch.tanh(x)
            
            log_prob = normal.log_prob(x)
            action_safe = torch.clamp(action, -0.9999, 0.9999)
            log_prob = log_prob - torch.log(1 - action_safe.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            log_prob = torch.clamp(log_prob, min=-20.0, max=2.0)
        
        return action, log_prob, value, mean, None
    
    def evaluate_actions(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估给定状态-动作对"""
        mean, log_std, value, _ = self.forward(state)
        std = log_std.exp()
        std = torch.clamp(std, min=1e-6, max=10.0)
        
        normal = Normal(mean, std)
        
        action_clipped = torch.clamp(action, -0.9999, 0.9999)
        x = torch.atanh(action_clipped)
        
        log_prob = normal.log_prob(x)
        log_prob = log_prob - torch.log(1 - action_clipped.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob = torch.clamp(log_prob, min=-20.0, max=2.0)
        
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        entropy = torch.clamp(entropy, min=-10.0, max=10.0)
        
        return log_prob, value, entropy
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """获取状态价值"""
        state = torch.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)
        if state.dim() == 2:
            state = state.unsqueeze(0)
        
        x = self.input_projection(state)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        
        return self.critic_net(x)


class DiscreteTransformerActorCritic(nn.Module):
    """
    离散动作空间的 Transformer Actor-Critic 网络
    
    动作空间: 3 个离散动作
    - 0: 卖出清仓 (0%)
    - 1: 持有不变
    - 2: 买入满仓 (100%)
    
    输入格式: (batch, seq_len, n_features)
    """
    
    def __init__(
        self,
        input_dim: int,           # 每个时间步的特征数
        action_dim: int = 3,      # 离散动作数量
        hidden_dims: List[int] = [256, 128],
        critic_hidden_dims: Optional[List[int]] = None,
        d_model: int = 128,       # Transformer 隐藏维度
        nhead: int = 4,           # 注意力头数
        num_layers: int = 2,      # Transformer 层数
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 200
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.d_model = d_model
        
        if critic_hidden_dims is None:
            critic_hidden_dims = hidden_dims
        
        # 输入投影层
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 共享的后处理层
        shared_layers = []
        prev_dim = d_model
        for hidden_dim in hidden_dims[:-1] if len(hidden_dims) > 1 else []:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.LayerNorm(hidden_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.shared_net = nn.Sequential(*shared_layers) if shared_layers else nn.Identity()
        
        # Actor 头 - 输出动作 logits
        actor_hidden = hidden_dims[-1] if hidden_dims else d_model
        self.actor_head = nn.Sequential(
            nn.Linear(prev_dim, actor_hidden),
            nn.LayerNorm(actor_hidden),
            nn.ReLU(),
            nn.Linear(actor_hidden, action_dim)
        )
        
        # Critic 头 - 独立网络
        critic_layers = []
        critic_prev_dim = d_model
        for hidden_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(critic_prev_dim, hidden_dim))
            critic_layers.append(nn.LayerNorm(hidden_dim))
            critic_layers.append(nn.ReLU())
            critic_layers.append(nn.Dropout(dropout))
            critic_prev_dim = hidden_dim
        critic_layers.append(nn.Linear(critic_prev_dim, 1))
        self.critic_net = nn.Sequential(*critic_layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self, 
        state: torch.Tensor,
        hidden: Optional[Tuple] = None  # 保留接口兼容性
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """
        前向传播
        
        Returns:
            action_logits: 动作 logits (batch, action_dim)
            value: 状态价值 (batch, 1)
            None: 保持接口兼容
        """
        state = torch.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)
        
        if state.dim() == 2:
            state = state.unsqueeze(0)
        
        # 输入投影
        x = self.input_projection(state)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)
        
        # 取最后一个时间步
        x = x[:, -1, :]
        
        # Actor 分支
        actor_features = self.shared_net(x)
        action_logits = self.actor_head(actor_features)
        
        # Critic 分支
        value = self.critic_net(x)
        
        action_logits = torch.nan_to_num(action_logits, nan=0.0)
        value = torch.nan_to_num(value, nan=0.0)
        
        return action_logits, value, None
    
    def get_action(
        self, 
        state: torch.Tensor,
        hidden: Optional[Tuple] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        获取动作
        
        Returns:
            action: 离散动作 (batch,)
            log_prob: 对数概率 (batch, 1)
            value: 状态价值 (batch, 1)
            action_probs: 动作概率分布 (batch, action_dim)
            None: 保持接口兼容
        """
        action_logits, value, _ = self.forward(state, hidden)
        
        dist = Categorical(logits=action_logits)
        
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
            log_prob = dist.log_prob(action).unsqueeze(-1)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action).unsqueeze(-1)
        
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action, log_prob, value, action_probs, None
    
    def evaluate_actions(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估给定状态-动作对"""
        action_logits, value, _ = self.forward(state)
        
        if action.dim() > 1:
            action = action.squeeze(-1)
        action = action.long()
        
        dist = Categorical(logits=action_logits)
        
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        
        return log_prob, value, entropy
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """获取状态价值"""
        state = torch.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)
        if state.dim() == 2:
            state = state.unsqueeze(0)
        
        x = self.input_projection(state)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        
        return self.critic_net(x)
