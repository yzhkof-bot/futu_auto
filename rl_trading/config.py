"""
配置文件 - 定义所有超参数和常量
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class EnvConfig:
    """交易环境配置"""
    # MAG7 股票训练 - 增加数据多样性
    symbols: List[str] = field(default_factory=lambda: [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'
    ])
    
    # 数据参数 - 扩大数据范围
    start_date: str = "2015-01-01"  # META 2012年上市，留一些缓冲
    end_date: str = "2025-11-01"
    train_ratio: float = 0.85  # 训练集比例 (更多训练数据)
    
    # 交易参数
    initial_balance: float = 1_000_000.0  # 初始资金 100万
    max_position_per_stock: float = 1.0   # 单股策略：最大仓位 100%
    transaction_cost: float = 0.0001      # 交易成本 0.01%
    slippage: float = 0.0001              # 滑点 0.01%
    
    # 状态空间参数
    lookback_window: int = 60  # 回看窗口 60天（约3个月趋势）


@dataclass
class PPOConfig:
    """PPO 算法配置"""
    # 网络结构
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])  # MLP 层
    
    # 模型架构选择: "lstm" 或 "transformer"
    model_type: str = "transformer"  # 默认使用 Transformer
    
    # LSTM 配置（当 model_type="lstm" 时使用）
    use_lstm: bool = False  # 兼容旧代码，优先使用 model_type
    lstm_hidden_size: int = 128
    num_lstm_layers: int = 2
    
    # Transformer 配置（当 model_type="transformer" 时使用）
    d_model: int = 128            # Transformer 隐藏维度
    nhead: int = 4                # 注意力头数
    num_transformer_layers: int = 2  # Transformer 层数
    dim_feedforward: int = 256    # FFN 隐藏维度
    
    # 离散动作空间配置 - 简化为2动作
    discrete_action: bool = True  # 使用离散动作
    action_dim: int = 2  # 0=空仓, 1=满仓（去掉"持有"，强制模型做决策）
    
    # Critic 独立配置
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128])
    
    # PPO 超参数
    learning_rate: float = 1e-4           # Transformer 用较小学习率
    gamma: float = 0.95                   # 折扣因子
    gae_lambda: float = 0.95              # GAE lambda
    clip_epsilon: float = 0.2             # PPO clip 参数
    value_coef: float = 0.5               # 价值损失系数
    entropy_coef: float = 0.001           # 熵系数
    max_grad_norm: float = 0.5            # 梯度裁剪
    
    # 训练参数
    batch_size: int = 64
    n_epochs: int = 10
    n_steps: int = 2048
    
    # 学习率调度
    lr_schedule: str = "cosine"
    min_lr: float = 1e-6
    
    # Dropout
    dropout: float = 0.1
    
    # 设备
    device: str = field(default_factory=lambda: 
        "mps" if torch.backends.mps.is_available() 
        else "cuda" if torch.cuda.is_available() 
        else "cpu"
    )


@dataclass
class TrainConfig:
    """训练配置"""
    total_timesteps: int = 500_000    # 增加到 500k（状态空间变大需要更多训练）
    eval_freq: int = 10_000           # 评估频率
    save_freq: int = 50_000           # 保存频率
    log_freq: int = 2048              # 日志频率（与 n_steps 对齐）
    
    # 早停
    early_stop_patience: int = 30     # 增加耐心值
    early_stop_threshold: float = 0.0 # 早停阈值
    
    # 路径
    model_dir: str = "rl_trading/models"
    log_dir: str = "rl_trading/logs"
    
    # 随机种子
    seed: int = 42


@dataclass
class FeatureConfig:
    """特征工程配置"""
    # 技术指标
    use_macd: bool = True
    use_rsi: bool = True
    use_bollinger: bool = True
    use_atr: bool = True
    use_adx: bool = True
    use_obv: bool = True
    use_stochastic: bool = True
    use_williams_r: bool = True
    use_cci: bool = True
    use_momentum: bool = True
    
    # 移动平均
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    ema_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    
    # RSI 周期
    rsi_periods: List[int] = field(default_factory=lambda: [14, 21])
    
    # 归一化
    normalize_features: bool = True
    clip_features: float = 5.0  # 特征裁剪范围 [-5, 5]


# 默认配置实例
DEFAULT_ENV_CONFIG = EnvConfig()
DEFAULT_PPO_CONFIG = PPOConfig()
DEFAULT_TRAIN_CONFIG = TrainConfig()
DEFAULT_FEATURE_CONFIG = FeatureConfig()
