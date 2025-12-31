"""
RL Trading - 强化学习股票策略训练框架
基于 PPO 算法，支持 MAG7 股票池训练

使用方法:
---------
1. 快速训练:
   python -m rl_trading.train --total-timesteps 100000

2. 完整训练 (MAG7):
   python -m rl_trading.train

3. 评估模型:
   python -m rl_trading.evaluate --model rl_trading/models/best_model.pt

4. 回测:
   python -m rl_trading.backtest --model rl_trading/models/best_model.pt

配置说明:
---------
- config.py: 所有超参数配置
- EnvConfig: 交易环境配置 (股票池、资金、仓位限制等)
- PPOConfig: PPO 算法配置 (学习率、网络结构等)
- TrainConfig: 训练配置 (步数、保存频率等)
- FeatureConfig: 特征工程配置 (技术指标选择)
"""

__version__ = "0.1.0"
__author__ = "FUTU_auto"

from .env import TradingEnv, MultiStockEnv
from .agent import PPOAgent
from .trainer import Trainer
from .config import EnvConfig, PPOConfig, TrainConfig, FeatureConfig

__all__ = [
    'TradingEnv', 
    'MultiStockEnv',
    'PPOAgent', 
    'Trainer',
    'EnvConfig',
    'PPOConfig', 
    'TrainConfig',
    'FeatureConfig'
]
