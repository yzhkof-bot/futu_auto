#!/usr/bin/env python3
"""
训练脚本 - 单股票 PPO 策略训练

所有配置统一在 config.py 中管理，命令行只提供少量覆盖选项
"""

import argparse
import sys
from pathlib import Path

# 强制刷新输出
import functools
print = functools.partial(print, flush=True)

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rl_trading.config import EnvConfig, PPOConfig, TrainConfig, FeatureConfig
from rl_trading.trainer import Trainer


def parse_args():
    """解析命令行参数 - 只提供少量覆盖选项"""
    parser = argparse.ArgumentParser(
        description='训练 RL 股票交易策略\n所有配置在 config.py 中管理',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 只保留最常用的覆盖选项
    parser.add_argument('--total-timesteps', type=int, default=None,
                       help='总训练步数 (覆盖 config.py)')
    parser.add_argument('--load-model', type=str, default=None,
                       help='加载已有模型继续训练')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子 (覆盖 config.py)')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("RL 股票交易策略训练")
    print("=" * 60)
    
    # 直接使用 config.py 的默认配置
    env_config = EnvConfig()
    ppo_config = PPOConfig()
    train_config = TrainConfig()
    feature_config = FeatureConfig()
    
    # 命令行覆盖（如果指定）
    if args.total_timesteps is not None:
        train_config.total_timesteps = args.total_timesteps
    if args.seed is not None:
        train_config.seed = args.seed
    
    # 打印配置
    print("\n配置信息 (来自 config.py):")
    print(f"  股票池: {env_config.symbols}")
    print(f"  数据范围: {env_config.start_date} ~ {env_config.end_date}")
    print(f"  初始资金: ${env_config.initial_balance:,.0f}")
    print(f"  单股最大仓位: {env_config.max_position_per_stock:.0%}")
    print(f"  回看窗口: {env_config.lookback_window} 天")
    print(f"  总训练步数: {train_config.total_timesteps:,}")
    print(f"  网络结构: {ppo_config.hidden_dims}")
    print(f"  学习率: {ppo_config.learning_rate} ({ppo_config.lr_schedule} decay)")
    print(f"  batch_size: {ppo_config.batch_size}")
    print(f"  n_steps: {ppo_config.n_steps}")
    print(f"  设备: {ppo_config.device}")
    print(f"  使用 LSTM: {ppo_config.use_lstm}")
    
    # 创建训练器
    trainer = Trainer(
        env_config=env_config,
        ppo_config=ppo_config,
        train_config=train_config,
        feature_config=feature_config
    )
    
    # 准备数据和环境
    trainer.prepare_data()
    trainer.create_environments()
    trainer.create_agent()
    
    # 加载已有模型继续训练
    if args.load_model:
        print(f"\n加载模型继续训练: {args.load_model}")
        trainer.agent.load(args.load_model)
    
    # 开始训练
    try:
        results = trainer.train()
        
        print("\n" + "=" * 60)
        print("训练结果")
        print("=" * 60)
        print(f"总步数: {results['total_timesteps']:,}")
        print(f"总回合: {results['total_episodes']}")
        print(f"总时间: {results['total_time']/60:.1f} 分钟")
        print(f"最佳测试收益: {results['best_eval_return']:.2%}")
        print(f"最终训练收益: {results['final_train_return']:.2%}")
        
        print("\n" + trainer.get_training_summary())
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        trainer.save_model('interrupted_model.pt')
        print("已保存中断时的模型")
    
    except Exception as e:
        print(f"\n训练出错: {e}")
        raise


if __name__ == '__main__':
    main()
