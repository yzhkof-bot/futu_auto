#!/usr/bin/env python3
"""
快速启动脚本 - 用于验证环境和快速训练
"""

import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_dependencies():
    """检查依赖是否安装"""
    print("检查依赖...")
    missing = []
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        if torch.backends.mps.is_available():
            print(f"    - MPS (Apple Silicon) 可用")
        elif torch.cuda.is_available():
            print(f"    - CUDA 可用")
        else:
            print(f"    - 使用 CPU")
    except ImportError:
        missing.append("torch")
        print("  ✗ PyTorch 未安装")
    
    try:
        import gymnasium
        print(f"  ✓ Gymnasium {gymnasium.__version__}")
    except ImportError:
        missing.append("gymnasium")
        print("  ✗ Gymnasium 未安装")
    
    try:
        import sklearn
        print(f"  ✓ scikit-learn {sklearn.__version__}")
    except ImportError:
        missing.append("scikit-learn")
        print("  ✗ scikit-learn 未安装")
    
    try:
        import numpy
        print(f"  ✓ NumPy {numpy.__version__}")
    except ImportError:
        missing.append("numpy")
        print("  ✗ NumPy 未安装")
    
    try:
        import pandas
        print(f"  ✓ Pandas {pandas.__version__}")
    except ImportError:
        missing.append("pandas")
        print("  ✗ Pandas 未安装")
    
    try:
        import yfinance
        print(f"  ✓ yfinance {yfinance.__version__}")
    except ImportError:
        missing.append("yfinance")
        print("  ✗ yfinance 未安装")
    
    try:
        import matplotlib
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
    except ImportError:
        missing.append("matplotlib")
        print("  ✗ Matplotlib 未安装")
    
    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        print("请运行以下命令安装:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n所有依赖已安装 ✓")
    return True


def quick_train():
    """快速训练演示 - 单只股票，少量步数"""
    print("\n" + "=" * 60)
    print("快速训练演示 (AAPL, 50000 步)")
    print("=" * 60)
    
    from rl_trading.config import EnvConfig, PPOConfig, TrainConfig, FeatureConfig
    from rl_trading.trainer import Trainer
    
    # 简化配置 - 只用一只股票，快速验证
    env_config = EnvConfig(
        symbols=['AAPL'],  # 只用 AAPL
        start_date='2020-01-01',
        end_date='2024-01-01',
        initial_balance=100_000.0,
        max_position_per_stock=1.0  # 单只股票可以满仓
    )
    
    ppo_config = PPOConfig(
        hidden_dims=[128, 128],  # 更小的网络
        learning_rate=3e-4,
        n_epochs=5,
        batch_size=32
    )
    
    train_config = TrainConfig(
        total_timesteps=50_000,  # 快速训练
        eval_freq=5_000,
        save_freq=10_000,
        log_freq=1_000
    )
    
    # 创建训练器
    trainer = Trainer(
        env_config=env_config,
        ppo_config=ppo_config,
        train_config=train_config
    )
    
    # 训练
    results = trainer.train()
    
    print("\n" + trainer.get_training_summary())
    
    return results


def main():
    print("=" * 60)
    print("RL Trading 快速启动")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 询问是否开始训练
    print("\n准备开始快速训练演示...")
    print("这将训练一个简单的单股票策略 (AAPL)")
    print("预计耗时: 5-15 分钟")
    
    response = input("\n是否开始? [y/N]: ").strip().lower()
    
    if response == 'y':
        quick_train()
    else:
        print("\n你可以通过以下命令手动启动训练:")
        print("\n1. 快速训练 (单股票):")
        print("   python -m rl_trading.train --symbols AAPL --total-timesteps 50000")
        print("\n2. 完整训练 (MAG7):")
        print("   python -m rl_trading.train")
        print("\n3. 评估模型:")
        print("   python -m rl_trading.evaluate --model rl_trading/models/best_model.pt")
        print("\n4. 回测:")
        print("   python -m rl_trading.backtest --model rl_trading/models/best_model.pt")


if __name__ == '__main__':
    main()
