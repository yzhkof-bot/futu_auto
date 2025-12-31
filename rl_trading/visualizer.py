"""
训练过程可视化模块
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class TrainingVisualizer:
    """训练过程实时可视化"""
    
    def __init__(self, save_dir: str = "rl_trading/logs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def plot_training_curves(
        self,
        history: Dict[str, List],
        save_path: Optional[str] = None
    ):
        """
        绘制训练曲线
        
        Args:
            history: 训练历史数据
            save_path: 保存路径（可选）
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=14, fontweight='bold')
        
        episodes = range(1, len(history.get('entropy', [])) + 1)
        
        # 1. Entropy 曲线
        ax = axes[0, 0]
        if history.get('entropy'):
            ax.plot(episodes, history['entropy'], 'b-', linewidth=1.5, label='Entropy')
            ax.axhline(y=history['entropy'][0], color='gray', linestyle='--', alpha=0.5, label='Initial')
            # 添加移动平均
            if len(history['entropy']) >= 10:
                ma = np.convolve(history['entropy'], np.ones(10)/10, mode='valid')
                ax.plot(range(10, len(history['entropy'])+1), ma, 'r-', linewidth=2, alpha=0.7, label='MA(10)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Entropy')
        ax.set_title('Policy Entropy')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 2. Episode Return 曲线
        ax = axes[0, 1]
        if history.get('episode_returns'):
            returns = [r * 100 for r in history['episode_returns']]  # 转换为百分比
            colors = ['green' if r >= 0 else 'red' for r in returns]
            ax.bar(episodes, returns, color=colors, alpha=0.6, width=0.8)
            # 添加移动平均
            if len(returns) >= 10:
                ma = np.convolve(returns, np.ones(10)/10, mode='valid')
                ax.plot(range(10, len(returns)+1), ma, 'b-', linewidth=2, label='MA(10)')
                ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return (%)')
        ax.set_title('Episode Total Return')
        ax.grid(True, alpha=0.3)
        
        # 3. Test Return 曲线
        ax = axes[0, 2]
        if history.get('eval_returns'):
            eval_returns = [r * 100 for r in history['eval_returns']]  # 转换为百分比
            eval_episodes = np.linspace(1, len(history['entropy']), len(eval_returns))
            ax.plot(eval_episodes, eval_returns, 'go-', linewidth=2, markersize=6, label='Test Return')
            # 标记最佳
            best_idx = np.argmax(eval_returns)
            ax.scatter([eval_episodes[best_idx]], [eval_returns[best_idx]], 
                      color='red', s=100, zorder=5, label=f'Best: {eval_returns[best_idx]:.1f}%')
            ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return (%)')
        ax.set_title('Evaluation Return')
        ax.grid(True, alpha=0.3)
        
        # 4. Policy Loss 曲线
        ax = axes[1, 0]
        if history.get('policy_loss'):
            ax.plot(episodes, history['policy_loss'], 'purple', linewidth=1, alpha=0.6)
            if len(history['policy_loss']) >= 10:
                ma = np.convolve(history['policy_loss'], np.ones(10)/10, mode='valid')
                ax.plot(range(10, len(history['policy_loss'])+1), ma, 'purple', linewidth=2, label='MA(10)')
                ax.legend()
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title('Policy Loss')
        ax.grid(True, alpha=0.3)
        
        # 5. Value Loss 曲线
        ax = axes[1, 1]
        if history.get('value_loss'):
            ax.plot(episodes, history['value_loss'], 'orange', linewidth=1, alpha=0.6)
            if len(history['value_loss']) >= 10:
                ma = np.convolve(history['value_loss'], np.ones(10)/10, mode='valid')
                ax.plot(range(10, len(history['value_loss'])+1), ma, 'orange', linewidth=2, label='MA(10)')
                ax.legend()
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title('Value Loss')
        ax.grid(True, alpha=0.3)
        
        # 6. Sharpe Ratio 曲线
        ax = axes[1, 2]
        if history.get('episode_sharpe'):
            sharpe = history['episode_sharpe']
            ax.plot(episodes, sharpe, 'teal', linewidth=1, alpha=0.6)
            if len(sharpe) >= 10:
                ma = np.convolve(sharpe, np.ones(10)/10, mode='valid')
                ax.plot(range(10, len(sharpe)+1), ma, 'teal', linewidth=2, label='MA(10)')
                ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe=1')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Episode Sharpe Ratio')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_path = self.save_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_summary(self, history: Dict[str, List], save_path: Optional[str] = None):
        """绘制训练摘要（单图）"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = range(1, len(history.get('entropy', [])) + 1)
        
        # 双 Y 轴
        ax2 = ax.twinx()
        
        # 左轴：Entropy
        if history.get('entropy'):
            line1, = ax.plot(episodes, history['entropy'], 'b-', linewidth=2, label='Entropy')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Entropy', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # 右轴：Return MA
        if history.get('episode_returns') and len(history['episode_returns']) >= 10:
            returns = [r * 100 for r in history['episode_returns']]  # 转换为百分比
            ma = np.convolve(returns, np.ones(10)/10, mode='valid')
            line2, = ax2.plot(range(10, len(returns)+1), ma, 'g-', linewidth=2, label='Return MA(10)')
        ax2.set_ylabel('Return (%)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 图例
        lines = [line1, line2] if 'line2' in dir() else [line1]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        ax.set_title('Training Summary: Entropy vs Return')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / 'training_summary.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
