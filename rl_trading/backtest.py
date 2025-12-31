#!/usr/bin/env python3
"""
回测脚本 - 使用训练好的模型测试任意单只股票

设计理念：
- 模型用多只股票训练，学习通用的交易模式
- 回测/使用时输入任意一只股票，返回买/卖/持有策略
- 动作空间固定为 1 维，与训练时使用的股票数量无关
"""

import argparse
import sys
import functools
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# 强制刷新输出
print = functools.partial(print, flush=True)

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rl_trading.config import EnvConfig, PPOConfig
from rl_trading.env import SingleStockEnv
from rl_trading.agent import PPOAgent
from rl_trading.features import FeatureEngineer
from src.data.data_fetcher import DataFetcher


class Backtester:
    """
    回测器 - 使用训练好的模型测试任意单只股票
    
    模型是单股票策略：
    - 输入：单只股票的技术指标
    - 输出：买/卖/持有动作 [-1, 1]
    """
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        初始化回测器
        
        Args:
            model_path: 模型文件路径
            device: 计算设备
        """
        self.model_path = Path(model_path)
        self.device = device
        self.data_fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer(normalize=True)
        self.agent: Optional[PPOAgent] = None
        
        # 模型维度
        self.model_state_dim: int = 0
        self.model_action_dim: int = 0
        
        # 回测结果
        self.results: Dict = {}
    
    def _detect_model_config(self) -> Tuple[int, int, str, bool]:
        """从模型文件检测维度、模型类型、是否离散动作
        
        Returns:
            (state_dim, action_dim, model_type, discrete_action)
            model_type: 'lstm', 'transformer', 或 'mlp'
        """
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # 检测模型类型和离散动作
        model_type = 'mlp'
        discrete_action = False
        
        if 'network_state_dict' in checkpoint:
            network_state = checkpoint['network_state_dict']
            
            # 检测模型类型
            for key in network_state.keys():
                if 'lstm' in key.lower():
                    model_type = 'lstm'
                    break
                if 'transformer' in key.lower() or 'pos_encoder' in key.lower():
                    model_type = 'transformer'
                    break
            
            # 检测是否离散动作（离散模型有 actor_head 而没有 log_std）
            has_actor_head = any('actor_head' in key for key in network_state.keys())
            has_log_std = any('log_std' in key for key in network_state.keys())
            if has_actor_head and not has_log_std:
                discrete_action = True
        
        # 优先从 checkpoint 中读取保存的配置
        if 'model_type' in checkpoint:
            model_type = checkpoint['model_type']
        
        # 从网络状态获取维度
        if 'network_state_dict' in checkpoint:
            network_state = checkpoint['network_state_dict']
            
            state_dim = None
            action_dim = None
            
            if model_type in ['lstm', 'transformer']:
                # LSTM/Transformer 模型：从 input_norm 或 input_proj/input_projection 获取输入维度
                for key, value in network_state.items():
                    if 'input_norm.weight' in key:
                        state_dim = value.shape[0]
                        break
                    if 'input_proj.weight' in key or 'input_projection.0.weight' in key:
                        # Transformer input_proj: (d_model, input_dim)
                        state_dim = value.shape[1]
                        break
                    if 'lstm.weight_ih_l0' in key:
                        # LSTM input size = weight_ih shape[1]
                        state_dim = value.shape[1]
                        break
                
                # 获取输出维度 - 找最后一层的输出
                if discrete_action:
                    # 离散动作：从 actor_head 最后一层获取
                    # 找到所有 actor_head 的 Linear weight 层（排除 LayerNorm）
                    # Linear weight 形状是 (out, in)，LayerNorm 形状是 (features,)
                    actor_head_weights = []
                    for key, value in network_state.items():
                        if 'actor_head' in key and 'weight' in key:
                            # 只取 2D 的 weight（Linear 层），排除 1D 的（LayerNorm）
                            if len(value.shape) == 2:
                                actor_head_weights.append((key, value))
                    if actor_head_weights:
                        # 按 key 中的数字排序，取最后一个（最大的层号）
                        actor_head_weights.sort(key=lambda x: int(x[0].split('.')[1]))
                        last_key, last_value = actor_head_weights[-1]
                        action_dim = last_value.shape[0]
                else:
                    # 连续动作：从 mean_layer 获取
                    for key, value in network_state.items():
                        if 'mean_layer.weight' in key:
                            action_dim = value.shape[0]
                            break
            else:
                # MLP 模型
                for key, value in network_state.items():
                    if 'shared.0.weight' in key or 'actor.0.weight' in key:
                        state_dim = value.shape[1]
                        break
                for key in network_state.keys():
                    if 'actor_mean' in key and 'weight' in key:
                        action_dim = network_state[key].shape[0]
                        break
            
            if state_dim is None or action_dim is None:
                # 尝试其他方式
                for key, value in network_state.items():
                    if 'weight' in key and len(value.shape) == 2:
                        if state_dim is None:
                            state_dim = value.shape[1]
                        action_dim = value.shape[0]
                
            return state_dim, action_dim, model_type, discrete_action
        
        raise ValueError("无法从模型文件检测维度")
        
    def load_model(self) -> None:
        """加载模型"""
        print(f"加载模型: {self.model_path}")
        
        # 检测模型配置
        state_dim, action_dim, model_type, discrete_action = self._detect_model_config()
        print(f"  模型状态维度: {state_dim}")
        print(f"  模型动作维度: {action_dim}")
        print(f"  模型类型: {model_type}")
        print(f"  离散动作: {discrete_action}")
        
        self.model_state_dim = state_dim
        self.model_action_dim = action_dim
        self.model_type = model_type
        self.use_lstm = model_type in ['lstm', 'transformer']  # 兼容旧代码
        self.discrete_action = discrete_action
        
        # 创建 PPO 配置
        ppo_config = PPOConfig()
        
        # 创建并加载模型
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
            model_type=model_type,
            use_lstm=(model_type == 'lstm'),
            lstm_hidden_size=ppo_config.lstm_hidden_size,
            num_lstm_layers=ppo_config.num_lstm_layers,
            hidden_dims=ppo_config.hidden_dims,
            critic_hidden_dims=ppo_config.critic_hidden_dims,
            discrete_action=discrete_action,
            # Transformer 配置
            d_model=ppo_config.d_model,
            nhead=ppo_config.nhead,
            num_transformer_layers=ppo_config.num_transformer_layers,
            dim_feedforward=ppo_config.dim_feedforward,
            dropout=ppo_config.dropout
        )
        self.agent.load(str(self.model_path))
        self.agent.set_training_mode(False)
        print("模型加载成功")
        
    def run_backtest(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_balance: float = 1_000_000,
        transaction_cost: float = 0.001,
        lookback_window: int = 20,
        full_position_mode: bool = False
    ) -> Dict:
        """
        运行单只股票回测
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            initial_balance: 初始资金
            transaction_cost: 交易成本
            lookback_window: 回看窗口
            full_position_mode: 满仓模式（True: 0-100%, False: 半仓模式）
            
        Returns:
            回测结果字典
        """
        print("=" * 60)
        print(f"回测配置")
        print("=" * 60)
        print(f"  股票: {symbol}")
        print(f"  时间: {start_date} ~ {end_date}")
        print(f"  初始资金: ${initial_balance:,.0f}")
        print(f"  交易成本: {transaction_cost:.2%}")
        print(f"  仓位模式: {'满仓 (0-100%)' if full_position_mode else '半仓 (action=0 时 50%)'}")
        
        # 1. 获取数据
        print("\n获取历史数据...")
        try:
            df = self.data_fetcher.fetch_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                use_cache=True
            )
            print(f"  {symbol}: {len(df)} 条数据")
        except Exception as e:
            raise ValueError(f"获取 {symbol} 数据失败: {e}")
        
        if len(df) == 0:
            raise ValueError(f"无法获取 {symbol} 数据")
        
        # 2. 计算特征
        print("\n计算技术指标...")
        processed_df = self.feature_engineer.compute_features(df, fit_scaler=True)
        print(f"  有效数据: {len(processed_df)} 条")
        print(f"  特征数量: {len(self.feature_engineer.get_feature_columns())}")
        
        # 3. 获取特征列 - 使用与训练时相同的核心特征
        feature_columns = self.feature_engineer.get_core_features()
        print(f"  使用核心特征: {len(feature_columns)} 个")
        
        # 4. 加载模型
        if self.agent is None:
            self.load_model()
        
        # 5. 创建回测环境
        print("\n创建回测环境...")
        
        # 检测是否离散动作模式
        discrete_action = getattr(self, 'discrete_action', False)
        
        env = SingleStockEnv(
            dfs={symbol: processed_df},
            feature_columns=feature_columns,
            initial_balance=initial_balance,
            transaction_cost=transaction_cost,
            lookback_window=lookback_window,
            training_mode=False,  # 推理模式
            full_position_mode=full_position_mode,
            discrete_action=discrete_action
        )
        
        print(f"  环境状态维度: {env.observation_space.shape}")
        if discrete_action:
            action_dim = getattr(self, 'model_action_dim', 2)
            if action_dim == 2:
                print(f"  环境动作空间: 离散 2动作 (0=空仓, 1=满仓)")
            else:
                print(f"  环境动作空间: 离散 {action_dim}动作")
        else:
            print(f"  环境动作维度: {env.action_space.shape}")
        
        # 验证维度匹配（LSTM/Transformer 模式下是 (lookback, features)）
        if self.model_type in ['lstm', 'transformer']:
            env_feature_dim = env.observation_space.shape[1]  # 每个时间步的特征数
            if env_feature_dim != self.model_state_dim:
                print(f"  警告: 环境特征维度 {env_feature_dim} 与模型 {self.model_state_dim} 不匹配")
        else:
            if env.observation_space.shape[0] != self.model_state_dim:
                print(f"  警告: 环境状态维度 {env.observation_space.shape[0]} 与模型 {self.model_state_dim} 不匹配")
        
        # 6. 运行回测
        print("\n开始回测...")
        state, _ = env.reset()
        done = False
        step = 0
        
        # 记录详细数据
        portfolio_values = []
        actions_history = []
        positions_history = []
        prices_history = []
        
        while not done:
            # 获取动作
            action, _, _ = self.agent.select_action(state, deterministic=True)
            
            # 执行动作
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
            # 记录（step 后的数据，prices 和 portfolio 一一对应）
            portfolio_values.append(info['portfolio_value'])
            prices_history.append(env._get_price())
            
            # 处理不同格式的 action
            discrete_action = getattr(self, 'discrete_action', False)
            if discrete_action:
                if isinstance(action, np.ndarray):
                    action_val = int(action.item()) if action.ndim == 0 else int(action[0])
                else:
                    action_val = int(action)
            else:
                if isinstance(action, np.ndarray):
                    action_val = float(action.item()) if action.ndim == 0 else float(action[0])
                else:
                    action_val = float(action)
            actions_history.append(action_val)
            positions_history.append(info.get('position_ratio', 0))
            
            # 打印进度
            if step % 50 == 0:
                print(f"  Step {step}: 资产 ${info['portfolio_value']:,.0f}, "
                      f"仓位 {info.get('position_ratio', 0):.1%}, "
                      f"收益 {info['total_return']:.2%}")
        
        # 7. 获取统计
        stats = env.get_episode_stats()
        
        # 8. 计算收益率 - 统一使用 portfolio_values 计算
        # portfolio_values[0] 是第一天收盘后的资产，prices[0] 是第一天收盘价
        if len(prices_history) > 0 and len(portfolio_values) > 0:
            # 策略收益：从 portfolio[0] 到 portfolio[-1]
            strategy_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            # 基准收益：从 prices[0] 到 prices[-1]（与策略同期）
            benchmark_return = (prices_history[-1] - prices_history[0]) / prices_history[0]
        else:
            strategy_return = 0.0
            benchmark_return = 0.0
        
        # 保存结果
        self.results = {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'initial_balance': initial_balance,
            'final_value': portfolio_values[-1] if portfolio_values else initial_balance,
            'total_return': strategy_return,  # 使用统一计算的策略收益
            'sharpe_ratio': stats['sharpe_ratio'],
            'max_drawdown': stats['max_drawdown'],
            'total_trades': stats['total_trades'],
            'win_rate': stats.get('win_rate', 0),
            'benchmark_return': benchmark_return,
            'excess_return': strategy_return - benchmark_return,
            'portfolio_values': portfolio_values,
            'actions_history': actions_history,
            'positions_history': positions_history,
            'prices_history': prices_history
        }
        
        return self.results
    
    def print_results(self) -> None:
        """打印回测结果"""
        if not self.results:
            print("尚未运行回测")
            return
        
        r = self.results
        print("\n" + "=" * 60)
        print("回测结果")
        print("=" * 60)
        print(f"股票: {r['symbol']}")
        print(f"时间: {r['start_date']} ~ {r['end_date']}")
        print("-" * 60)
        print(f"初始资金:     ${r['initial_balance']:>15,.0f}")
        print(f"最终资产:     ${r['final_value']:>15,.0f}")
        print(f"总收益率:     {r['total_return']:>15.2%}")
        print(f"基准收益率:   {r['benchmark_return']:>15.2%}")
        print(f"超额收益:     {r['excess_return']:>15.2%}")
        print("-" * 60)
        print(f"夏普比率:     {r['sharpe_ratio']:>15.2f}")
        print(f"最大回撤:     {r['max_drawdown']:>15.2%}")
        print(f"总交易次数:   {r['total_trades']:>15}")
        print(f"胜率:         {r['win_rate']:>15.2%}")
        print("=" * 60)
        
        # 判断策略表现
        if r['excess_return'] > 0:
            print(f"✓ 策略跑赢基准 {r['excess_return']:.2%}")
        else:
            print(f"✗ 策略跑输基准 {r['excess_return']:.2%}")
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """绘制回测结果图表，包含买卖点标注"""
        if not self.results:
            print("尚未运行回测")
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        
        prices = self.results['prices_history']
        positions = self.results['positions_history']
        actions = self.results['actions_history']
        portfolio_values = self.results['portfolio_values']
        
        # 检测买卖点（仓位变化超过阈值）
        buy_points = []
        sell_points = []
        threshold = 0.01  # 仓位变化超过 1% 视为交易
        
        # 检测第一个点（从 0 仓位开始）
        if len(positions) > 0 and positions[0] > threshold:
            buy_points.append(0)
        
        for i in range(1, len(positions)):
            pos_change = positions[i] - positions[i-1]
            if pos_change > threshold:
                buy_points.append(i)
            elif pos_change < -threshold:
                sell_points.append(i)
        
        # 1. 股价 + 买卖点
        ax1 = axes[0, 0]
        ax1.plot(prices, label='股价', linewidth=1.5, color='black')
        if buy_points:
            ax1.scatter(buy_points, [prices[i] for i in buy_points], 
                       color='red', marker='^', s=80, label=f'买入 ({len(buy_points)}次)', zorder=5)
        if sell_points:
            ax1.scatter(sell_points, [prices[i] for i in sell_points], 
                       color='green', marker='v', s=80, label=f'卖出 ({len(sell_points)}次)', zorder=5)
        ax1.set_title(f'{self.results["symbol"]} 股价与买卖点')
        ax1.set_xlabel('交易日')
        ax1.set_ylabel('股价 ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 资产曲线
        ax2 = axes[0, 1]
        ax2.plot(portfolio_values, label='策略资产', linewidth=2, color='blue')
        ax2.axhline(y=self.results['initial_balance'], color='gray', linestyle='--', label='初始资金')
        ax2.set_title(f'策略资产曲线')
        ax2.set_xlabel('交易日')
        ax2.set_ylabel('资产价值 ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 股价 + 仓位（双 Y 轴）
        ax3 = axes[1, 0]
        ax3_twin = ax3.twinx()
        ax3.plot(prices, label='股价', linewidth=1.5, color='black')
        ax3_twin.fill_between(range(len(positions)), positions, alpha=0.3, color='blue')
        ax3_twin.plot(positions, color='blue', linewidth=1, label='仓位')
        ax3.set_title('股价与仓位变化')
        ax3.set_xlabel('交易日')
        ax3.set_ylabel('股价 ($)', color='black')
        ax3_twin.set_ylabel('仓位比例', color='blue')
        ax3_twin.set_ylim(-0.05, 1.05)
        ax3.grid(True, alpha=0.3)
        
        # 4. 收益率对比
        ax4 = axes[1, 1]
        # portfolio_values 和 prices 现在一一对应
        strategy_returns = [(v / portfolio_values[0] - 1) * 100 for v in portfolio_values]
        benchmark_returns = [(p / prices[0] - 1) * 100 for p in prices]
        ax4.plot(strategy_returns, label='策略收益', color='blue', linewidth=2)
        ax4.plot(benchmark_returns, label='Buy & Hold', color='orange', linewidth=2)
        ax4.axhline(y=0, color='gray', linestyle='--')
        ax4.set_title('累计收益率对比')
        ax4.set_xlabel('交易日')
        ax4.set_ylabel('收益率 (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 动作时序图
        ax5 = axes[2, 0]
        discrete_action = getattr(self, 'discrete_action', False)
        if discrete_action:
            # 离散动作：绘制阶梯图
            ax5.step(range(len(actions)), actions, where='mid', linewidth=1.5, color='purple')
            ax5.set_yticks([0, 1, 2])
            ax5.set_yticklabels(['卖出', '持有', '买入'])
            ax5.set_title('离散动作时序')
            ax5.set_ylim(-0.5, 2.5)
        else:
            ax5.plot(actions, linewidth=0.8, color='purple', alpha=0.7)
            ax5.axhline(y=0, color='gray', linestyle='--')
            ax5.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='买入倾向')
            ax5.axhline(y=-0.5, color='green', linestyle=':', alpha=0.5, label='卖出倾向')
            ax5.set_title('动作值时序')
            ax5.set_ylim(-1.1, 1.1)
            ax5.legend()
        ax5.set_xlabel('交易日')
        ax5.set_ylabel('动作')
        ax5.grid(True, alpha=0.3)
        
        # 6. 动作分布
        ax6 = axes[2, 1]
        if discrete_action:
            # 离散动作：绘制柱状图
            action_counts = [actions.count(i) for i in range(3)]
            bars = ax6.bar(['卖出 (0)', '持有 (1)', '买入 (2)'], action_counts, 
                          color=['green', 'gray', 'red'], alpha=0.7, edgecolor='black')
            ax6.set_title('动作分布')
            ax6.set_ylabel('次数')
            # 在柱子上显示数量
            for bar, count in zip(bars, action_counts):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        str(count), ha='center', va='bottom')
        else:
            ax6.hist(actions, bins=50, alpha=0.7, color='purple', edgecolor='black')
            ax6.axvline(x=0, color='red', linestyle='--', label='中性')
            ax6.axvline(x=np.mean(actions), color='blue', linestyle='-', linewidth=2, label=f'均值: {np.mean(actions):.3f}')
            ax6.set_title('动作分布')
            ax6.set_xlabel('动作值 (-1=清仓, 0=半仓, 1=满仓)')
            ax6.set_ylabel('频次')
            ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.results["symbol"]} 回测分析 ({self.results["start_date"]} ~ {self.results["end_date"]})\n'
                     f'策略: {self.results["total_return"]:.2%} | Buy&Hold: {self.results["benchmark_return"]:.2%} | '
                     f'夏普: {self.results["sharpe_ratio"]:.2f} | 最大回撤: {self.results["max_drawdown"]:.2%}', 
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def export_trades(self, output_path: str) -> None:
        """导出交易记录"""
        if not self.results:
            print("尚未运行回测")
            return
        
        # 构建交易记录 DataFrame
        records = []
        for i in range(len(self.results['actions_history'])):
            record = {
                'step': i,
                'action': self.results['actions_history'][i],
                'position': self.results['positions_history'][i],
                'price': self.results['prices_history'][i],
                'portfolio_value': self.results['portfolio_values'][i + 1]
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        print(f"交易记录已导出到: {output_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用训练好的模型回测单只股票')
    
    parser.add_argument('--model', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--symbol', type=str, required=True,
                       help='股票代码')
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                       help='开始日期')
    parser.add_argument('--end-date', type=str, default='2024-12-01',
                       help='结束日期')
    parser.add_argument('--initial-balance', type=float, default=1_000_000,
                       help='初始资金')
    parser.add_argument('--transaction-cost', type=float, default=0.001,
                       help='交易成本')
    parser.add_argument('--lookback-window', type=int, default=20,
                       help='回看窗口')
    parser.add_argument('--device', type=str, default='cpu',
                       help='计算设备')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='保存图表路径')
    parser.add_argument('--export-trades', type=str, default=None,
                       help='导出交易记录路径')
    parser.add_argument('--full-position', action='store_true',
                       help='使用满仓模式 (action 直接映射到 0-100%%)')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建回测器
    backtester = Backtester(
        model_path=args.model,
        device=args.device
    )
    
    # 运行回测
    results = backtester.run_backtest(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_balance=args.initial_balance,
        transaction_cost=args.transaction_cost,
        lookback_window=args.lookback_window,
        full_position_mode=args.full_position
    )
    
    # 打印结果
    backtester.print_results()
    
    # 绘制图表
    backtester.plot_results(save_path=args.save_plot)
    
    # 导出交易记录
    if args.export_trades:
        backtester.export_trades(args.export_trades)


if __name__ == '__main__':
    main()
