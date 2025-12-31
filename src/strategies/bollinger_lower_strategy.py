"""
布林带下轨策略
买入条件：价格低于布林带下轨1%以上
卖出条件：止损10% / 止盈10%
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .base_strategy import BaseStrategy


class BollingerLowerStrategy(BaseStrategy):
    """
    布林带下轨策略
    
    买入信号：价格低于布林带下轨1%以上（超卖反弹机会）
    卖出信号：
    - 止损：亏损10%
    - 止盈：盈利10%
    """
    
    def __init__(self, 
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 below_threshold: float = 0.01,  # 低于下轨1%
                 stop_loss_pct: float = 0.10,    # 止损10%
                 take_profit_pct: float = 0.10,  # 止盈10%
                 **kwargs):
        """
        初始化策略
        
        Args:
            bb_period: 布林带周期
            bb_std: 布林带标准差倍数
            below_threshold: 低于下轨的阈值（百分比）
            stop_loss_pct: 止损比例
            take_profit_pct: 止盈比例
        """
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.below_threshold = below_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        super().__init__(name="BollingerLowerStrategy", **kwargs)
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        
        df = data.copy()
        
        # 确保有布林带指标
        bb_lower_col = f'BB_Lower_{self.bb_period}'
        bb_upper_col = f'BB_Upper_{self.bb_period}'
        bb_middle_col = f'BB_Middle_{self.bb_period}'
        
        # 如果没有布林带指标，手动计算
        if bb_lower_col not in df.columns:
            df[bb_middle_col] = df['Close'].rolling(self.bb_period).mean()
            rolling_std = df['Close'].rolling(self.bb_period).std()
            df[bb_upper_col] = df[bb_middle_col] + self.bb_std * rolling_std
            df[bb_lower_col] = df[bb_middle_col] - self.bb_std * rolling_std
        
        # 计算价格相对于下轨的偏离程度
        df['Below_Lower_Pct'] = (df[bb_lower_col] - df['Close']) / df[bb_lower_col]
        
        # 初始化信号列
        df['Signal'] = 0
        df['Position'] = 0
        
        # 买入条件：价格低于下轨1%以上
        buy_condition = df['Below_Lower_Pct'] >= self.below_threshold
        
        # 生成买入信号
        df.loc[buy_condition, 'Signal'] = 1
        
        return df
    
    def get_signal_with_metadata(self, data: pd.DataFrame, current_idx: int) -> Tuple[int, Dict]:
        """
        获取当前信号及元数据
        
        Returns:
            (signal, metadata) - signal: 1买入, -1卖出, 0持有
                                metadata: 包含止损止盈价格等信息
        """
        if current_idx < self.bb_period:
            return 0, {}
        
        current_row = data.iloc[current_idx]
        current_price = current_row['Close']
        
        bb_lower_col = f'BB_Lower_{self.bb_period}'
        
        # 检查是否有布林带数据
        if bb_lower_col not in data.columns:
            return 0, {}
        
        bb_lower = current_row[bb_lower_col]
        
        if pd.isna(bb_lower):
            return 0, {}
        
        # 计算偏离程度
        below_pct = (bb_lower - current_price) / bb_lower
        
        # 买入条件
        if below_pct >= self.below_threshold:
            metadata = {
                'entry_price': current_price,
                'stop_loss_price': current_price * (1 - self.stop_loss_pct),
                'take_profit_price': current_price * (1 + self.take_profit_pct),
                'below_lower_pct': below_pct * 100,
                'bb_lower': bb_lower,
                'reason': f'价格低于下轨 {below_pct*100:.2f}%'
            }
            return 1, metadata
        
        return 0, {}
    
    def check_exit_conditions(self, 
                              entry_price: float, 
                              current_price: float,
                              position_data: Dict = None) -> Tuple[bool, str]:
        """
        检查是否满足退出条件
        
        Returns:
            (should_exit, reason)
        """
        if entry_price <= 0:
            return False, ""
        
        pnl_pct = (current_price - entry_price) / entry_price
        
        # 止损
        if pnl_pct <= -self.stop_loss_pct:
            return True, f"止损 ({pnl_pct*100:.2f}%)"
        
        # 止盈
        if pnl_pct >= self.take_profit_pct:
            return True, f"止盈 ({pnl_pct*100:.2f}%)"
        
        return False, ""
    
    def get_parameters(self) -> Dict:
        """获取策略参数"""
        return {
            'bb_period': self.bb_period,
            'bb_std': self.bb_std,
            'below_threshold': self.below_threshold,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct
        }
    
    def _validate_parameters(self) -> bool:
        """验证策略参数"""
        if self.bb_period <= 0:
            return False
        if self.bb_std <= 0:
            return False
        if self.below_threshold < 0 or self.below_threshold > 1:
            return False
        if self.stop_loss_pct <= 0 or self.stop_loss_pct > 1:
            return False
        if self.take_profit_pct <= 0 or self.take_profit_pct > 1:
            return False
        return True
