"""
风险管理模块
"""

from typing import Dict, Optional
import logging

from ..config import MAX_POSITION_SIZE, MAX_DAILY_LOSS, MAX_ORDER_VALUE

logger = logging.getLogger(__name__)


class RiskManager:
    """风险管理器"""
    
    def __init__(self, max_position_size: float = MAX_POSITION_SIZE,
                 max_daily_loss: float = MAX_DAILY_LOSS,
                 max_order_value: float = MAX_ORDER_VALUE):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_order_value = max_order_value
        self.daily_pnl = 0.0
        self.initial_equity = 0.0
        
    def set_initial_equity(self, equity: float):
        """设置初始权益"""
        self.initial_equity = equity
        
    def update_daily_pnl(self, pnl: float):
        """更新日内盈亏"""
        self.daily_pnl = pnl
        
    def check_daily_loss_limit(self) -> bool:
        """检查是否超过日亏损限制"""
        if self.initial_equity <= 0:
            return True
        loss_ratio = -self.daily_pnl / self.initial_equity
        if loss_ratio >= self.max_daily_loss:
            logger.warning(f"触发日亏损限制: {loss_ratio:.2%} >= {self.max_daily_loss:.2%}")
            return False
        return True
    
    def check_position_limit(self, symbol: str, current_value: float, 
                            account_value: float) -> bool:
        """检查是否超过单股持仓限制"""
        if account_value <= 0:
            return False
        position_ratio = current_value / account_value
        if position_ratio >= self.max_position_size:
            logger.warning(f"{symbol} 持仓比例超限: {position_ratio:.2%} >= {self.max_position_size:.2%}")
            return False
        return True
    
    def check_order_value(self, order_value: float) -> bool:
        """检查订单金额是否超限"""
        if order_value > self.max_order_value:
            logger.warning(f"订单金额超限: {order_value} > {self.max_order_value}")
            return False
        return True
    
    def calculate_max_quantity(self, symbol: str, price: float, 
                               account_value: float, current_position_value: float = 0) -> int:
        """计算最大可买数量"""
        # 基于持仓限制
        max_position_value = account_value * self.max_position_size
        available_value = max_position_value - current_position_value
        
        # 基于单笔订单限制
        available_value = min(available_value, self.max_order_value)
        
        if available_value <= 0 or price <= 0:
            return 0
            
        return int(available_value / price)
    
    def can_trade(self) -> bool:
        """检查是否可以交易"""
        return self.check_daily_loss_limit()
