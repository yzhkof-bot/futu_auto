"""
策略基类
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """交易策略基类"""
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.positions: Dict[str, int] = {}  # symbol -> quantity
        
    @abstractmethod
    def generate_signal(self, symbol: str, data: Any) -> Optional[str]:
        """
        生成交易信号
        返回: "BUY", "SELL", "HOLD" 或 None
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, symbol: str, signal: str, 
                                account_value: float, current_price: float) -> int:
        """计算下单数量"""
        pass
    
    def on_bar(self, symbol: str, bar: Any):
        """K线更新回调"""
        pass
    
    def on_tick(self, symbol: str, tick: Any):
        """Tick更新回调"""
        pass
    
    def on_order_filled(self, symbol: str, quantity: int, price: float, action: str):
        """订单成交回调"""
        if action == "BUY":
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        else:
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity
        logger.info(f"{self.name}: {action} {quantity} {symbol} @ {price}")
    
    def get_position(self, symbol: str) -> int:
        """获取持仓数量"""
        return self.positions.get(symbol, 0)
