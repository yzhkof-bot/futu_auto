"""
Portfolio management module for multi-asset backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class PortfolioPosition:
    """Portfolio position data class."""
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    weight: float

class Portfolio:
    """
    Portfolio manager for multi-asset strategies.
    """
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.total_value = initial_capital
        
    def add_position(self, symbol: str, quantity: float, price: float) -> None:
        """Add or update position."""
        if symbol in self.positions:
            # Update existing position
            pos = self.positions[symbol]
            total_cost = pos.quantity * pos.avg_price + quantity * price
            total_quantity = pos.quantity + quantity
            pos.avg_price = total_cost / total_quantity if total_quantity != 0 else 0
            pos.quantity = total_quantity
        else:
            # New position
            self.positions[symbol] = PortfolioPosition(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                market_value=quantity * price,
                unrealized_pnl=0.0,
                weight=0.0
            )
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update portfolio with current prices."""
        total_position_value = 0
        
        for symbol, position in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                total_position_value += position.market_value
        
        self.total_value = self.cash + total_position_value
        
        # Update weights
        for position in self.positions.values():
            position.weight = position.market_value / self.total_value if self.total_value > 0 else 0
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary."""
        return {
            'total_value': self.total_value,
            'cash': self.cash,
            'invested_value': sum(pos.market_value for pos in self.positions.values()),
            'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values()),
            'num_positions': len(self.positions),
            'cash_weight': self.cash / self.total_value if self.total_value > 0 else 0
        }