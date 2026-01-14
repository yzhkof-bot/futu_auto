"""
自动交易主程序
"""

import time
import logging
from typing import List, Optional

from .client import IBKRClient
from .strategies.base_strategy import BaseStrategy
from .utils.risk_manager import RiskManager
from .config import LOG_LEVEL, LOG_FILE

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AutoTrader:
    """自动交易器"""
    
    def __init__(self, strategy: BaseStrategy, symbols: List[str]):
        self.client = IBKRClient()
        self.strategy = strategy
        self.symbols = symbols
        self.risk_manager = RiskManager()
        self.running = False
        
    def start(self):
        """启动交易"""
        if not self.client.connect():
            logger.error("无法连接到 IBKR，交易终止")
            return
            
        # 获取账户信息
        summary = self.client.get_account_summary()
        account_value = float(summary.get('NetLiquidation', 0))
        self.risk_manager.set_initial_equity(account_value)
        logger.info(f"账户净值: ${account_value:,.2f}")
        
        self.running = True
        logger.info(f"开始交易，监控股票: {self.symbols}")
        
        try:
            self._run_loop()
        except KeyboardInterrupt:
            logger.info("收到中断信号，停止交易")
        finally:
            self.stop()
    
    def _run_loop(self):
        """交易主循环"""
        while self.running:
            # 检查风控
            if not self.risk_manager.can_trade():
                logger.warning("风控限制，暂停交易")
                time.sleep(60)
                continue
            
            # 遍历监控股票
            for symbol in self.symbols:
                self._process_symbol(symbol)
            
            # 等待下一个周期
            time.sleep(5)
    
    def _process_symbol(self, symbol: str):
        """处理单个股票"""
        try:
            # 获取历史数据
            bars = self.client.get_historical_data(symbol, "1 M", "1 day")
            if not bars:
                return
            
            # 生成信号
            signal = self.strategy.generate_signal(symbol, bars)
            if not signal or signal == "HOLD":
                return
            
            # 获取账户信息
            summary = self.client.get_account_summary()
            account_value = float(summary.get('NetLiquidation', 0))
            current_price = bars[-1].close
            
            # 计算下单数量
            quantity = self.strategy.calculate_position_size(
                symbol, signal, account_value, current_price
            )
            
            if quantity <= 0:
                return
            
            # 风控检查
            order_value = quantity * current_price
            if not self.risk_manager.check_order_value(order_value):
                return
            
            # 下单
            if signal == "BUY":
                self.client.place_market_order(symbol, quantity, "BUY")
            elif signal == "SELL":
                position = self.strategy.get_position(symbol)
                if position > 0:
                    sell_qty = min(quantity, position)
                    self.client.place_market_order(symbol, sell_qty, "SELL")
                    
        except Exception as e:
            logger.error(f"处理 {symbol} 时出错: {e}")
    
    def stop(self):
        """停止交易"""
        self.running = False
        self.client.disconnect()
        logger.info("交易已停止")


def main():
    """示例入口"""
    from .strategies.base_strategy import BaseStrategy
    
    # 这里需要实现具体策略
    # strategy = YourStrategy()
    # symbols = ["AAPL", "GOOGL", "MSFT"]
    # trader = AutoTrader(strategy, symbols)
    # trader.start()
    print("请实现具体策略后运行")


if __name__ == "__main__":
    main()
