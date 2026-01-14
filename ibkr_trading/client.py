"""
IBKR API 客户端封装
使用 ib_insync 库连接 TWS/Gateway
"""

from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder
from typing import Optional, List, Callable
import logging

from .config import TWS_HOST, TWS_PORT, CLIENT_ID, DEFAULT_EXCHANGE, DEFAULT_CURRENCY

logger = logging.getLogger(__name__)


class IBKRClient:
    """IBKR API 客户端"""
    
    def __init__(self, host: str = TWS_HOST, port: int = TWS_PORT, client_id: int = CLIENT_ID):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self._tick_callbacks: dict = {}  # symbol -> callback
        
    def connect(self) -> bool:
        """连接到 TWS/Gateway"""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            logger.info(f"成功连接到 IBKR: {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("已断开 IBKR 连接")
    
    def get_account_summary(self) -> dict:
        """获取账户摘要"""
        summary = {}
        for item in self.ib.accountSummary():
            summary[item.tag] = item.value
        return summary
    
    def get_positions(self) -> List:
        """获取当前持仓"""
        return self.ib.positions()
    
    def get_portfolio(self) -> List:
        """获取投资组合"""
        return self.ib.portfolio()
    
    def create_stock_contract(self, symbol: str, exchange: str = DEFAULT_EXCHANGE, 
                              currency: str = DEFAULT_CURRENCY) -> Stock:
        """创建股票合约"""
        return Stock(symbol, exchange, currency)
    
    def get_market_data(self, contract: Stock) -> Optional[dict]:
        """获取实时行情 (快照模式, 延时 300-500ms)"""
        self.ib.qualifyContracts(contract)
        ticker = self.ib.reqMktData(contract)
        self.ib.sleep(1)  # 等待数据
        return {
            'bid': ticker.bid,
            'ask': ticker.ask,
            'last': ticker.last,
            'volume': ticker.volume
        }
    
    def subscribe_tick_by_tick(self, symbol: str, callback: Callable, 
                                tick_type: str = "Last") -> bool:
        """
        订阅逐笔数据 (低延时模式, 50-200ms)
        
        Args:
            symbol: 股票代码
            callback: 回调函数, 签名: callback(symbol, time, price, size)
            tick_type: "Last" (最新成交), "AllLast" (含场外), "BidAsk" (买卖盘)
        
        Returns:
            是否订阅成功
        """
        contract = self.create_stock_contract(symbol)
        self.ib.qualifyContracts(contract)
        
        # 存储回调
        self._tick_callbacks[symbol] = callback
        
        # 订阅逐笔数据
        ticker = self.ib.reqTickByTickData(contract, tick_type)
        
        # 设置事件处理
        if tick_type in ("Last", "AllLast"):
            ticker.updateEvent += lambda t: self._on_tick_last(symbol, t)
        else:  # BidAsk
            ticker.updateEvent += lambda t: self._on_tick_bidask(symbol, t)
        
        logger.info(f"订阅逐笔数据: {symbol} ({tick_type})")
        return True
    
    def _on_tick_last(self, symbol: str, ticker):
        """处理 Last/AllLast 逐笔数据"""
        if symbol in self._tick_callbacks and ticker.last is not None:
            try:
                self._tick_callbacks[symbol](
                    symbol=symbol,
                    time=ticker.time,
                    price=ticker.last,
                    size=ticker.lastSize
                )
            except Exception as e:
                logger.error(f"逐笔回调错误 {symbol}: {e}")
    
    def _on_tick_bidask(self, symbol: str, ticker):
        """处理 BidAsk 逐笔数据"""
        if symbol in self._tick_callbacks:
            try:
                self._tick_callbacks[symbol](
                    symbol=symbol,
                    time=ticker.time,
                    bid=ticker.bid,
                    ask=ticker.ask,
                    bid_size=ticker.bidSize,
                    ask_size=ticker.askSize
                )
            except Exception as e:
                logger.error(f"逐笔回调错误 {symbol}: {e}")
    
    def unsubscribe_tick_by_tick(self, symbol: str):
        """取消逐笔数据订阅"""
        if symbol in self._tick_callbacks:
            del self._tick_callbacks[symbol]
            logger.info(f"取消逐笔订阅: {symbol}")
    
    def place_market_order(self, symbol: str, quantity: int, action: str = "BUY") -> Optional[str]:
        """
        下市价单
        action: "BUY" 或 "SELL"
        """
        contract = self.create_stock_contract(symbol)
        self.ib.qualifyContracts(contract)
        order = MarketOrder(action, quantity)
        trade = self.ib.placeOrder(contract, order)
        logger.info(f"下单: {action} {quantity} {symbol}")
        return trade
    
    def place_limit_order(self, symbol: str, quantity: int, limit_price: float, 
                          action: str = "BUY") -> Optional[str]:
        """下限价单"""
        contract = self.create_stock_contract(symbol)
        self.ib.qualifyContracts(contract)
        order = LimitOrder(action, quantity, limit_price)
        trade = self.ib.placeOrder(contract, order)
        logger.info(f"下单: {action} {quantity} {symbol} @ {limit_price}")
        return trade
    
    def place_stop_order(self, symbol: str, quantity: int, stop_price: float,
                         action: str = "SELL") -> Optional[str]:
        """下止损单"""
        contract = self.create_stock_contract(symbol)
        self.ib.qualifyContracts(contract)
        order = StopOrder(action, quantity, stop_price)
        trade = self.ib.placeOrder(contract, order)
        logger.info(f"止损单: {action} {quantity} {symbol} @ {stop_price}")
        return trade
    
    def cancel_order(self, trade) -> bool:
        """取消订单"""
        self.ib.cancelOrder(trade.order)
        logger.info(f"取消订单: {trade.order.orderId}")
        return True
    
    def get_open_orders(self) -> List:
        """获取未成交订单"""
        return self.ib.openOrders()
    
    def get_historical_data(self, symbol: str, duration: str = "1 M", 
                            bar_size: str = "1 day") -> List:
        """
        获取历史数据
        duration: "1 D", "1 W", "1 M", "1 Y"
        bar_size: "1 min", "5 mins", "1 hour", "1 day"
        """
        contract = self.create_stock_contract(symbol)
        self.ib.qualifyContracts(contract)
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True
        )
        return bars
