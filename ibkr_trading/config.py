"""
IBKR 交易配置
"""

# TWS/Gateway 连接配置
TWS_HOST = "127.0.0.1"
TWS_PORT = 7497  # TWS Paper Trading: 7497, TWS Live: 7496, Gateway Paper: 4002, Gateway Live: 4001
CLIENT_ID = 1

# 交易配置
DEFAULT_CURRENCY = "USD"
DEFAULT_EXCHANGE = "SMART"

# 风控配置
MAX_POSITION_SIZE = 0.1  # 单只股票最大仓位比例
MAX_DAILY_LOSS = 0.02    # 日最大亏损比例
MAX_ORDER_VALUE = 10000  # 单笔订单最大金额

# 日志配置
LOG_LEVEL = "INFO"
LOG_FILE = "ibkr_trading.log"
