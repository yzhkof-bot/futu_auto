import backtrader as bt
import yfinance as yf
import pandas as pd
import datetime

# --- 1. 定义策略 ---
class TrendFilterStrategy(bt.Strategy):
    """
    趋势动量策略 v3.0
    
    核心思路:
    1. 趋势确认: 价格在长期均线上方 + 短期均线向上
    2. 动量入场: 等待回调到短期均线附近再入场 (更好的入场点)
    3. 波动过滤: 用ATR过滤震荡行情
    4. 动态止损: ATR止损 + 移动止盈保护利润
    5. 趋势跟踪: 只要趋势不破就持有
    """
    
    params = (
        # 趋势判断
        ('fast_period', 10),    # 短期均线 (回调参考)
        ('mid_period', 20),     # 中期均线 (趋势方向)
        ('slow_period', 50),    # 长期均线 (趋势过滤)
        # ATR参数
        ('atr_period', 14),
        ('atr_stop_mult', 1.5), # 止损: 1.5倍ATR
        # 移动止盈
        ('trailing_pct', 0.12), # 从最高点回撤12%止盈
        # 入场过滤
        ('pullback_pct', 0.02), # 回调到短期均线2%范围内
        # 仓位
        ('risk_percent', 0.95),
        # 日志
        ('print_log', False),
        ('print_trades', False),
    )

    def __init__(self):
        # 三均线系统
        self.fast_ma = bt.indicators.EMA(self.data.close, period=self.params.fast_period)
        self.mid_ma = bt.indicators.EMA(self.data.close, period=self.params.mid_period)
        self.slow_ma = bt.indicators.EMA(self.data.close, period=self.params.slow_period)
        
        # ATR
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        
        # 动量指标
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.macd = bt.indicators.MACD(self.data.close)
        
        # 交易状态
        self.order = None
        self.buy_price = None
        self.stop_price = None
        self.highest_price = None
        self.trade_count = 0

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status == order.Completed:
            if order.isbuy():
                self.buy_price = order.executed.price
                self.stop_price = self.buy_price - (self.atr[0] * self.params.atr_stop_mult)
                self.highest_price = order.executed.price
                if self.params.print_trades:
                    print(f'买入: {self.data.datetime.date(0)}, 价格: ${order.executed.price:.2f}, '
                          f'数量: {order.executed.size:.0f}, 止损: ${self.stop_price:.2f}')
            elif order.issell():
                if self.params.print_trades:
                    print(f'卖出: {self.data.datetime.date(0)}, 价格: ${order.executed.price:.2f}')
                self.buy_price = None
                self.stop_price = None
                self.highest_price = None
                
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_count += 1
            if self.params.print_trades:
                cost = abs(trade.price * trade.size) if trade.size != 0 else 1
                pnl_pct = (trade.pnlcomm / cost) * 100
                print(f'交易 #{self.trade_count}, 净利润: ${trade.pnlcomm:,.2f} ({pnl_pct:+.1f}%)\n')

    def next(self):
        if self.order:
            return
            
        price = self.data.close[0]
        
        # === 趋势判断 ===
        uptrend = (price > self.slow_ma[0] and 
                   self.mid_ma[0] > self.slow_ma[0] and
                   self.fast_ma[0] > self.mid_ma[0])
        
        # === 持仓管理 ===
        if self.position:
            # 更新最高价和移动止损
            if price > self.highest_price:
                self.highest_price = price
                # 提高止损位 (跟踪止损)
                new_stop = self.highest_price * (1 - self.params.trailing_pct)
                if new_stop > self.stop_price:
                    self.stop_price = new_stop
            
            # 出场条件
            should_sell = False
            reason = ""
            
            # 1. 触及止损
            if price < self.stop_price:
                should_sell = True
                reason = f"止损 (止损价: ${self.stop_price:.2f})"
            
            # 2. 趋势反转 - 价格跌破中期均线
            elif price < self.mid_ma[0] and self.fast_ma[0] < self.mid_ma[0]:
                should_sell = True
                reason = "趋势反转"
            
            if should_sell:
                if self.params.print_trades:
                    pnl = (price - self.buy_price) / self.buy_price * 100
                    print(f'  -> {reason}, 浮盈: {pnl:+.1f}%')
                self.order = self.close()
                return
        
        # === 入场逻辑 ===
        else:
            # 条件1: 上升趋势确认
            if not uptrend:
                return
            
            # 条件2: 回调到短期均线附近 (好的入场点)
            pullback_zone = price <= self.fast_ma[0] * (1 + self.params.pullback_pct)
            
            # 条件3: RSI不超买
            rsi_ok = self.rsi[0] < 70
            
            # 条件4: MACD柱状图为正 (动量向上)
            macd_ok = self.macd.macd[0] > self.macd.signal[0]
            
            if pullback_zone and rsi_ok and macd_ok:
                cash = self.broker.getcash()
                size = int((cash * self.params.risk_percent) / price)
                if size > 0:
                    self.order = self.buy(size=size)

        # === 信号播报 ===
        is_latest = len(self.data) == self.data.buflen()
        if is_latest and self.params.print_log:
            self._print_signal(price, uptrend)

    def _print_signal(self, price, uptrend):
        dt = self.data.datetime.date(0)
        print(f"\n======== 趋势动量扫描: {dt} ========")
        print(f"标的: {self.data._name}")
        print(f"当前价格: ${price:.2f}")
        print(f"----------------------------------------")
        print(f"EMA10: ${self.fast_ma[0]:.2f} | EMA20: ${self.mid_ma[0]:.2f} | EMA50: ${self.slow_ma[0]:.2f}")
        print(f"RSI: {self.rsi[0]:.1f} | MACD: {self.macd.macd[0]:.2f}")
        print(f"ATR: ${self.atr[0]:.2f}")
        print(f"----------------------------------------")
        
        if uptrend:
            if self.rsi[0] < 70:
                print("信号: 【上升趋势】等待回调到EMA10附近买入")
            else:
                print("信号: 【超买】趋势向上但短期超买，等待回调")
        else:
            print("信号: 【观望】趋势未确认，保持耐心")
        print("========================================\n")


def download_data(symbol: str, start: str, end: str = None) -> pd.DataFrame:
    """下载股票数据的工具函数"""
    print(f"正在从 Yahoo Finance 下载 {symbol} 数据...")
    data_df = yf.download(symbol, start=start, end=end, progress=False)
    
    # 处理 MultiIndex
    if isinstance(data_df.columns, pd.MultiIndex):
        data_df.columns = data_df.columns.get_level_values(0)
    
    return data_df


def create_cerebro(data_df: pd.DataFrame, symbol: str, cash: float = 10000, 
                   commission: float = 0.001, **strategy_params) -> bt.Cerebro:
    """创建并配置 Cerebro 引擎"""
    cerebro = bt.Cerebro()
    
    # 添加策略
    cerebro.addstrategy(TrendFilterStrategy, **strategy_params)
    
    # 添加数据
    data = bt.feeds.PandasData(dataname=data_df, plot=False)
    data._name = symbol
    cerebro.adddata(data)
    
    # 设置资金和手续费
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    
    return cerebro


def run_signal_scan(symbol: str = 'NVDA', start: str = '2022-01-01'):
    """运行每日信号扫描"""
    data_df = download_data(symbol, start)
    cerebro = create_cerebro(data_df, symbol, print_log=True)
    print("开始计算策略...")
    cerebro.run()
    return cerebro


# --- 2. 运行脚本 ---
if __name__ == '__main__':
    cerebro = run_signal_scan()
    # 画图
    cerebro.plot(style='candlestick', volume=False)
