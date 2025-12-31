"""
多信号组合策略

买入条件：至少3个信号同时出现
1. 均线空头排列 + RSI超卖(<30)
2. RSI超卖 + KDJ超卖(K,D<20)
3. 接近60日低点 + KDJ超卖
4. MACD零下金叉

卖出条件：持有N个交易日（默认20天）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Signal:
    """信号数据类"""
    date: pd.Timestamp
    price: float
    signal_count: int
    signals: List[str]
    rsi: float
    k: float
    d: float
    macd: float
    signal_line: float


@dataclass
class Trade:
    """交易数据类"""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl_pct: float
    signal_count: int
    signals: List[str]


class MultiSignalStrategy:
    """多信号组合策略"""
    
    def __init__(
        self,
        min_signals: int = 3,
        hold_days: int = 20,
        rsi_oversold: float = 30,
        kdj_oversold: float = 20,
        low_threshold: float = 1.05
    ):
        """
        初始化策略参数
        
        Args:
            min_signals: 最少需要的信号数量
            hold_days: 持有天数
            rsi_oversold: RSI超卖阈值
            kdj_oversold: KDJ超卖阈值
            low_threshold: 接近低点的阈值（1.05表示在60日低点的5%以内）
        """
        self.min_signals = min_signals
        self.hold_days = hold_days
        self.rsi_oversold = rsi_oversold
        self.kdj_oversold = kdj_oversold
        self.low_threshold = low_threshold
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术指标
        
        Args:
            df: 包含OHLCV数据的DataFrame，需要有 Open, High, Low, Close, Volume 列
        
        Returns:
            添加了技术指标的DataFrame
        """
        df = df.copy()
        
        # 确保索引是日期类型
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # 均线
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # KDJ
        low_min = df['Low'].rolling(window=9).min()
        high_max = df['High'].rolling(window=9).max()
        rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
        df['K'] = rsv.ewm(com=2, adjust=False).mean()
        df['D'] = df['K'].ewm(com=2, adjust=False).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
        
        # MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
        
        # 60日低点
        df['Low_60'] = df['Low'].rolling(window=60).min()
        
        # 均线排列
        df['MA_Bearish'] = (df['MA5'] < df['MA10']) & (df['MA10'] < df['MA20'])
        df['MA_Bullish'] = (df['MA5'] > df['MA10']) & (df['MA10'] > df['MA20'])
        
        return df
    
    def detect_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        检测所有买入信号
        
        Args:
            df: 已计算指标的DataFrame
        
        Returns:
            添加了信号列的DataFrame
        """
        df = df.copy()
        
        # 信号1: 均线空头 + RSI超卖
        df['Signal_1'] = df['MA_Bearish'] & (df['RSI'] < self.rsi_oversold)
        
        # 信号2: RSI超卖 + KDJ超卖
        df['Signal_2'] = (df['RSI'] < self.rsi_oversold) & (df['K'] < self.kdj_oversold) & (df['D'] < self.kdj_oversold)
        
        # 信号3: 接近60日低点 + KDJ超卖
        df['Signal_3'] = (df['Close'] < df['Low_60'] * self.low_threshold) & (df['K'] < self.kdj_oversold)
        
        # 信号4: MACD零下金叉
        df['MACD_Cross'] = (df['MACD'] < 0) & (df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1))
        df['Signal_4'] = df['MACD_Cross']
        
        # 计算信号总数
        df['Signal_Count'] = df['Signal_1'].astype(int) + df['Signal_2'].astype(int) + df['Signal_3'].astype(int) + df['Signal_4'].astype(int)
        
        # 买入信号
        df['Buy_Signal'] = df['Signal_Count'] >= self.min_signals
        
        return df
    
    def get_buy_signals(self, df: pd.DataFrame) -> List[Signal]:
        """
        获取所有买入信号
        
        Args:
            df: 已检测信号的DataFrame
        
        Returns:
            Signal对象列表
        """
        signals = []
        buy_dates = df[df['Buy_Signal']].index
        
        for date in buy_dates:
            row = df.loc[date]
            signal_names = []
            if row['Signal_1']:
                signal_names.append('均线空头+RSI超卖')
            if row['Signal_2']:
                signal_names.append('RSI+KDJ超卖')
            if row['Signal_3']:
                signal_names.append('60日低点+KDJ超卖')
            if row['Signal_4']:
                signal_names.append('MACD零下金叉')
            
            signals.append(Signal(
                date=date,
                price=row['Close'],
                signal_count=int(row['Signal_Count']),
                signals=signal_names,
                rsi=row['RSI'],
                k=row['K'],
                d=row['D'],
                macd=row['MACD'],
                signal_line=row['Signal_Line']
            ))
        
        return signals
    
    def is_buy_signal_today(self, df: pd.DataFrame) -> Tuple[bool, Optional[Signal]]:
        """
        判断今天（最后一个交易日）是否为买点
        
        Args:
            df: 原始OHLCV数据
        
        Returns:
            (是否为买点, Signal对象或None)
        """
        df = self.calculate_indicators(df)
        df = self.detect_signals(df)
        
        # 获取最后一行
        last_row = df.iloc[-1]
        last_date = df.index[-1]
        
        if last_row['Buy_Signal']:
            signal_names = []
            if last_row['Signal_1']:
                signal_names.append('均线空头+RSI超卖')
            if last_row['Signal_2']:
                signal_names.append('RSI+KDJ超卖')
            if last_row['Signal_3']:
                signal_names.append('60日低点+KDJ超卖')
            if last_row['Signal_4']:
                signal_names.append('MACD零下金叉')
            
            signal = Signal(
                date=last_date,
                price=last_row['Close'],
                signal_count=int(last_row['Signal_Count']),
                signals=signal_names,
                rsi=last_row['RSI'],
                k=last_row['K'],
                d=last_row['D'],
                macd=last_row['MACD'],
                signal_line=last_row['Signal_Line']
            )
            return True, signal
        
        return False, None
    
    def check_ticker(self, ticker: str, days: int = 120) -> Tuple[bool, Optional[Signal]]:
        """
        检查某只股票今天是否为买点（便捷方法）
        
        Args:
            ticker: 股票代码，如 'TSLA', 'AAPL'
            days: 获取多少天的历史数据（需要足够计算指标）
        
        Returns:
            (是否为买点, Signal对象或None)
        """
        import yfinance as yf
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = yf.download(
            ticker, 
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )
        
        if data.empty:
            return False, None
        
        # 处理MultiIndex列
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        return self.is_buy_signal_today(data)
    
    def scan_nasdaq100(self) -> List[Tuple[str, Signal]]:
        """
        扫描纳斯达克100所有股票，返回今天是买点的股票
        
        Returns:
            [(股票代码, Signal对象), ...] 列表
        """
        import yfinance as yf
        
        # 纳斯达克100成分股
        nasdaq100 = [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'COST',
            'NFLX', 'TMUS', 'ASML', 'CSCO', 'ADBE', 'AMD', 'PEP', 'LIN', 'INTC', 'INTU',
            'TXN', 'CMCSA', 'QCOM', 'AMGN', 'AMAT', 'HON', 'ISRG', 'BKNG', 'SBUX', 'VRTX',
            'LRCX', 'ADP', 'GILD', 'ADI', 'MU', 'MDLZ', 'REGN', 'PANW', 'KLAC', 'SNPS',
            'CDNS', 'MELI', 'MAR', 'PYPL', 'CSX', 'CRWD', 'ORLY', 'CTAS', 'NXPI', 'MNST',
            'MRVL', 'PCAR', 'WDAY', 'ADSK', 'ABNB', 'CPRT', 'ROP', 'AEP', 'FTNT', 'PAYX',
            'AZN', 'CHTR', 'ROST', 'KDP', 'ODFL', 'DXCM', 'KHC', 'FAST', 'TTD', 'MCHP',
            'GEHC', 'VRSK', 'EA', 'CTSH', 'EXC', 'LULU', 'CSGP', 'FANG', 'IDXX', 'BKR',
            'XEL', 'CCEP', 'ON', 'ANSS', 'TEAM', 'CDW', 'BIIB', 'ZS', 'GFS', 'DDOG',
            'ILMN', 'WBD', 'MDB', 'MRNA', 'DLTR', 'CEG', 'SMCI', 'ARM', 'DASH', 'COIN'
        ]
        
        buy_signals = []
        print(f"扫描纳斯达克100股票...")
        
        for i, ticker in enumerate(nasdaq100):
            try:
                is_buy, signal = self.check_ticker(ticker)
                if is_buy:
                    buy_signals.append((ticker, signal))
                    print(f"  ✓ {ticker}: 买点! 信号数={signal.signal_count}")
            except Exception as e:
                pass  # 跳过下载失败的股票
            
            # 进度显示
            if (i + 1) % 20 == 0:
                print(f"  已扫描 {i + 1}/100...")
        
        print(f"\n扫描完成，共发现 {len(buy_signals)} 个买点")
        return buy_signals
    
    def backtest(self, df: pd.DataFrame) -> Tuple[List[Trade], Dict]:
        """
        回测策略
        
        Args:
            df: 原始OHLCV数据
        
        Returns:
            (交易列表, 统计指标字典)
        """
        # 计算指标和信号
        df = self.calculate_indicators(df)
        df = self.detect_signals(df)
        
        trades = []
        buy_signals = self.get_buy_signals(df)
        
        for signal in buy_signals:
            entry_idx = df.index.get_loc(signal.date)
            exit_idx = min(entry_idx + self.hold_days, len(df) - 1)
            
            if exit_idx > entry_idx:
                exit_date = df.index[exit_idx]
                exit_price = df.iloc[exit_idx]['Close']
                pnl_pct = (exit_price - signal.price) / signal.price
                
                trades.append(Trade(
                    entry_date=signal.date,
                    exit_date=exit_date,
                    entry_price=signal.price,
                    exit_price=exit_price,
                    pnl_pct=pnl_pct,
                    signal_count=signal.signal_count,
                    signals=signal.signals
                ))
        
        # 计算统计指标
        if trades:
            pnl_list = [t.pnl_pct for t in trades]
            wins = [p for p in pnl_list if p > 0]
            losses = [p for p in pnl_list if p <= 0]
            
            stats = {
                'total_trades': len(trades),
                'winning_trades': len(wins),
                'losing_trades': len(losses),
                'win_rate': len(wins) / len(trades) * 100,
                'avg_return': np.mean(pnl_list) * 100,
                'median_return': np.median(pnl_list) * 100,
                'max_return': np.max(pnl_list) * 100,
                'min_return': np.min(pnl_list) * 100,
                'total_return': np.sum(pnl_list) * 100,
                'avg_win': np.mean(wins) * 100 if wins else 0,
                'avg_loss': np.mean(losses) * 100 if losses else 0,
            }
        else:
            stats = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'median_return': 0,
                'max_return': 0,
                'min_return': 0,
                'total_return': 0,
                'avg_win': 0,
                'avg_loss': 0,
            }
        
        return trades, stats
    
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        运行策略，返回带有所有指标和信号的DataFrame
        
        Args:
            df: 原始OHLCV数据
        
        Returns:
            带有指标和信号的DataFrame
        """
        df = self.calculate_indicators(df)
        df = self.detect_signals(df)
        return df


def get_strategy(
    min_signals: int = 3,
    hold_days: int = 20,
    rsi_oversold: float = 30,
    kdj_oversold: float = 20
) -> MultiSignalStrategy:
    """
    获取策略实例的便捷函数
    
    Args:
        min_signals: 最少需要的信号数量
        hold_days: 持有天数
        rsi_oversold: RSI超卖阈值
        kdj_oversold: KDJ超卖阈值
    
    Returns:
        MultiSignalStrategy实例
    """
    return MultiSignalStrategy(
        min_signals=min_signals,
        hold_days=hold_days,
        rsi_oversold=rsi_oversold,
        kdj_oversold=kdj_oversold
    )


# 示例用法
if __name__ == '__main__':
    import yfinance as yf
    
    # 下载数据
    ticker = 'TSLA'
    data = yf.download(ticker, start='2021-01-01', end='2024-12-31', progress=False)
    
    # 处理MultiIndex列（yfinance新版本）
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # 创建策略
    strategy = get_strategy(min_signals=3, hold_days=20)
    
    # 回测
    trades, stats = strategy.backtest(data)
    
    # 打印结果
    print(f"\n=== {ticker} 多信号策略回测结果 ===\n")
    print(f"总交易次数: {stats['total_trades']}")
    print(f"胜率: {stats['win_rate']:.2f}%")
    print(f"平均收益: {stats['avg_return']:.2f}%")
    print(f"总收益: {stats['total_return']:.2f}%")
    
    print("\n交易详情:")
    for i, trade in enumerate(trades, 1):
        result = '✓' if trade.pnl_pct > 0 else '✗'
        print(f"{i}. {trade.entry_date.strftime('%Y-%m-%d')} -> {trade.exit_date.strftime('%Y-%m-%d')}: {trade.pnl_pct*100:+.2f}% {result}")
        print(f"   信号: {', '.join(trade.signals)}")
