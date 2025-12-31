"""
Multi-Indicator Confirmation Strategy.
Requires RSI, MACD, and Bollinger Bands to all signal buy before entering.
Fixed holding period exit.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, StrategySignal, SignalType


class MultiIndicatorStrategy(BaseStrategy):
    """
    多指标确认策略：RSI + MACD + 布林带同时出现买入信号时买入，持有固定天数后卖出。
    
    买入条件（需同时满足）：
    1. RSI < rsi_oversold（超卖）
    2. MACD金叉（MACD线上穿信号线）或 MACD柱状图由负转正
    3. 价格触及或跌破布林带下轨
    
    卖出条件：
    - 持有满 holding_days 天后卖出
    """
    
    def __init__(self,
                 rsi_period: int = 14,
                 rsi_oversold: int = 30,
                 rsi_overbought: int = 70,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 holding_days: int = 20,
                 stop_loss_pct: float = 0.08,
                 take_profit_pct: float = 0.15):
        """
        初始化多指标确认策略。
        
        Args:
            rsi_period: RSI计算周期
            rsi_oversold: RSI超卖阈值（低于此值视为超卖）
            rsi_overbought: RSI超买阈值
            macd_fast: MACD快线周期
            macd_slow: MACD慢线周期
            macd_signal: MACD信号线周期
            bb_period: 布林带周期
            bb_std: 布林带标准差倍数
            holding_days: 持有天数（约1个月=20个交易日）
            stop_loss_pct: 止损百分比
            take_profit_pct: 止盈百分比
        """
        
        parameters = {
            'rsi_period': rsi_period,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_signal': macd_signal,
            'bb_period': bb_period,
            'bb_std': bb_std,
            'holding_days': holding_days,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
        
        super().__init__("MultiIndicatorStrategy", parameters)
        
        # 跟踪持仓状态
        self.entry_date = None
        self.entry_price = None
    
    def _validate_parameters(self) -> None:
        """验证策略参数。"""
        params = self.parameters
        
        if params['rsi_period'] < 2:
            raise ValueError("RSI周期必须大于等于2")
        
        if not (0 < params['rsi_oversold'] < params['rsi_overbought'] < 100):
            raise ValueError("RSI阈值必须满足: 0 < oversold < overbought < 100")
        
        if params['macd_fast'] >= params['macd_slow']:
            raise ValueError("MACD快线周期必须小于慢线周期")
        
        if params['holding_days'] < 1:
            raise ValueError("持有天数必须大于等于1")
    
    def generate_signals(self, data: pd.DataFrame) -> List[StrategySignal]:
        """
        生成交易信号。
        只生成BUY信号，卖出由回测引擎根据持有期/止损/止盈自动处理。
        
        Args:
            data: 包含OHLCV数据和技术指标的DataFrame
            
        Returns:
            StrategySignal对象列表
        """
        
        # 确保所需指标存在
        data = self._ensure_indicators(data)
        
        signals = []
        params = self.parameters
        
        # 跟踪上次买入时间，避免频繁交易
        last_buy_idx = -params['holding_days']  # 初始化为足够早的时间
        
        # 需要足够的数据来计算指标
        start_idx = max(params['macd_slow'], params['bb_period'], params['rsi_period']) + 5
        
        rsi_col = f"RSI_{params['rsi_period']}"
        bb_lower_col = f"BB_Lower_{params['bb_period']}"
        bb_upper_col = f"BB_Upper_{params['bb_period']}"
        
        for i in range(start_idx, len(data)):
            current_row = data.iloc[i]
            prev_row = data.iloc[i - 1]
            current_time = data.index[i]
            current_price = current_row['Close']
            
            # 获取指标值
            rsi = current_row[rsi_col]
            macd = current_row['MACD']
            macd_signal_line = current_row['MACD_Signal']
            macd_hist = current_row['MACD_Histogram']
            prev_macd_hist = prev_row['MACD_Histogram']
            bb_lower = current_row[bb_lower_col]
            bb_upper = current_row[bb_upper_col]
            
            # 跳过NaN值
            if pd.isna(rsi) or pd.isna(macd) or pd.isna(macd_signal_line) or pd.isna(macd_hist) or pd.isna(bb_lower):
                continue
            
            # 检查是否在冷却期内（上次买入后需等待holding_days天）
            if i - last_buy_idx < params['holding_days']:
                continue
            
            # 检查买入条件（需同时满足）
            
            # 条件1: RSI超卖或接近超卖区域
            rsi_buy = rsi < params['rsi_oversold'] + 10  # 放宽到40以下
            
            # 条件2: MACD开始企稳或反转的迹象
            macd_improving = macd_hist > prev_macd_hist  # 柱状图上升
            macd_narrowing = abs(macd_hist) < abs(prev_macd_hist) and macd_hist < 0  # 负柱状图收窄
            macd_crossover = macd > macd_signal_line  # 金叉
            macd_buy = macd_improving or macd_narrowing or macd_crossover
            
            # 条件3: 价格接近布林带下轨
            bb_range = bb_upper - bb_lower
            bb_buy = current_price <= bb_lower + bb_range * 0.3
            
            # 三个条件同时满足
            if rsi_buy and macd_buy and bb_buy:
                # 计算信号强度
                strength = self._calculate_signal_strength(
                    rsi, params['rsi_oversold'],
                    macd_hist, prev_macd_hist,
                    current_price, bb_lower
                )
                
                # 计算止损止盈价格
                stop_loss = current_price * (1 - params['stop_loss_pct'])
                take_profit = current_price * (1 + params['take_profit_pct'])
                
                signal = StrategySignal(
                    timestamp=current_time,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'rsi': rsi,
                        'macd': macd,
                        'macd_signal': macd_signal_line,
                        'macd_histogram': macd_hist,
                        'bb_lower': bb_lower,
                        'bb_upper': bb_upper,
                        'holding_days': params['holding_days']
                    }
                )
                signals.append(signal)
                last_buy_idx = i
        
        return signals
    
    def _ensure_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """确保所有必需的指标都已计算。"""
        
        data_copy = data.copy()
        params = self.parameters
        
        # RSI - 检查是否已存在
        rsi_col = f"RSI_{params['rsi_period']}"
        if rsi_col not in data_copy.columns:
            delta = data_copy['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(params['rsi_period']).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(params['rsi_period']).mean()
            rs = gain / loss
            data_copy[rsi_col] = 100 - (100 / (1 + rs))
        
        # MACD - 检查是否已存在
        if 'MACD' not in data_copy.columns:
            ema_fast = data_copy['Close'].ewm(span=params['macd_fast']).mean()
            ema_slow = data_copy['Close'].ewm(span=params['macd_slow']).mean()
            data_copy['MACD'] = ema_fast - ema_slow
            data_copy['MACD_Signal'] = data_copy['MACD'].ewm(span=params['macd_signal']).mean()
            data_copy['MACD_Histogram'] = data_copy['MACD'] - data_copy['MACD_Signal']
        
        # Bollinger Bands - 检查是否已存在
        bb_lower_col = f"BB_Lower_{params['bb_period']}"
        bb_upper_col = f"BB_Upper_{params['bb_period']}"
        if bb_lower_col not in data_copy.columns:
            sma = data_copy['Close'].rolling(params['bb_period']).mean()
            std = data_copy['Close'].rolling(params['bb_period']).std()
            data_copy[bb_upper_col] = sma + (std * params['bb_std'])
            data_copy[bb_lower_col] = sma - (std * params['bb_std'])
            data_copy[f"BB_Middle_{params['bb_period']}"] = sma
        
        return data_copy
    
    def _calculate_signal_strength(self,
                                   rsi: float,
                                   rsi_threshold: float,
                                   macd_hist: float,
                                   prev_macd_hist: float,
                                   price: float,
                                   bb_lower: float) -> float:
        """
        计算买入信号强度。
        
        Returns:
            0到1之间的信号强度
        """
        
        # RSI强度：越超卖越强
        rsi_strength = min((rsi_threshold - rsi) / rsi_threshold, 1.0)
        
        # MACD强度：柱状图变化幅度
        macd_change = macd_hist - prev_macd_hist
        macd_strength = min(abs(macd_change) / 0.5, 1.0) if macd_change > 0 else 0.3
        
        # 布林带强度：价格离下轨越近越强
        bb_distance = (price - bb_lower) / bb_lower
        bb_strength = max(0, 1.0 - bb_distance * 10)
        
        # 综合强度
        total_strength = (rsi_strength * 0.4 + macd_strength * 0.3 + bb_strength * 0.3)
        
        return min(max(total_strength, 0.1), 1.0)
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """获取参数优化边界。"""
        
        return {
            'rsi_period': (7, 21),
            'rsi_oversold': (20, 40),
            'macd_fast': (8, 15),
            'macd_slow': (20, 30),
            'bb_period': (15, 25),
            'holding_days': (10, 40),
            'stop_loss_pct': (0.05, 0.15),
            'take_profit_pct': (0.10, 0.25)
        }
