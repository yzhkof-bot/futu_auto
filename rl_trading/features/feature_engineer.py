"""
特征工程 - 进阶版技术指标
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    特征工程类
    计算进阶技术指标并进行归一化
    """
    
    def __init__(
        self,
        normalize: bool = True,
        scaler_type: str = 'robust',  # 'standard' or 'robust'
        clip_value: float = 5.0
    ):
        """
        初始化特征工程器
        
        Args:
            normalize: 是否归一化特征
            scaler_type: 归一化方法
            clip_value: 特征裁剪范围
        """
        self.normalize = normalize
        self.scaler_type = scaler_type
        self.clip_value = clip_value
        self.scalers: Dict[str, object] = {}
        self.feature_columns: List[str] = []
        self.fitted = False
    
    def compute_features(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = True
    ) -> pd.DataFrame:
        """
        计算所有技术指标特征
        
        Args:
            df: 原始 OHLCV 数据
            fit_scaler: 是否拟合归一化器
            
        Returns:
            包含所有特征的 DataFrame
        """
        result = df.copy()
        
        # 1. 价格特征
        result = self._add_price_features(result)
        
        # 2. 移动平均
        result = self._add_moving_averages(result)
        
        # 3. MACD
        result = self._add_macd(result)
        
        # 4. RSI
        result = self._add_rsi(result)
        
        # 5. 布林带
        result = self._add_bollinger_bands(result)
        
        # 6. ATR (Average True Range)
        result = self._add_atr(result)
        
        # 7. ADX (Average Directional Index)
        result = self._add_adx(result)
        
        # 8. OBV (On-Balance Volume)
        result = self._add_obv(result)
        
        # 9. 随机指标
        result = self._add_stochastic(result)
        
        # 10. Williams %R
        result = self._add_williams_r(result)
        
        # 11. CCI (Commodity Channel Index)
        result = self._add_cci(result)
        
        # 12. 动量指标
        result = self._add_momentum(result)
        
        # 13. 波动率特征
        result = self._add_volatility_features(result)
        
        # 14. 价格位置特征
        result = self._add_price_position_features(result)
        
        # 获取特征列
        self.feature_columns = [col for col in result.columns 
                               if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
        
        # 处理缺失值 - 更强的 NaN 处理
        result = result.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 处理无穷值
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(0)
        
        # 跳过前 N 行（技术指标需要预热期）
        warmup_period = 60  # 最长周期指标需要约 50 天
        if len(result) > warmup_period:
            result = result.iloc[warmup_period:].reset_index(drop=True)
        
        # 归一化
        if self.normalize:
            if fit_scaler:
                result = self._fit_transform(result)
                self.fitted = True
            else:
                result = self._transform(result)
        
        return result
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加价格相关特征"""
        # 收益率
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 日内价格范围
        df['daily_range'] = (df['High'] - df['Low']) / df['Close']
        df['daily_range_pct'] = (df['High'] - df['Low']) / df['Open']
        
        # 跳空
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # 上下影线
        df['upper_shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Close']
        df['lower_shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Close']
        
        # 实体
        df['body'] = (df['Close'] - df['Open']) / df['Open']
        df['body_abs'] = np.abs(df['body'])
        
        return df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加移动平均特征"""
        periods = [5, 10, 20, 50]
        
        for period in periods:
            # SMA
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}'] - 1
            
            # EMA
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'ema_{period}_ratio'] = df['Close'] / df[f'ema_{period}'] - 1
        
        # 均线交叉信号
        df['sma_5_10_cross'] = (df['sma_5'] > df['sma_10']).astype(float)
        df['sma_10_20_cross'] = (df['sma_10'] > df['sma_20']).astype(float)
        df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(float)
        
        # 均线斜率
        for period in periods:
            df[f'sma_{period}_slope'] = df[f'sma_{period}'].pct_change(5)
        
        return df
    
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加 MACD 特征"""
        # 标准 MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 归一化 MACD
        df['macd_norm'] = df['macd'] / df['Close']
        df['macd_signal_norm'] = df['macd_signal'] / df['Close']
        df['macd_histogram_norm'] = df['macd_histogram'] / df['Close']
        
        # MACD 交叉信号
        df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(float)
        df['macd_histogram_positive'] = (df['macd_histogram'] > 0).astype(float)
        
        return df
    
    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加 RSI 特征"""
        for period in [7, 14, 21]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # RSI 归一化到 [-1, 1]
            df[f'rsi_{period}_norm'] = (df[f'rsi_{period}'] - 50) / 50
            
            # RSI 超买超卖信号
            df[f'rsi_{period}_overbought'] = (df[f'rsi_{period}'] > 70).astype(float)
            df[f'rsi_{period}_oversold'] = (df[f'rsi_{period}'] < 30).astype(float)
        
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加布林带特征"""
        for period in [20]:
            sma = df['Close'].rolling(period).mean()
            std = df['Close'].rolling(period).std()
            
            df[f'bb_upper_{period}'] = sma + 2 * std
            df[f'bb_lower_{period}'] = sma - 2 * std
            df[f'bb_middle_{period}'] = sma
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
            
            # 价格在布林带中的位置 (0-1)
            df[f'bb_position_{period}'] = (df['Close'] - df[f'bb_lower_{period}']) / \
                                          (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)
            
            # 布林带挤压
            df[f'bb_squeeze_{period}'] = df[f'bb_width_{period}'] / df[f'bb_width_{period}'].rolling(50).mean()
        
        return df
    
    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加 ATR 特征"""
        for period in [14, 21]:
            tr = np.maximum(
                df['High'] - df['Low'],
                np.maximum(
                    np.abs(df['High'] - df['Close'].shift(1)),
                    np.abs(df['Low'] - df['Close'].shift(1))
                )
            )
            df[f'atr_{period}'] = tr.rolling(period).mean()
            df[f'atr_{period}_norm'] = df[f'atr_{period}'] / df['Close']
            
            # ATR 变化率
            df[f'atr_{period}_change'] = df[f'atr_{period}'].pct_change(5)
        
        return df
    
    def _add_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加 ADX 特征"""
        period = 14
        
        # +DM 和 -DM
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # True Range
        tr = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                np.abs(df['High'] - df['Close'].shift(1)),
                np.abs(df['Low'] - df['Close'].shift(1))
            )
        )
        
        # 平滑
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))
        
        # DX 和 ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(period).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # ADX 归一化
        df['adx_norm'] = df['adx'] / 100
        
        # 趋势强度信号
        df['adx_strong_trend'] = (df['adx'] > 25).astype(float)
        df['di_cross'] = (df['plus_di'] > df['minus_di']).astype(float)
        
        return df
    
    def _add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加 OBV 特征"""
        obv = np.where(df['Close'] > df['Close'].shift(1), df['Volume'],
                       np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0))
        df['obv'] = np.cumsum(obv)
        
        # OBV 变化率
        df['obv_change'] = df['obv'].pct_change(5)
        
        # OBV 与价格背离
        df['obv_sma_20'] = df['obv'].rolling(20).mean()
        df['obv_trend'] = (df['obv'] > df['obv_sma_20']).astype(float)
        
        return df
    
    def _add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加随机指标"""
        for period in [14]:
            low_min = df['Low'].rolling(period).min()
            high_max = df['High'].rolling(period).max()
            
            df[f'stoch_k_{period}'] = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-10)
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(3).mean()
            
            # 归一化到 [-1, 1]
            df[f'stoch_k_{period}_norm'] = (df[f'stoch_k_{period}'] - 50) / 50
            df[f'stoch_d_{period}_norm'] = (df[f'stoch_d_{period}'] - 50) / 50
            
            # 超买超卖
            df[f'stoch_{period}_overbought'] = (df[f'stoch_k_{period}'] > 80).astype(float)
            df[f'stoch_{period}_oversold'] = (df[f'stoch_k_{period}'] < 20).astype(float)
        
        return df
    
    def _add_williams_r(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加 Williams %R"""
        period = 14
        
        high_max = df['High'].rolling(period).max()
        low_min = df['Low'].rolling(period).min()
        
        df['williams_r'] = -100 * (high_max - df['Close']) / (high_max - low_min + 1e-10)
        df['williams_r_norm'] = (df['williams_r'] + 50) / 50  # 归一化到 [-1, 1]
        
        return df
    
    def _add_cci(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加 CCI"""
        period = 20
        
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad + 1e-10)
        df['cci_norm'] = df['cci'] / 200  # 粗略归一化
        df['cci_norm'] = df['cci_norm'].clip(-1, 1)
        
        return df
    
    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加动量指标"""
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            df[f'roc_{period}'] = df['Close'].pct_change(period)
        
        # 加速度（动量的变化）
        df['momentum_acceleration'] = df['momentum_10'].diff(5)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加波动率特征"""
        # 历史波动率
        for period in [10, 20, 50]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std() * np.sqrt(252)
        
        # 波动率变化
        df['volatility_change'] = df['volatility_20'] / df['volatility_50'] - 1
        
        # Parkinson 波动率
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            ((np.log(df['High'] / df['Low'])) ** 2).rolling(20).mean()
        ) * np.sqrt(252)
        
        return df
    
    def _add_price_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加价格位置特征"""
        # 相对于历史高低点的位置
        for period in [20, 50, 100]:
            rolling_high = df['High'].rolling(period).max()
            rolling_low = df['Low'].rolling(period).min()
            
            df[f'price_position_{period}'] = (df['Close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)
            
            # 距离历史高点
            df[f'dist_from_high_{period}'] = (df['Close'] - rolling_high) / rolling_high
            
            # 距离历史低点
            df[f'dist_from_low_{period}'] = (df['Close'] - rolling_low) / rolling_low
        
        return df
    
    def _fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合并转换特征"""
        result = df.copy()
        
        for col in self.feature_columns:
            if col in result.columns:
                if self.scaler_type == 'standard':
                    scaler = StandardScaler()
                else:
                    scaler = RobustScaler()
                
                values = result[col].values.reshape(-1, 1)
                # 处理无穷值
                values = np.nan_to_num(values, nan=0, posinf=self.clip_value, neginf=-self.clip_value)
                
                result[col] = scaler.fit_transform(values).flatten()
                result[col] = result[col].clip(-self.clip_value, self.clip_value)
                
                self.scalers[col] = scaler
        
        return result
    
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换特征（使用已拟合的 scaler）"""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call compute_features with fit_scaler=True first.")
        
        result = df.copy()
        
        for col in self.feature_columns:
            if col in result.columns and col in self.scalers:
                values = result[col].values.reshape(-1, 1)
                values = np.nan_to_num(values, nan=0, posinf=self.clip_value, neginf=-self.clip_value)
                
                result[col] = self.scalers[col].transform(values).flatten()
                result[col] = result[col].clip(-self.clip_value, self.clip_value)
        
        return result
    
    def get_feature_columns(self) -> List[str]:
        """获取特征列名"""
        return self.feature_columns.copy()
    
    def get_feature_importance_columns(self) -> List[str]:
        """获取最重要的特征列（用于简化状态空间）"""
        important_features = [
            # 价格特征
            'returns', 'log_returns', 'daily_range', 'gap',
            # 均线
            'sma_5_ratio', 'sma_20_ratio', 'ema_10_ratio',
            'sma_5_10_cross', 'sma_20_50_cross',
            # MACD
            'macd_norm', 'macd_histogram_norm', 'macd_cross',
            # RSI
            'rsi_14_norm', 'rsi_14_overbought', 'rsi_14_oversold',
            # 布林带
            'bb_position_20', 'bb_width_20', 'bb_squeeze_20',
            # ATR
            'atr_14_norm', 'atr_14_change',
            # ADX
            'adx_norm', 'di_cross', 'adx_strong_trend',
            # OBV
            'obv_change', 'obv_trend',
            # 随机指标
            'stoch_k_14_norm', 'stoch_14_overbought', 'stoch_14_oversold',
            # Williams %R
            'williams_r_norm',
            # CCI
            'cci_norm',
            # 动量
            'momentum_10', 'momentum_20', 'momentum_acceleration',
            # 波动率
            'volatility_20', 'volatility_change',
            # 价格位置
            'price_position_20', 'price_position_50',
            'dist_from_high_20', 'dist_from_low_20'
        ]
        
        return [f for f in important_features if f in self.feature_columns]
    
    def get_core_features(self) -> List[str]:
        """
        获取核心特征列（精简版，约 12 个特征）
        
        设计原则：
        - 去掉高度相关的冗余特征
        - 保留每个类别最有代表性的指标
        - 减少噪声，帮助模型更快收敛
        """
        core_features = [
            # 价格动量 (3个) - 最基础的信号
            'returns',           # 日收益率
            'momentum_10',       # 10日动量
            'momentum_20',       # 20日动量
            
            # 趋势 (3个) - 判断方向
            'sma_20_ratio',      # 价格相对20日均线位置
            'macd_norm',         # MACD 归一化
            'adx_norm',          # 趋势强度
            
            # 波动率 (2个) - 风险度量
            'volatility_20',     # 20日波动率
            'atr_14_norm',       # ATR 归一化
            
            # 超买超卖 (2个) - 反转信号
            'rsi_14_norm',       # RSI 归一化
            'bb_position_20',    # 布林带位置
            
            # 价格位置 (2个) - 相对位置
            'price_position_20', # 20日高低点位置
            'price_position_50', # 50日高低点位置
        ]
        
        return [f for f in core_features if f in self.feature_columns]
