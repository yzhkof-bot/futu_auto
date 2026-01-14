"""
Panel 数据管理器

工业级数据结构：日期 × 股票 的矩阵
支持批量获取、缓存、对齐
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import os
import pickle
import warnings

warnings.filterwarnings('ignore')

# 导入统一股票池
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.stock_pool import get_stock_pool


class PanelDataManager:
    """
    Panel 数据管理器
    
    数据结构:
        - close_panel: DataFrame (日期 × 股票) 收盘价
        - open_panel: DataFrame (日期 × 股票) 开盘价
        - high_panel: DataFrame (日期 × 股票) 最高价
        - low_panel: DataFrame (日期 × 股票) 最低价
        - volume_panel: DataFrame (日期 × 股票) 成交量
        - return_panel: DataFrame (日期 × 股票) 日收益率
    """
    
    def __init__(self, cache_dir: str = None):
        """
        初始化
        
        Args:
            cache_dir: 缓存目录，默认 gp_alpha/.cache
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), '.cache')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 数据面板
        self.open_panel: Optional[pd.DataFrame] = None
        self.high_panel: Optional[pd.DataFrame] = None
        self.low_panel: Optional[pd.DataFrame] = None
        self.close_panel: Optional[pd.DataFrame] = None
        self.volume_panel: Optional[pd.DataFrame] = None
        self.return_panel: Optional[pd.DataFrame] = None
        
        # 元数据
        self.symbols: List[str] = []
        self.dates: pd.DatetimeIndex = None
        self.start_date: str = None
        self.end_date: str = None
    
    def fetch(self, 
              symbols: List[str] = None,
              start_date: str = None,
              end_date: str = None,
              pool_type: str = 'all',
              use_cache: bool = True,
              min_data_days: int = 200,
              verbose: bool = True) -> 'PanelDataManager':
        """
        批量获取股票数据
        
        Args:
            symbols: 股票列表，None 则使用股票池
            start_date: 开始日期
            end_date: 结束日期
            pool_type: 股票池类型 ('nasdaq100', 'bluechip', 'all')
            use_cache: 是否使用缓存
            min_data_days: 最少数据天数，过滤数据不足的股票
            verbose: 是否打印进度
        
        Returns:
            self (链式调用)
        """
        # 默认日期
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        
        self.start_date = start_date
        self.end_date = end_date
        
        # 获取股票列表
        if symbols is None:
            symbols = get_stock_pool(pool_type)
        
        if verbose:
            print(f"获取数据: {len(symbols)} 只股票")
            print(f"日期范围: {start_date} ~ {end_date}")
        
        # 检查缓存
        cache_key = f"panel_{pool_type}_{start_date}_{end_date}"
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if use_cache and os.path.exists(cache_path):
            if verbose:
                print(f"从缓存加载: {cache_path}")
            return self._load_cache(cache_path)
        
        # 批量下载
        if verbose:
            print("下载中...")
        
        try:
            data = yf.download(
                symbols, 
                start=start_date, 
                end=end_date, 
                progress=verbose,
                threads=True
            )
        except Exception as e:
            print(f"下载失败: {e}")
            return self
        
        if data.empty:
            print("无数据")
            return self
        
        # 解析 MultiIndex 结构
        if isinstance(data.columns, pd.MultiIndex):
            # 多股票情况: columns = (Price, Symbol)
            self.open_panel = data['Open']
            self.high_panel = data['High']
            self.low_panel = data['Low']
            self.close_panel = data['Close']
            self.volume_panel = data['Volume']
        else:
            # 单股票情况
            symbol = symbols[0]
            self.open_panel = data[['Open']].rename(columns={'Open': symbol})
            self.high_panel = data[['High']].rename(columns={'High': symbol})
            self.low_panel = data[['Low']].rename(columns={'Low': symbol})
            self.close_panel = data[['Close']].rename(columns={'Close': symbol})
            self.volume_panel = data[['Volume']].rename(columns={'Volume': symbol})
        
        # 过滤数据不足的股票
        valid_symbols = []
        for symbol in self.close_panel.columns:
            valid_days = self.close_panel[symbol].notna().sum()
            if valid_days >= min_data_days:
                valid_symbols.append(symbol)
        
        if len(valid_symbols) < len(self.close_panel.columns):
            dropped = len(self.close_panel.columns) - len(valid_symbols)
            if verbose:
                print(f"过滤数据不足的股票: {dropped} 只")
            
            self.open_panel = self.open_panel[valid_symbols]
            self.high_panel = self.high_panel[valid_symbols]
            self.low_panel = self.low_panel[valid_symbols]
            self.close_panel = self.close_panel[valid_symbols]
            self.volume_panel = self.volume_panel[valid_symbols]
        
        # 计算日收益率
        self.return_panel = self.close_panel.pct_change()
        
        # 更新元数据
        self.symbols = list(self.close_panel.columns)
        self.dates = self.close_panel.index
        
        if verbose:
            print(f"有效股票: {len(self.symbols)} 只")
            print(f"数据天数: {len(self.dates)} 天")
        
        # 保存缓存
        if use_cache:
            self._save_cache(cache_path)
            if verbose:
                print(f"已缓存: {cache_path}")
        
        return self
    
    def _save_cache(self, path: str):
        """保存缓存"""
        cache_data = {
            'open_panel': self.open_panel,
            'high_panel': self.high_panel,
            'low_panel': self.low_panel,
            'close_panel': self.close_panel,
            'volume_panel': self.volume_panel,
            'return_panel': self.return_panel,
            'symbols': self.symbols,
            'dates': self.dates,
            'start_date': self.start_date,
            'end_date': self.end_date,
        }
        with open(path, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def _load_cache(self, path: str) -> 'PanelDataManager':
        """加载缓存"""
        with open(path, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.open_panel = cache_data['open_panel']
        self.high_panel = cache_data['high_panel']
        self.low_panel = cache_data['low_panel']
        self.close_panel = cache_data['close_panel']
        self.volume_panel = cache_data['volume_panel']
        self.return_panel = cache_data['return_panel']
        self.symbols = cache_data['symbols']
        self.dates = cache_data['dates']
        self.start_date = cache_data['start_date']
        self.end_date = cache_data['end_date']
        
        return self
    
    def get_forward_return(self, days: int = 1) -> pd.DataFrame:
        """
        计算未来 N 日收益率
        
        Args:
            days: 未来天数
        
        Returns:
            DataFrame (日期 × 股票) 未来收益率
        """
        if self.close_panel is None:
            raise ValueError("请先调用 fetch() 获取数据")
        
        # 未来收益 = (未来价格 - 当前价格) / 当前价格
        future_close = self.close_panel.shift(-days)
        forward_return = (future_close - self.close_panel) / self.close_panel
        
        return forward_return
    
    def get_feature_panels(self) -> Dict[str, pd.DataFrame]:
        """
        获取工业级特征面板
        
        Returns:
            特征名 -> DataFrame 的字典
        """
        if self.close_panel is None:
            raise ValueError("请先调用 fetch() 获取数据")
        
        features = {}
        
        # ==================== 基础价量 (6个) ====================
        features['open'] = self.open_panel
        features['high'] = self.high_panel
        features['low'] = self.low_panel
        features['close'] = self.close_panel
        features['volume'] = self.volume_panel
        features['amount'] = self.close_panel * self.volume_panel  # 成交额
        
        # ==================== 收益率 (6个) ====================
        features['return_1'] = self.return_panel
        features['return_2'] = self.close_panel.pct_change(2)
        features['return_3'] = self.close_panel.pct_change(3)
        features['return_5'] = self.close_panel.pct_change(5)
        features['return_10'] = self.close_panel.pct_change(10)
        features['return_20'] = self.close_panel.pct_change(20)
        
        # ==================== VWAP 相关 (4个) ====================
        # 典型价格 (HLC 均值)
        typical_price = (self.high_panel + self.low_panel + self.close_panel) / 3
        features['vwap'] = typical_price
        # 价格相对 VWAP 偏离
        features['close_vwap_ratio'] = self.close_panel / typical_price
        # 加权均价（近似）
        features['wap'] = (self.high_panel + self.low_panel + 2 * self.close_panel) / 4
        features['close_wap_ratio'] = self.close_panel / features['wap']
        
        # ==================== 价格位置 (6个) ====================
        # 日内位置
        daily_range = self.high_panel - self.low_panel
        features['intraday_pos'] = (self.close_panel - self.low_panel) / daily_range.replace(0, np.nan)
        # 开盘位置
        features['open_pos'] = (self.open_panel - self.low_panel) / daily_range.replace(0, np.nan)
        # N日高低点位置
        for n in [5, 10, 20]:
            high_n = self.high_panel.rolling(n).max()
            low_n = self.low_panel.rolling(n).min()
            range_n = high_n - low_n
            features[f'pos_{n}d'] = (self.close_panel - low_n) / range_n.replace(0, np.nan)
        
        # ==================== 价格比率 (8个) ====================
        features['hl_ratio'] = self.high_panel / self.low_panel  # 振幅比
        features['co_ratio'] = self.close_panel / self.open_panel  # 收盘/开盘
        features['hc_ratio'] = self.high_panel / self.close_panel  # 上影线
        features['lc_ratio'] = self.low_panel / self.close_panel  # 下影线
        features['ho_ratio'] = self.high_panel / self.open_panel  # 高/开
        features['lo_ratio'] = self.low_panel / self.open_panel  # 低/开
        # 跳空
        features['gap'] = self.open_panel / self.close_panel.shift(1) - 1
        features['gap_hl'] = self.open_panel / self.high_panel.shift(1) - 1
        
        # ==================== 振幅相关 (5个) ====================
        features['amplitude'] = daily_range / self.close_panel.shift(1)
        features['amplitude_5'] = daily_range.rolling(5).mean() / self.close_panel.shift(1)
        features['amplitude_10'] = daily_range.rolling(10).mean() / self.close_panel.shift(1)
        # 真实波幅 ATR
        tr1 = self.high_panel - self.low_panel
        tr2 = (self.high_panel - self.close_panel.shift(1)).abs()
        tr3 = (self.low_panel - self.close_panel.shift(1)).abs()
        # 使用 np.maximum 逐元素取最大
        true_range = np.maximum(np.maximum(tr1, tr2), tr3)
        features['atr_5'] = true_range.rolling(5).mean()
        features['atr_14'] = true_range.rolling(14).mean()
        
        # ==================== 成交量相关 (10个) ====================
        # 量比
        features['vol_ratio_5'] = self.volume_panel / self.volume_panel.rolling(5).mean()
        features['vol_ratio_10'] = self.volume_panel / self.volume_panel.rolling(10).mean()
        features['vol_ratio_20'] = self.volume_panel / self.volume_panel.rolling(20).mean()
        # 量变化
        features['vol_change_1'] = self.volume_panel.pct_change(1)
        features['vol_change_5'] = self.volume_panel.pct_change(5)
        # 量价相关
        features['vol_price_corr_10'] = self.close_panel.rolling(10).corr(self.volume_panel)
        features['vol_price_corr_20'] = self.close_panel.rolling(20).corr(self.volume_panel)
        # 量标准差
        features['vol_std_5'] = self.volume_panel.rolling(5).std() / self.volume_panel.rolling(5).mean()
        features['vol_std_10'] = self.volume_panel.rolling(10).std() / self.volume_panel.rolling(10).mean()
        # 成交额变化
        features['amount_ratio_5'] = features['amount'] / features['amount'].rolling(5).mean()
        
        # ==================== 动量指标 (8个) ====================
        # ROC
        features['roc_5'] = self.close_panel / self.close_panel.shift(5) - 1
        features['roc_10'] = self.close_panel / self.close_panel.shift(10) - 1
        features['roc_20'] = self.close_panel / self.close_panel.shift(20) - 1
        # 动量
        features['mom_5'] = self.close_panel - self.close_panel.shift(5)
        features['mom_10'] = self.close_panel - self.close_panel.shift(10)
        # 加速度（动量的变化）
        features['acc_5'] = features['mom_5'] - features['mom_5'].shift(5)
        features['acc_10'] = features['mom_10'] - features['mom_10'].shift(10)
        # 动量强度
        features['mom_strength'] = features['return_5'] / features['return_20'].replace(0, np.nan)
        
        # ==================== 均线相关 (12个) ====================
        ma5 = self.close_panel.rolling(5).mean()
        ma10 = self.close_panel.rolling(10).mean()
        ma20 = self.close_panel.rolling(20).mean()
        ma60 = self.close_panel.rolling(60).mean()
        
        # 价格相对均线
        features['close_ma5_ratio'] = self.close_panel / ma5
        features['close_ma10_ratio'] = self.close_panel / ma10
        features['close_ma20_ratio'] = self.close_panel / ma20
        features['close_ma60_ratio'] = self.close_panel / ma60
        
        # 均线斜率
        features['ma5_slope'] = ma5.pct_change(5)
        features['ma10_slope'] = ma10.pct_change(5)
        features['ma20_slope'] = ma20.pct_change(5)
        
        # 均线排列
        features['ma5_ma10_ratio'] = ma5 / ma10
        features['ma5_ma20_ratio'] = ma5 / ma20
        features['ma10_ma20_ratio'] = ma10 / ma20
        features['ma20_ma60_ratio'] = ma20 / ma60
        
        # 均线距离
        features['ma_spread'] = (ma5 - ma20) / ma20
        
        # ==================== 波动率 (8个) ====================
        features['volatility_5'] = self.return_panel.rolling(5).std()
        features['volatility_10'] = self.return_panel.rolling(10).std()
        features['volatility_20'] = self.return_panel.rolling(20).std()
        features['volatility_60'] = self.return_panel.rolling(60).std()
        
        # 波动率变化
        features['vol_change_ratio'] = features['volatility_5'] / features['volatility_20']
        
        # 高低波动
        features['high_vol_5'] = self.high_panel.rolling(5).std() / self.close_panel
        features['low_vol_5'] = self.low_panel.rolling(5).std() / self.close_panel
        
        # 实现波动率
        features['realized_vol_10'] = (self.return_panel ** 2).rolling(10).sum().apply(np.sqrt)
        
        # ==================== 技术指标 (12个) ====================
        # RSI
        delta = self.close_panel.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain_14 = gain.rolling(14).mean()
        avg_loss_14 = loss.rolling(14).mean()
        rs = avg_gain_14 / avg_loss_14.replace(0, np.nan)
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        avg_gain_6 = gain.rolling(6).mean()
        avg_loss_6 = loss.rolling(6).mean()
        rs6 = avg_gain_6 / avg_loss_6.replace(0, np.nan)
        features['rsi_6'] = 100 - (100 / (1 + rs6))
        
        # 布林带
        bb_mid = ma20
        bb_std = self.close_panel.rolling(20).std()
        features['bb_upper'] = (bb_mid + 2 * bb_std - self.close_panel) / self.close_panel
        features['bb_lower'] = (self.close_panel - bb_mid + 2 * bb_std) / self.close_panel
        features['bb_width'] = 4 * bb_std / bb_mid
        features['bb_pos'] = (self.close_panel - bb_mid) / (2 * bb_std)
        
        # MACD
        ema12 = self.close_panel.ewm(span=12, adjust=False).mean()
        ema26 = self.close_panel.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd'] = macd / self.close_panel
        features['macd_signal'] = signal / self.close_panel
        features['macd_hist'] = (macd - signal) / self.close_panel
        
        # CCI
        tp = typical_price
        tp_ma = tp.rolling(20).mean()
        tp_std = tp.rolling(20).std()
        features['cci_20'] = (tp - tp_ma) / (0.015 * tp_std)
        
        # ==================== 统计特征 (8个) ====================
        # 偏度
        features['skew_10'] = self.return_panel.rolling(10).skew()
        features['skew_20'] = self.return_panel.rolling(20).skew()
        
        # 峰度
        features['kurt_10'] = self.return_panel.rolling(10).kurt()
        features['kurt_20'] = self.return_panel.rolling(20).kurt()
        
        # 分位数
        features['quantile_10'] = self.close_panel.rolling(10).apply(
            lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )
        features['quantile_20'] = self.close_panel.rolling(20).apply(
            lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )
        
        # Z-Score
        features['zscore_10'] = (self.close_panel - ma10) / self.close_panel.rolling(10).std()
        features['zscore_20'] = (self.close_panel - ma20) / self.close_panel.rolling(20).std()
        
        # ==================== 相对强弱 (4个) ====================
        # 上涨天数比例
        features['up_ratio_5'] = (self.return_panel > 0).rolling(5).mean()
        features['up_ratio_10'] = (self.return_panel > 0).rolling(10).mean()
        features['up_ratio_20'] = (self.return_panel > 0).rolling(20).mean()
        
        # 连涨连跌
        features['streak'] = np.sign(self.return_panel).rolling(5).sum()
        
        # 清理无穷值
        for name in features:
            features[name] = features[name].replace([np.inf, -np.inf], np.nan)
        
        return features
    
    def split_train_test(self, train_ratio: float = 0.7) -> Tuple['PanelDataManager', 'PanelDataManager']:
        """
        按时间切分训练集和测试集
        
        Args:
            train_ratio: 训练集比例
        
        Returns:
            (训练集 PanelDataManager, 测试集 PanelDataManager)
        """
        if self.dates is None:
            raise ValueError("请先调用 fetch() 获取数据")
        
        split_idx = int(len(self.dates) * train_ratio)
        train_dates = self.dates[:split_idx]
        test_dates = self.dates[split_idx:]
        
        # 创建训练集
        train_dm = PanelDataManager(self.cache_dir)
        train_dm.open_panel = self.open_panel.loc[train_dates]
        train_dm.high_panel = self.high_panel.loc[train_dates]
        train_dm.low_panel = self.low_panel.loc[train_dates]
        train_dm.close_panel = self.close_panel.loc[train_dates]
        train_dm.volume_panel = self.volume_panel.loc[train_dates]
        train_dm.return_panel = self.return_panel.loc[train_dates]
        train_dm.symbols = self.symbols
        train_dm.dates = train_dates
        train_dm.start_date = str(train_dates[0].date())
        train_dm.end_date = str(train_dates[-1].date())
        
        # 创建测试集
        test_dm = PanelDataManager(self.cache_dir)
        test_dm.open_panel = self.open_panel.loc[test_dates]
        test_dm.high_panel = self.high_panel.loc[test_dates]
        test_dm.low_panel = self.low_panel.loc[test_dates]
        test_dm.close_panel = self.close_panel.loc[test_dates]
        test_dm.volume_panel = self.volume_panel.loc[test_dates]
        test_dm.return_panel = self.return_panel.loc[test_dates]
        test_dm.symbols = self.symbols
        test_dm.dates = test_dates
        test_dm.start_date = str(test_dates[0].date())
        test_dm.end_date = str(test_dates[-1].date())
        
        return train_dm, test_dm
    
    def info(self) -> Dict:
        """获取数据信息"""
        if self.close_panel is None:
            return {'status': 'empty'}
        
        return {
            'symbols': len(self.symbols),
            'days': len(self.dates),
            'start_date': self.start_date,
            'end_date': self.end_date,
            'missing_ratio': self.close_panel.isna().sum().sum() / self.close_panel.size,
        }


if __name__ == '__main__':
    # 测试
    dm = PanelDataManager()
    dm.fetch(pool_type='nasdaq100', use_cache=True)
    
    print("\n数据信息:")
    print(dm.info())
    
    print("\n收盘价面板 (前5行, 前5列):")
    print(dm.close_panel.iloc[:5, :5])
    
    print("\n未来1日收益 (前5行, 前5列):")
    print(dm.get_forward_return(1).iloc[:5, :5])
