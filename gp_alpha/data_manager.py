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
        工业级 GP 特征面板
        
        设计原则：
        1. 只提供"原材料"，让 GP 自己组合（不提供 MA/MACD 等熟食）
        2. 高信息密度、低相关性
        3. 归一化处理，便于 GP 跨股票比较
        
        Returns:
            特征名 -> DataFrame 的字典
        """
        if self.close_panel is None:
            raise ValueError("请先调用 fetch() 获取数据")
        
        features = {}
        
        # ============================================================
        # 第一类：原始价量数据 (6个) - GP 的核心原材料
        # ============================================================
        features['open'] = self.open_panel
        features['high'] = self.high_panel
        features['low'] = self.low_panel
        features['close'] = self.close_panel
        features['volume'] = self.volume_panel
        # 成交额 = 价格 × 成交量（比单独的 volume 更有意义）
        features['amount'] = self.close_panel * self.volume_panel
        
        # ============================================================
        # 第二类：VWAP 相关 (3个) - 机构成本线，极强因子
        # ============================================================
        # 真实 VWAP = 成交额 / 成交量
        vwap = features['amount'] / self.volume_panel
        features['vwap'] = vwap
        # 价格相对 VWAP 偏离（核心因子！）
        features['close_vwap'] = self.close_panel / vwap - 1
        # 开盘相对 VWAP（隔夜情绪）
        features['open_vwap'] = self.open_panel / vwap - 1
        
        # ============================================================
        # 第三类：日内微观结构 (5个) - 多空博弈信息
        # ============================================================
        daily_range = self.high_panel - self.low_panel
        
        # 振幅（多空分歧强度）- 归一化
        features['hl_range'] = daily_range / self.close_panel
        
        # 日内位置（收盘在当天的位置，0=最低，1=最高）
        features['intraday_pos'] = (self.close_panel - self.low_panel) / daily_range.replace(0, np.nan)
        
        # 上影线比例（卖压）
        features['upper_shadow'] = (self.high_panel - np.maximum(self.open_panel, self.close_panel)) / daily_range.replace(0, np.nan)
        
        # 下影线比例（买盘支撑）
        features['lower_shadow'] = (np.minimum(self.open_panel, self.close_panel) - self.low_panel) / daily_range.replace(0, np.nan)
        
        # 实体比例（趋势强度）
        features['body_ratio'] = (self.close_panel - self.open_panel).abs() / daily_range.replace(0, np.nan)
        
        # ============================================================
        # 第四类：跳空与隔夜 (2个) - 隔夜信息冲击
        # ============================================================
        # 跳空幅度
        features['gap'] = self.open_panel / self.close_panel.shift(1) - 1
        # 跳空方向（归一化）
        features['gap_direction'] = np.sign(self.open_panel - self.close_panel.shift(1))
        
        # ============================================================
        # 第五类：收益率 (1个) - 只保留 log return，其他让 GP 自己算
        # ============================================================
        # Log return（更符合统计假设，可加性）
        features['log_return'] = np.log(self.close_panel / self.close_panel.shift(1))
        
        # ============================================================
        # 第六类：成交量特征 (4个) - 归一化的量能信息
        # ============================================================
        # 量比（当日成交量 / 5日均量）- 放量/缩量信号
        vol_ma5 = self.volume_panel.rolling(5).mean()
        features['vol_ratio'] = self.volume_panel / vol_ma5
        
        # 量的变化率
        features['vol_change'] = self.volume_panel / self.volume_panel.shift(1) - 1
        
        # 成交额比（比量比更稳定）
        amount_ma5 = features['amount'].rolling(5).mean()
        features['amount_ratio'] = features['amount'] / amount_ma5
        
        # 量价背离指标（价涨量缩 or 价跌量增）
        price_direction = np.sign(self.close_panel - self.close_panel.shift(1))
        vol_direction = np.sign(self.volume_panel - self.volume_panel.shift(1))
        features['vol_price_diverge'] = price_direction * vol_direction  # -1=背离, 1=同向
        
        # ============================================================
        # 第七类：波动率 (2个) - 让 GP 自己选周期
        # ============================================================
        # 已实现波动率（用 log return 计算更准确）
        features['realized_vol'] = features['log_return'].rolling(10).std()
        
        # 波动率的波动率（波动率聚集效应）
        features['vol_of_vol'] = features['realized_vol'].rolling(10).std() / features['realized_vol'].rolling(10).mean()
        
        # ============================================================
        # 第八类：真实波幅 ATR (1个) - 归一化
        # ============================================================
        tr1 = self.high_panel - self.low_panel
        tr2 = (self.high_panel - self.close_panel.shift(1)).abs()
        tr3 = (self.low_panel - self.close_panel.shift(1)).abs()
        true_range = np.maximum(np.maximum(tr1, tr2), tr3)
        # ATR 归一化（除以价格）
        features['atr_norm'] = true_range.rolling(14).mean() / self.close_panel
        
        # ============================================================
        # 第九类：RSI (1个) - 唯一保留的技术指标（自带归一化 0-100）
        # ============================================================
        delta = self.close_panel.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        # 归一化到 -1 到 1（比 0-100 更适合 GP）
        features['rsi'] = (100 - (100 / (1 + rs))) / 50 - 1
        
        # ============================================================
        # 清理无穷值
        # ============================================================
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
