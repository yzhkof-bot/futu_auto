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
import hashlib

warnings.filterwarnings('ignore')

# 导入统一股票池
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.stock_pool import get_stock_pool


class HistoricalDataCache:
    """
    历史数据缓存管理器
    
    特点：
    1. 精确到个股缓存，每只股票单独存储
    2. 支持增量下载，只下载缺失的股票
    3. 一次下载，永久使用
    
    使用方法：
        cache = HistoricalDataCache()
        # 首次运行会下载，后续直接从缓存读取
        dm = cache.get_data(start_date='2011-01-01', end_date='2015-12-31')
    """
    
    def __init__(self, cache_dir: str = None):
        """
        初始化
        
        Args:
            cache_dir: 缓存目录，默认 gp_alpha/.historical_cache
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), '.historical_cache')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 元数据文件
        self.meta_path = os.path.join(cache_dir, '_metadata.pkl')
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """加载元数据"""
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'rb') as f:
                return pickle.load(f)
        return {'cached_symbols': {}, 'version': '1.0'}
    
    def _save_metadata(self):
        """保存元数据"""
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def _get_cache_key(self, symbol: str, start_date: str, end_date: str) -> str:
        """生成缓存文件名"""
        # 使用日期范围作为 key，确保相同日期范围的数据可以复用
        key = f"{symbol}_{start_date}_{end_date}"
        return hashlib.md5(key.encode()).hexdigest()[:12]
    
    def _get_symbol_cache_path(self, symbol: str, start_date: str, end_date: str) -> str:
        """获取个股缓存路径"""
        cache_key = self._get_cache_key(symbol, start_date, end_date)
        return os.path.join(self.cache_dir, f"{symbol}_{cache_key}.pkl")
    
    def _is_cached(self, symbol: str, start_date: str, end_date: str) -> bool:
        """检查个股是否已缓存"""
        cache_path = self._get_symbol_cache_path(symbol, start_date, end_date)
        return os.path.exists(cache_path)
    
    def _save_symbol_data(self, symbol: str, data: pd.DataFrame, start_date: str, end_date: str):
        """保存个股数据"""
        cache_path = self._get_symbol_cache_path(symbol, start_date, end_date)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        
        # 更新元数据
        cache_key = f"{start_date}_{end_date}"
        if cache_key not in self.metadata['cached_symbols']:
            self.metadata['cached_symbols'][cache_key] = []
        if symbol not in self.metadata['cached_symbols'][cache_key]:
            self.metadata['cached_symbols'][cache_key].append(symbol)
        self._save_metadata()
    
    def _load_symbol_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """加载个股数据"""
        cache_path = self._get_symbol_cache_path(symbol, start_date, end_date)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def download_all_history(self,
                             symbols: List[str] = None,
                             pool_type: str = 'all',
                             force_download: bool = False,
                             verbose: bool = True,
                             delay_seconds: float = 0.5) -> Dict[str, pd.DataFrame]:
        """
        下载股票的全部历史数据（从上市到现在）
        
        使用 period='max' 获取所有可用数据，不管股票上市时间长短
        
        Args:
            symbols: 股票列表，None 则使用股票池
            pool_type: 股票池类型
            force_download: 是否强制重新下载
            verbose: 是否打印进度
            delay_seconds: 每次下载间隔秒数
        
        Returns:
            Dict[symbol, DataFrame] 个股数据字典
        """
        import time
        
        # 使用特殊的日期标记表示全部历史
        start_date = 'MAX'
        end_date = 'MAX'
        
        # 获取股票列表
        if symbols is None:
            symbols = get_stock_pool(pool_type)
        
        if verbose:
            print(f"=== 全部历史数据下载器 ===")
            print(f"模式: period='max' (下载所有可用历史)")
            print(f"股票池: {len(symbols)} 只")
        
        # 检查哪些股票需要下载
        cached_symbols = []
        missing_symbols = []
        
        for symbol in symbols:
            if not force_download and self._is_cached(symbol, start_date, end_date):
                cached_symbols.append(symbol)
            else:
                missing_symbols.append(symbol)
        
        if verbose:
            print(f"已缓存: {len(cached_symbols)} 只")
            print(f"需下载: {len(missing_symbols)} 只")
        
        # 逐个下载
        if missing_symbols:
            if verbose:
                print(f"\n开始下载 {len(missing_symbols)} 只股票的全部历史...")
                print(f"(每只间隔 {delay_seconds}s)")
            
            success_count = 0
            fail_count = 0
            
            for idx, symbol in enumerate(missing_symbols):
                if verbose and (idx + 1) % 10 == 0:
                    print(f"  进度: {idx + 1}/{len(missing_symbols)} ({success_count} 成功, {fail_count} 失败)")
                
                try:
                    ticker = yf.Ticker(symbol)
                    # period='max' 获取所有历史数据
                    data = ticker.history(period='max', auto_adjust=False)
                    
                    if not data.empty and len(data) > 50:
                        symbol_data = pd.DataFrame({
                            'Open': data['Open'],
                            'High': data['High'],
                            'Low': data['Low'],
                            'Close': data['Close'],
                            'Volume': data['Volume'],
                        })
                        self._save_symbol_data(symbol, symbol_data, start_date, end_date)
                        success_count += 1
                        
                        # 显示数据范围
                        first_date = data.index[0].strftime('%Y-%m-%d')
                        last_date = data.index[-1].strftime('%Y-%m-%d')
                        if verbose:
                            print(f"    ✓ {symbol}: {len(data)} 天 ({first_date} ~ {last_date})")
                    else:
                        fail_count += 1
                        if verbose:
                            print(f"    ✗ {symbol}: 数据不足")
                            
                except Exception as e:
                    fail_count += 1
                    if verbose:
                        print(f"    ✗ {symbol}: {str(e)[:50]}")
                
                time.sleep(delay_seconds)
            
            if verbose:
                print(f"\n下载完成: {success_count} 成功, {fail_count} 失败")
        
        # 加载所有缓存数据
        result = {}
        for symbol in symbols:
            data = self._load_symbol_data(symbol, start_date, end_date)
            if data is not None:
                result[symbol] = data
        
        if verbose:
            print(f"\n=== 缓存完成 ===")
            print(f"有效股票: {len(result)} 只")
            print(f"缓存目录: {self.cache_dir}")
        
        return result
    
    def get_all_history(self,
                        symbols: List[str] = None,
                        pool_type: str = 'all',
                        start_date: str = None,
                        end_date: str = None,
                        min_data_days: int = 200,
                        verbose: bool = True) -> 'PanelDataManager':
        """
        获取全部历史数据，支持按日期范围筛选
        
        Args:
            symbols: 股票列表
            pool_type: 股票池类型
            start_date: 筛选开始日期（可选，从缓存中筛选）
            end_date: 筛选结束日期（可选，从缓存中筛选）
            min_data_days: 最少数据天数
            verbose: 是否打印进度
        
        Returns:
            PanelDataManager 实例
        """
        # 下载/加载全部历史数据
        stock_data = self.download_all_history(
            symbols=symbols,
            pool_type=pool_type,
            verbose=verbose
        )
        
        if not stock_data:
            raise ValueError("没有获取到任何数据")
        
        # 按日期筛选
        if start_date or end_date:
            filtered_data = {}
            for symbol, data in stock_data.items():
                # 转换索引为日期（去掉时区）
                data.index = pd.to_datetime(data.index).tz_localize(None)
                
                if start_date:
                    data = data[data.index >= start_date]
                if end_date:
                    data = data[data.index <= end_date]
                
                if len(data) > 0:
                    filtered_data[symbol] = data
            stock_data = filtered_data
        
        # 构建 Panel 数据
        dm = PanelDataManager()
        dm.start_date = start_date or 'MAX'
        dm.end_date = end_date or 'MAX'
        
        open_dict = {}
        high_dict = {}
        low_dict = {}
        close_dict = {}
        volume_dict = {}
        
        for symbol, data in stock_data.items():
            if data['Close'].notna().sum() < min_data_days:
                continue
            open_dict[symbol] = data['Open']
            high_dict[symbol] = data['High']
            low_dict[symbol] = data['Low']
            close_dict[symbol] = data['Close']
            volume_dict[symbol] = data['Volume']
        
        if not close_dict:
            raise ValueError("没有满足最少数据天数要求的股票")
        
        dm.open_panel = pd.DataFrame(open_dict)
        dm.high_panel = pd.DataFrame(high_dict)
        dm.low_panel = pd.DataFrame(low_dict)
        dm.close_panel = pd.DataFrame(close_dict)
        dm.volume_panel = pd.DataFrame(volume_dict)
        
        # 对齐日期索引
        all_dates = dm.close_panel.index.union(dm.open_panel.index)
        dm.open_panel = dm.open_panel.reindex(all_dates)
        dm.high_panel = dm.high_panel.reindex(all_dates)
        dm.low_panel = dm.low_panel.reindex(all_dates)
        dm.close_panel = dm.close_panel.reindex(all_dates)
        dm.volume_panel = dm.volume_panel.reindex(all_dates)
        
        dm.return_panel = dm.close_panel.pct_change()
        dm.symbols = list(dm.close_panel.columns)
        dm.dates = dm.close_panel.index
        
        if verbose:
            print(f"\n数据加载完成:")
            print(f"  有效股票: {len(dm.symbols)} 只")
            print(f"  数据天数: {len(dm.dates)} 天")
            if len(dm.dates) > 0:
                print(f"  日期范围: {dm.dates[0].strftime('%Y-%m-%d')} ~ {dm.dates[-1].strftime('%Y-%m-%d')}")
        
        return dm

    def download_and_cache(self, 
                           symbols: List[str] = None,
                           start_date: str = '2011-01-01',
                           end_date: str = '2015-12-31',
                           pool_type: str = 'all',
                           force_download: bool = False,
                           verbose: bool = True,
                           retry_count: int = 3,
                           delay_seconds: float = 0.5) -> Dict[str, pd.DataFrame]:
        """
        下载并缓存历史数据
        
        Args:
            symbols: 股票列表，None 则使用股票池
            start_date: 开始日期
            end_date: 结束日期
            pool_type: 股票池类型
            force_download: 是否强制重新下载
            verbose: 是否打印进度
            retry_count: 下载失败重试次数
            delay_seconds: 每次下载间隔秒数（避免限速）
        
        Returns:
            Dict[symbol, DataFrame] 个股数据字典
        """
        import time
        
        # 获取股票列表
        if symbols is None:
            symbols = get_stock_pool(pool_type)
        
        if verbose:
            print(f"=== 历史数据缓存管理器 ===")
            print(f"日期范围: {start_date} ~ {end_date}")
            print(f"股票池: {len(symbols)} 只")
        
        # 检查哪些股票需要下载
        cached_symbols = []
        missing_symbols = []
        
        for symbol in symbols:
            if not force_download and self._is_cached(symbol, start_date, end_date):
                cached_symbols.append(symbol)
            else:
                missing_symbols.append(symbol)
        
        if verbose:
            print(f"已缓存: {len(cached_symbols)} 只")
            print(f"需下载: {len(missing_symbols)} 只")
        
        # 逐个下载缺失的股票（避免限速）
        if missing_symbols:
            if verbose:
                print(f"\n开始逐个下载 {len(missing_symbols)} 只股票...")
                print(f"(每只间隔 {delay_seconds}s，预计 {len(missing_symbols) * delay_seconds / 60:.1f} 分钟)")
            
            success_count = 0
            fail_count = 0
            
            for idx, symbol in enumerate(missing_symbols):
                if verbose and (idx + 1) % 10 == 0:
                    print(f"  进度: {idx + 1}/{len(missing_symbols)} ({success_count} 成功, {fail_count} 失败)")
                
                # 重试机制
                for attempt in range(retry_count):
                    try:
                        # 单股票下载
                        ticker = yf.Ticker(symbol)
                        data = ticker.history(start=start_date, end=end_date, auto_adjust=False)
                        
                        if not data.empty and len(data) > 100:
                            # 标准化列名
                            symbol_data = pd.DataFrame({
                                'Open': data['Open'],
                                'High': data['High'],
                                'Low': data['Low'],
                                'Close': data['Close'],
                                'Volume': data['Volume'],
                            })
                            self._save_symbol_data(symbol, symbol_data, start_date, end_date)
                            success_count += 1
                            if verbose:
                                print(f"    ✓ {symbol}: {len(data)} 天")
                            break
                        else:
                            if attempt == retry_count - 1:
                                fail_count += 1
                                if verbose:
                                    print(f"    ✗ {symbol}: 数据不足")
                                    
                    except Exception as e:
                        if attempt < retry_count - 1:
                            time.sleep(delay_seconds * 2)  # 失败后等待更长时间
                        else:
                            fail_count += 1
                            if verbose:
                                print(f"    ✗ {symbol}: {str(e)[:50]}")
                
                # 下载间隔
                time.sleep(delay_seconds)
            
            if verbose:
                print(f"\n下载完成: {success_count} 成功, {fail_count} 失败")
        
        # 加载所有缓存数据
        result = {}
        for symbol in symbols:
            data = self._load_symbol_data(symbol, start_date, end_date)
            if data is not None:
                result[symbol] = data
        
        if verbose:
            print(f"\n=== 缓存完成 ===")
            print(f"有效股票: {len(result)} 只")
            print(f"缓存目录: {self.cache_dir}")
        
        return result
    
    def get_data(self,
                 symbols: List[str] = None,
                 start_date: str = '2011-01-01',
                 end_date: str = '2015-12-31',
                 pool_type: str = 'all',
                 min_data_days: int = 200,
                 verbose: bool = True) -> 'PanelDataManager':
        """
        获取数据，自动使用缓存
        
        Args:
            symbols: 股票列表
            start_date: 开始日期
            end_date: 结束日期
            pool_type: 股票池类型
            min_data_days: 最少数据天数
            verbose: 是否打印进度
        
        Returns:
            PanelDataManager 实例
        """
        # 下载/加载数据
        stock_data = self.download_and_cache(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            pool_type=pool_type,
            verbose=verbose
        )
        
        if not stock_data:
            raise ValueError("没有获取到任何数据")
        
        # 构建 Panel 数据
        dm = PanelDataManager()
        dm.start_date = start_date
        dm.end_date = end_date
        
        # 合并所有股票数据
        open_dict = {}
        high_dict = {}
        low_dict = {}
        close_dict = {}
        volume_dict = {}
        
        for symbol, data in stock_data.items():
            # 过滤数据不足的股票
            if data['Close'].notna().sum() < min_data_days:
                continue
            open_dict[symbol] = data['Open']
            high_dict[symbol] = data['High']
            low_dict[symbol] = data['Low']
            close_dict[symbol] = data['Close']
            volume_dict[symbol] = data['Volume']
        
        if not close_dict:
            raise ValueError("没有满足最少数据天数要求的股票")
        
        # 构建 DataFrame
        dm.open_panel = pd.DataFrame(open_dict)
        dm.high_panel = pd.DataFrame(high_dict)
        dm.low_panel = pd.DataFrame(low_dict)
        dm.close_panel = pd.DataFrame(close_dict)
        dm.volume_panel = pd.DataFrame(volume_dict)
        
        # 对齐日期索引
        all_dates = dm.close_panel.index.union(dm.open_panel.index)
        dm.open_panel = dm.open_panel.reindex(all_dates)
        dm.high_panel = dm.high_panel.reindex(all_dates)
        dm.low_panel = dm.low_panel.reindex(all_dates)
        dm.close_panel = dm.close_panel.reindex(all_dates)
        dm.volume_panel = dm.volume_panel.reindex(all_dates)
        
        # 计算收益率
        dm.return_panel = dm.close_panel.pct_change()
        
        # 更新元数据
        dm.symbols = list(dm.close_panel.columns)
        dm.dates = dm.close_panel.index
        
        if verbose:
            print(f"\n数据加载完成:")
            print(f"  有效股票: {len(dm.symbols)} 只")
            print(f"  数据天数: {len(dm.dates)} 天")
        
        return dm
    
    def get_cached_info(self) -> Dict:
        """获取缓存信息"""
        info = {
            'cache_dir': self.cache_dir,
            'date_ranges': {},
        }
        
        for date_range, symbols in self.metadata.get('cached_symbols', {}).items():
            info['date_ranges'][date_range] = len(symbols)
        
        # 计算缓存大小
        total_size = 0
        for f in os.listdir(self.cache_dir):
            fp = os.path.join(self.cache_dir, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
        info['total_size_mb'] = round(total_size / 1024 / 1024, 2)
        
        return info
    
    def clear_cache(self, date_range: str = None):
        """
        清除缓存
        
        Args:
            date_range: 指定日期范围 (如 '2011-01-01_2015-12-31')，None 则清除所有
        """
        if date_range is None:
            # 清除所有
            import shutil
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            self.metadata = {'cached_symbols': {}, 'version': '1.0'}
            self._save_metadata()
            print("已清除所有缓存")
        else:
            # 清除指定日期范围
            if date_range in self.metadata.get('cached_symbols', {}):
                symbols = self.metadata['cached_symbols'][date_range]
                start_date, end_date = date_range.split('_')
                for symbol in symbols:
                    cache_path = self._get_symbol_cache_path(symbol, start_date, end_date)
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
                del self.metadata['cached_symbols'][date_range]
                self._save_metadata()
                print(f"已清除 {date_range} 的缓存")


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
        - macro_data: Dict[str, pd.Series] 宏观指标数据
    """
    
    # 宏观指标配置
    MACRO_SYMBOLS = {
        '^TNX': 'us10y',      # 10年期美债收益率
        'DX-Y.NYB': 'dxy',    # 美元指数
        '^RUT': 'rut',        # 罗素2000
        '^GSPC': 'spx',       # 标普500
        '^VIX': 'vix',        # 恐慌指数
        'CL=F': 'oil',        # 原油期货
        'GC=F': 'gold',       # 黄金期货
    }
    
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
        
        # 宏观数据
        self.macro_data: Dict[str, pd.Series] = {}
        
        # 元数据
        self.symbols: List[str] = []
        self.dates: pd.DatetimeIndex = None
        self.start_date: str = None
        self.end_date: str = None
    
    def _fetch_macro_data(self, start_date: str, end_date: str, verbose: bool = True) -> Dict[str, pd.Series]:
        """
        获取宏观指标数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            verbose: 是否打印进度
        
        Returns:
            Dict[name, Series] 宏观数据字典
        """
        macro_cache_path = os.path.join(self.cache_dir, f"macro_{start_date}_{end_date}.pkl")
        
        # 尝试从缓存加载
        if os.path.exists(macro_cache_path):
            if verbose:
                print("  加载宏观数据缓存...")
            with open(macro_cache_path, 'rb') as f:
                return pickle.load(f)
        
        if verbose:
            print("  下载宏观指标数据...")
        
        macro_data = {}
        symbols_list = list(self.MACRO_SYMBOLS.keys())
        
        try:
            data = yf.download(symbols_list, start=start_date, end=end_date, progress=False)
            
            if not data.empty:
                for symbol, name in self.MACRO_SYMBOLS.items():
                    try:
                        if isinstance(data.columns, pd.MultiIndex):
                            close = data['Close'][symbol]
                        else:
                            close = data['Close']
                        
                        if close.notna().sum() > 50:
                            macro_data[name] = close
                            if verbose:
                                print(f"    ✓ {name} ({symbol}): {close.notna().sum()} 天")
                    except Exception as e:
                        if verbose:
                            print(f"    ✗ {name} ({symbol}): {str(e)[:30]}")
        except Exception as e:
            if verbose:
                print(f"  宏观数据下载失败: {e}")
        
        # 保存缓存
        if macro_data:
            with open(macro_cache_path, 'wb') as f:
                pickle.dump(macro_data, f)
        
        return macro_data
    
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
        
        # 获取宏观数据
        self.macro_data = self._fetch_macro_data(start_date, end_date, verbose)
        
        if verbose:
            print(f"有效股票: {len(self.symbols)} 只")
            print(f"数据天数: {len(self.dates)} 天")
            print(f"宏观指标: {len(self.macro_data)} 个")
        
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
            'macro_data': self.macro_data,
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
        self.macro_data = cache_data.get('macro_data', {})
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
        # 第十类：宏观指标特征 - 相对强弱 & 动态相关性
        # ============================================================
        if self.macro_data:
            stock_return = self.return_panel  # 个股日收益率
            
            # --- 10年期美债收益率 (^TNX) ---
            if 'us10y' in self.macro_data:
                us10y = self.macro_data['us10y'].reindex(self.dates).ffill()
                # 美债收益率变化（核心！利率上升对科技股杀伤力大）
                us10y_change = us10y.diff()
                features['us10y_change'] = self._broadcast_to_panel(us10y_change)
                # 美债收益率水平（归一化）
                features['us10y_level'] = self._broadcast_to_panel(us10y / 10)  # 除以10归一化
                # 个股与利率变化的动态相关性（利率敏感度）
                features['rate_sensitivity'] = self._rolling_corr_with_macro(
                    stock_return, us10y_change, window=20
                )
            
            # --- 美元指数 (DX-Y.NYB) ---
            if 'dxy' in self.macro_data:
                dxy = self.macro_data['dxy'].reindex(self.dates).ffill()
                dxy_return = dxy.pct_change()
                # 美元变化
                features['dxy_change'] = self._broadcast_to_panel(dxy_return)
                # 个股与美元的动态相关性（抗美元 or 随美元）
                features['dxy_sensitivity'] = self._rolling_corr_with_macro(
                    stock_return, dxy_return, window=20
                )
            
            # --- 罗素2000 (^RUT) vs 标普500 (^GSPC) ---
            if 'rut' in self.macro_data and 'spx' in self.macro_data:
                rut = self.macro_data['rut'].reindex(self.dates).ffill()
                spx = self.macro_data['spx'].reindex(self.dates).ffill()
                # Risk-On/Off 比率（罗素/标普）
                risk_ratio = rut / spx
                features['risk_on_off'] = self._broadcast_to_panel(
                    risk_ratio / risk_ratio.rolling(20).mean() - 1
                )
                # 个股相对罗素2000（跑赢小盘？）
                features['vs_rut'] = self.close_panel.div(rut, axis=0)
                features['vs_rut'] = features['vs_rut'] / features['vs_rut'].rolling(20).mean() - 1
                # 个股相对标普500
                features['vs_spx'] = self.close_panel.div(spx, axis=0)
                features['vs_spx'] = features['vs_spx'] / features['vs_spx'].rolling(20).mean() - 1
            
            # --- VIX 恐慌指数 ---
            if 'vix' in self.macro_data:
                vix = self.macro_data['vix'].reindex(self.dates).ffill()
                # VIX 水平（归一化）
                features['vix_level'] = self._broadcast_to_panel(vix / 50)  # 通常 10-50
                # VIX 变化（恐慌加剧/缓解）
                features['vix_change'] = self._broadcast_to_panel(vix.pct_change())
                # 个股与 VIX 的相关性（避险属性）
                features['vix_sensitivity'] = self._rolling_corr_with_macro(
                    stock_return, vix.pct_change(), window=20
                )
            
            # --- 原油 (CL=F) ---
            if 'oil' in self.macro_data:
                oil = self.macro_data['oil'].reindex(self.dates).ffill()
                oil_return = oil.pct_change()
                # 油价变化
                features['oil_change'] = self._broadcast_to_panel(oil_return)
                # 个股相对原油（能源敏感度）
                features['vs_oil'] = self.close_panel.div(oil, axis=0)
                features['vs_oil'] = features['vs_oil'] / features['vs_oil'].rolling(20).mean() - 1
            
            # --- 黄金 (GC=F) ---
            if 'gold' in self.macro_data:
                gold = self.macro_data['gold'].reindex(self.dates).ffill()
                gold_return = gold.pct_change()
                # 个股相对黄金（避险对比）
                features['vs_gold'] = self.close_panel.div(gold, axis=0)
                features['vs_gold'] = features['vs_gold'] / features['vs_gold'].rolling(20).mean() - 1
                # 个股与黄金相关性
                features['gold_sensitivity'] = self._rolling_corr_with_macro(
                    stock_return, gold_return, window=20
                )
        
        # ============================================================
        # 清理无穷值
        # ============================================================
        for name in features:
            features[name] = features[name].replace([np.inf, -np.inf], np.nan)
        
        return features
    
    def _broadcast_to_panel(self, series: pd.Series) -> pd.DataFrame:
        """
        将宏观指标 Series 广播为 Panel（每只股票相同值）
        
        Args:
            series: 宏观指标时间序列
        
        Returns:
            DataFrame (日期 × 股票)
        """
        return pd.DataFrame(
            np.tile(series.values.reshape(-1, 1), (1, len(self.symbols))),
            index=self.dates,
            columns=self.symbols
        )
    
    def _rolling_corr_with_macro(self, stock_return: pd.DataFrame, 
                                  macro_change: pd.Series, 
                                  window: int = 20) -> pd.DataFrame:
        """
        计算个股与宏观指标的滚动相关性
        
        Args:
            stock_return: 个股收益率 Panel
            macro_change: 宏观指标变化 Series
            window: 滚动窗口
        
        Returns:
            DataFrame (日期 × 股票) 相关性面板
        """
        # 对齐索引
        macro_aligned = macro_change.reindex(self.dates).ffill()
        
        # 逐列计算滚动相关
        corr_dict = {}
        for symbol in self.symbols:
            stock_ret = stock_return[symbol]
            corr_dict[symbol] = stock_ret.rolling(window).corr(macro_aligned)
        
        return pd.DataFrame(corr_dict)
    
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
        train_dm.macro_data = self.macro_data  # 宏观数据共享（会按日期自动切分）
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
        test_dm.macro_data = self.macro_data  # 宏观数据共享
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
    import argparse
    
    parser = argparse.ArgumentParser(description='数据管理器测试')
    parser.add_argument('--mode', type=str, default='historical', 
                        choices=['historical', 'panel', 'info'],
                        help='运行模式: historical=历史缓存, panel=面板数据, info=缓存信息')
    parser.add_argument('--start', type=str, default='2011-01-01', help='开始日期')
    parser.add_argument('--end', type=str, default='2015-12-31', help='结束日期')
    parser.add_argument('--pool', type=str, default='all', help='股票池类型')
    parser.add_argument('--force', action='store_true', help='强制重新下载')
    
    args = parser.parse_args()
    
    if args.mode == 'historical':
        # 测试历史数据缓存
        print("=" * 60)
        print("历史数据缓存测试")
        print("=" * 60)
        
        cache = HistoricalDataCache()
        
        # 获取数据（首次会下载，后续从缓存读取）
        dm = cache.get_data(
            start_date=args.start,
            end_date=args.end,
            pool_type=args.pool,
            verbose=True
        )
        
        print("\n" + "=" * 60)
        print("数据信息:")
        print(dm.info())
        
        print("\n收盘价面板 (前5行, 前5列):")
        print(dm.close_panel.iloc[:5, :5])
        
        print("\n缓存信息:")
        print(cache.get_cached_info())
        
    elif args.mode == 'panel':
        # 原有的 Panel 数据测试
        dm = PanelDataManager()
        dm.fetch(pool_type='nasdaq100', use_cache=True)
        
        print("\n数据信息:")
        print(dm.info())
        
        print("\n收盘价面板 (前5行, 前5列):")
        print(dm.close_panel.iloc[:5, :5])
        
        print("\n未来1日收益 (前5行, 前5列):")
        print(dm.get_forward_return(1).iloc[:5, :5])
        
    elif args.mode == 'info':
        # 查看缓存信息
        cache = HistoricalDataCache()
        info = cache.get_cached_info()
        print("缓存信息:")
        print(f"  缓存目录: {info['cache_dir']}")
        print(f"  缓存大小: {info['total_size_mb']} MB")
        print("  日期范围:")
        for date_range, count in info['date_ranges'].items():
            print(f"    {date_range}: {count} 只股票")
