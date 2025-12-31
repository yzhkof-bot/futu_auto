# 趋势跟踪策略回测框架

一个专业级的趋势跟踪策略回测框架，集成凯利公式仓位管理、yfinance数据源和专业可视化报告。

## 🚀 特性

### 核心功能
- **多策略支持**: 移动平均、突破、趋势跟踪等经典策略
- **凯利公式集成**: 动态仓位管理和风险控制
- **专业回测引擎**: 现实的滑点、手续费和执行模型
- **全面性能分析**: 60+ 专业指标和风险分析
- **高质量可视化**: 交互式图表和专业报告

### 数据处理
- **稳健数据获取**: 基于yfinance的缓存和错误处理
- **技术指标库**: 20+ 常用技术指标
- **数据质量验证**: 自动清洗和异常检测
- **多时间框架**: 支持日线、小时线等多种周期

### 风险管理
- **凯利公式**: 动态参数估计和保守缩放
- **ATR止损**: 基于波动性的自适应止损
- **组合风险**: 最大回撤控制和仓位限制
- **压力测试**: VaR、尾部风险和情景分析

### 性能优化
- **并行计算**: 多核参数优化
- **向量化回测**: 高效的信号生成和回测
- **内存优化**: 大数据集分块处理
- **智能缓存**: 减少重复计算

## 📦 安装

### 环境要求
- Python 3.8+
- 推荐使用虚拟环境

### 依赖安装
```bash
# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
pip install --no-user -r requirements.txt
```

### 主要依赖
```
pandas>=1.5.0
numpy>=1.21.0
yfinance>=0.2.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
scipy>=1.9.0
scikit-learn>=1.1.0
```

## 🎯 快速开始

### 简单回测示例
```python
from src.data.data_fetcher import DataFetcher
from src.strategies.ma_strategy import MovingAverageStrategy
from src.backtest.engine import BacktestEngine

# 1. 获取数据
fetcher = DataFetcher()
data = fetcher.fetch_stock_data("AAPL", "2022-01-01", "2023-12-31")

# 2. 创建策略
strategy = MovingAverageStrategy(short_period=10, long_period=30)

# 3. 运行回测
engine = BacktestEngine(initial_capital=100000)
results = engine.run_backtest(data, strategy, "AAPL")

# 4. 查看结果
print(f"总收益: {results['total_return_pct']:.2f}%")
print(f"夏普比率: {results['sharpe_ratio']:.3f}")
```

### 完整分析示例
```python
# 运行完整的策略分析
python examples/complete_backtest_example.py
```

## 📊 策略类型

### 1. 移动平均策略 (MovingAverageStrategy)
经典的双均线策略，结合RSI过滤
```python
strategy = MovingAverageStrategy(
    short_period=10,      # 短期均线
    long_period=30,       # 长期均线
    rsi_lower=30,         # RSI下限
    rsi_upper=70          # RSI上限
)
```

### 2. 突破策略 (BreakoutStrategy)  
价格突破策略，包含成交量确认
```python
strategy = BreakoutStrategy(
    breakout_period=20,        # 突破周期
    min_volume_ratio=1.5,      # 最小成交量比率
    rsi_momentum_threshold=50   # RSI动量阈值
)
```

### 3. 趋势跟踪策略 (TrendFollowingStrategy)
多指标趋势确认系统
```python
strategy = TrendFollowingStrategy(
    ema_fast=10,          # 快速EMA
    ema_medium=20,        # 中期EMA  
    ema_slow=50,          # 慢速EMA
    rsi_lower=30,         # RSI下限
    rsi_upper=70          # RSI上限
)
```

## 🔧 高级功能

### 参数优化
```python
from src.utils.parameter_tuner import ParameterTuner

# 定义参数范围
bounds = {
    'short_period': (5, 20),
    'long_period': (20, 50),
    'rsi_lower': (20, 40),
    'rsi_upper': (60, 80)
}

# 创建优化器
tuner = ParameterTuner(objective_function, bounds)

# 运行优化
result = tuner.differential_evolution_search(maxiter=100)
print(f"最优参数: {result.best_params}")
```

### 并行回测
```python
from src.utils.performance_optimizer import PerformanceOptimizer

optimizer = PerformanceOptimizer(n_jobs=4)
results = optimizer.parallel_backtest(
    backtest_func, parameter_sets, data
)
```

### 走势分析
```python
from src.analytics.analyzer import StrategyAnalyzer

analyzer = StrategyAnalyzer()
analysis = analyzer.analyze_strategy(backtest_results, benchmark_data)

# 生成报告
print(analyzer.generate_performance_report(analysis))
```

## 📈 可视化和报告

### 图表生成
```python
from src.visualization.charts import ChartGenerator

charts = ChartGenerator(style='professional')

# 净值曲线
fig1 = charts.create_equity_curve(equity_curve, benchmark)

# 回撤分析  
fig2 = charts.create_drawdown_chart(equity_curve)

# 交易分析
fig3 = charts.create_trade_analysis_chart(trades)
```

### 报告导出
```python
from src.visualization.reports import ReportGenerator

reporter = ReportGenerator(output_dir="./reports")

# HTML报告
html_path = reporter.generate_html_report(analysis, results, "策略名称")

# CSV导出
csv_path = reporter.generate_csv_export(results, "策略名称")
```

## 🏗️ 项目结构

```
FUTU_auto/
├── src/                          # 核心框架代码
│   ├── data/                     # 数据模块
│   │   ├── data_fetcher.py       # 数据获取
│   │   └── data_processor.py     # 数据处理
│   ├── strategies/               # 策略模块
│   │   ├── base_strategy.py      # 策略基类
│   │   ├── ma_strategy.py        # 移动平均策略
│   │   ├── breakout_strategy.py  # 突破策略
│   │   └── trend_following_strategy.py # 趋势跟踪策略
│   ├── backtest/                 # 回测模块
│   │   ├── engine.py             # 回测引擎
│   │   ├── metrics.py            # 性能指标
│   │   └── portfolio.py          # 组合管理
│   ├── analytics/                # 分析模块
│   │   ├── analyzer.py           # 策略分析器
│   │   └── risk_analyzer.py      # 风险分析器
│   ├── visualization/            # 可视化模块
│   │   ├── charts.py             # 图表生成
│   │   └── reports.py            # 报告生成
│   └── utils/                    # 工具模块
│       ├── performance_optimizer.py # 性能优化
│       └── parameter_tuner.py    # 参数调优
├── examples/                     # 示例代码
│   ├── complete_backtest_example.py    # 完整示例
│   ├── quick_start_example.py          # 快速开始
│   └── strategy_optimization_example.py # 参数优化示例
├── .codebuddy/skills/           # 专业技能模块
│   └── trend-backtest/          # 趋势回测技能
└── reports/                     # 输出报告目录
```

## 📋 使用示例

### 1. 快速回测
```bash
python examples/quick_start_example.py
```

### 2. 完整分析
```bash  
python examples/complete_backtest_example.py
```

### 3. 参数优化
```bash
python examples/strategy_optimization_example.py
```

## 🔍 性能指标

框架提供60+专业指标，包括：

### 收益指标
- 总收益率、年化收益率
- 复合年增长率(CAGR)
- 超额收益、相对收益

### 风险指标  
- 夏普比率、索提诺比率、卡玛比率
- 最大回撤、平均回撤
- 波动率、下行波动率
- VaR、条件VaR

### 交易指标
- 胜率、盈亏比、利润因子
- 平均持仓期、交易频率
- 最大连续亏损、最大连续盈利

### 基准比较
- Beta、Alpha、相关性
- 信息比率、跟踪误差
- 上涨捕获率、下跌捕获率

## ⚙️ 配置选项

### 回测引擎配置
```python
engine = BacktestEngine(
    initial_capital=100000,     # 初始资金
    commission_rate=0.001,      # 手续费率
    slippage_rate=0.0005,       # 滑点率
    max_position_size=0.25,     # 最大仓位
    use_kelly_sizing=True,      # 使用凯利公式
    kelly_scaling=0.25          # 凯利缩放因子
)
```

### 数据获取配置
```python
fetcher = DataFetcher(
    cache_dir=".cache",         # 缓存目录
    cache_expiry_hours=24       # 缓存过期时间
)
```

## 🚨 注意事项

### 数据质量
- 确保网络连接稳定，yfinance可能有访问限制
- 建议使用缓存减少重复请求
- 定期清理缓存以获取最新数据

### 回测假设
- 回测结果基于历史数据，不保证未来表现
- 考虑了交易成本但可能与实际有差异
- 流动性假设可能在极端市场条件下不成立

### 风险管理
- 凯利公式基于历史统计，参数会变化
- 建议使用保守的凯利缩放因子(0.25或更小)
- 实盘交易前请充分验证策略稳健性

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

### 代码规范
- 使用 Black 格式化代码
- 添加适当的文档字符串
- 编写单元测试
- 遵循 PEP 8 规范

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [yfinance](https://github.com/ranaroussi/yfinance) - 金融数据获取
- [pandas](https://pandas.pydata.org/) - 数据处理
- [matplotlib](https://matplotlib.org/) - 数据可视化
- [plotly](https://plotly.com/) - 交互式图表

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 创建 Issue
- 发送邮件
- 加入讨论群

---

**免责声明**: 本框架仅用于教育和研究目的。投资有风险，请谨慎决策。过往表现不代表未来收益。