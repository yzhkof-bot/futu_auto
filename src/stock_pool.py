"""
股票池定义模块
统一管理所有扫描脚本使用的股票池
"""

# 纳斯达克100成分股
NASDAQ100 = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'COST',
    'NFLX', 'TMUS', 'ASML', 'CSCO', 'ADBE', 'AMD', 'PEP', 'LIN', 'INTC', 'INTU',
    'TXN', 'CMCSA', 'QCOM', 'AMGN', 'AMAT', 'HON', 'ISRG', 'BKNG', 'SBUX', 'VRTX',
    'GILD', 'ADP', 'MDLZ', 'ADI', 'REGN', 'PANW', 'SNPS', 'LRCX', 'KLAC', 'CDNS',
    'MU', 'MELI', 'PYPL', 'MAR', 'ORLY', 'MNST', 'CTAS', 'NXPI', 'MCHP', 'FTNT',
    'ABNB', 'PCAR', 'KDP', 'AEP', 'PAYX', 'KHC', 'ODFL', 'CPRT', 'CHTR', 'ROST',
    'IDXX', 'DXCM', 'FAST', 'AZN', 'MRNA', 'EA', 'CTSH', 'EXC', 'VRSK', 'CSGP',
    'XEL', 'BKR', 'GEHC', 'FANG', 'TTWO', 'BIIB', 'ON', 'DLTR', 'WBD',
    'CDW', 'ZS', 'ILMN', 'MDB', 'TEAM', 'DDOG', 'GFS', 'LCID', 'SIRI',
    'CEG', 'CRWD', 'DASH', 'SMCI', 'ARM', 'COIN', 'TTD', 'PDD', 'LULU', 'WDAY'
]

# 精选非科技蓝筹股（道琼斯+标普500）
BLUECHIP_NON_TECH = [
    # 金融 (8只)
    'JPM',   # 摩根大通 - 美国最大银行
    'V',     # Visa - 支付双寡头
    'MA',    # 万事达 - 支付双寡头
    'GS',    # 高盛 - 顶级投行
    'BRK-B', # 伯克希尔 - 巴菲特旗舰
    'AXP',   # 美国运通 - 高端信用卡
    'BLK',   # 贝莱德 - 全球最大资管
    'SPGI',  # 标普全球 - 评级+数据垄断
    # 医疗健康 (6只)
    'UNH',   # 联合健康 - 医保龙头
    'LLY',   # 礼来 - 减肥药龙头
    'JNJ',   # 强生 - 医药+消费
    'MRK',   # 默沙东 - 肿瘤药龙头
    'ABBV',  # 艾伯维 - 免疫药龙头
    'TMO',   # 赛默飞 - 生命科学设备龙头
    # 消费 (6只)
    'WMT',   # 沃尔玛 - 零售之王
    'HD',    # 家得宝 - 家居零售龙头
    'PG',    # 宝洁 - 日用品之王
    'MCD',   # 麦当劳 - 快餐之王
    'NKE',   # 耐克 - 运动品牌龙头
    'KO',    # 可口可乐 - 饮料之王
    # 工业 (5只)
    'CAT',   # 卡特彼勒 - 工程机械龙头
    'BA',    # 波音 - 航空双寡头
    'GE',    # 通用电气 - 航空发动机
    'UNP',   # 联合太平洋 - 铁路龙头
    'RTX',   # 雷神 - 国防军工
    # 能源 (3只)
    'XOM',   # 埃克森美孚 - 石油巨头
    'CVX',   # 雪佛龙 - 石油巨头
    'SLB',   # 斯伦贝谢 - 油服龙头
    # 其他 (2只)
    'DIS',   # 迪士尼 - 娱乐帝国
    'NEE',   # NextEra - 清洁能源龙头
]

# 道琼斯30成分股（排除与 NASDAQ100 重复的）
# 重复的: AAPL, MSFT, AMZN, NVDA, INTC, CSCO, HON
DOW30_EXCLUSIVE = [
    # 金融
    'JPM',   # 摩根大通
    'GS',    # 高盛
    'AXP',   # 美国运通
    'TRV',   # 旅行者保险
    # 医疗健康
    'UNH',   # 联合健康
    'JNJ',   # 强生
    'MRK',   # 默沙东
    'AMGN',  # 安进 (也在 NASDAQ100，但保留)
    # 消费
    'WMT',   # 沃尔玛
    'HD',    # 家得宝
    'PG',    # 宝洁
    'MCD',   # 麦当劳
    'NKE',   # 耐克
    'KO',    # 可口可乐
    'DIS',   # 迪士尼
    # 工业
    'CAT',   # 卡特彼勒
    'BA',    # 波音
    'MMM',   # 3M
    'DOW',   # 陶氏化学
    # 能源
    'CVX',   # 雪佛龙
    # 通信
    'VZ',    # 威瑞森
    # 其他
    'IBM',   # IBM
    'CRM',   # Salesforce
    'WBA',   # 沃尔格林
    'SHW',   # 宣伟涂料
]

# S&P500 非科技龙头（排除 NASDAQ100 和 DOW30 重复）
SP500_NON_TECH = [
    # 金融
    'BAC',   # 美国银行
    'WFC',   # 富国银行
    'C',     # 花旗
    'MS',    # 摩根士丹利
    'BLK',   # 贝莱德
    'SCHW',  # 嘉信理财
    'BRK-B', # 伯克希尔
    # 医疗
    'LLY',   # 礼来
    'ABBV',  # 艾伯维
    'PFE',   # 辉瑞
    'TMO',   # 赛默飞
    'ABT',   # 雅培
    'DHR',   # 丹纳赫
    'BMY',   # 百时美施贵宝
    # 消费
    'LOW',   # 劳氏
    'TGT',   # 塔吉特
    'SBUX',  # 星巴克 (也在 NASDAQ100)
    'TJX',   # TJX
    'CMG',   # Chipotle
    'YUM',   # 百胜餐饮
    # 工业
    'UNP',   # 联合太平洋
    'RTX',   # 雷神
    'LMT',   # 洛克希德马丁
    'GE',    # 通用电气
    'DE',    # 迪尔
    'FDX',   # 联邦快递
    'UPS',   # UPS
    # 能源
    'XOM',   # 埃克森美孚
    'COP',   # 康菲石油
    'SLB',   # 斯伦贝谢
    'EOG',   # EOG资源
    # 公用事业
    'NEE',   # NextEra
    'DUK',   # 杜克能源
    'SO',    # 南方公司
    # 材料
    'LIN',   # 林德 (也在 NASDAQ100)
    'APD',   # 空气化工
    'FCX',   # 自由港麦克莫兰
    # 房地产
    'PLD',   # 普洛斯
    'AMT',   # 美国电塔
    'EQIX',  # Equinix
]

# 合并股票池（去重）
ALL_STOCKS = list(dict.fromkeys(NASDAQ100 + BLUECHIP_NON_TECH))

# 样本外验证池：道琼斯 + S&P500非科技（排除 NASDAQ100 重复）
_nasdaq_set = set(NASDAQ100)
OUT_OF_SAMPLE_POOL = [s for s in (DOW30_EXCLUSIVE + SP500_NON_TECH) if s not in _nasdaq_set]
OUT_OF_SAMPLE_POOL = list(dict.fromkeys(OUT_OF_SAMPLE_POOL))  # 去重


def get_stock_pool(pool_type: str = 'all') -> list:
    """
    获取股票池
    
    Args:
        pool_type: 'nasdaq100' | 'bluechip' | 'dow30' | 'sp500_non_tech' | 'out_of_sample' | 'all'
    
    Returns:
        股票代码列表
    """
    if pool_type == 'nasdaq100':
        return NASDAQ100.copy()
    elif pool_type == 'bluechip':
        return BLUECHIP_NON_TECH.copy()
    elif pool_type == 'dow30':
        return DOW30_EXCLUSIVE.copy()
    elif pool_type == 'sp500_non_tech':
        return SP500_NON_TECH.copy()
    elif pool_type == 'out_of_sample':
        return OUT_OF_SAMPLE_POOL.copy()
    else:
        return ALL_STOCKS.copy()


def get_pool_info() -> dict:
    """获取股票池信息"""
    return {
        'nasdaq100_count': len(NASDAQ100),
        'bluechip_count': len(BLUECHIP_NON_TECH),
        'dow30_count': len(DOW30_EXCLUSIVE),
        'sp500_non_tech_count': len(SP500_NON_TECH),
        'out_of_sample_count': len(OUT_OF_SAMPLE_POOL),
        'total_count': len(ALL_STOCKS),
    }
