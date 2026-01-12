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

# 合并股票池（去重）
ALL_STOCKS = list(dict.fromkeys(NASDAQ100 + BLUECHIP_NON_TECH))


def get_stock_pool(pool_type: str = 'all') -> list:
    """
    获取股票池
    
    Args:
        pool_type: 'nasdaq100' | 'bluechip' | 'all'
    
    Returns:
        股票代码列表
    """
    if pool_type == 'nasdaq100':
        return NASDAQ100.copy()
    elif pool_type == 'bluechip':
        return BLUECHIP_NON_TECH.copy()
    else:
        return ALL_STOCKS.copy()


def get_pool_info() -> dict:
    """获取股票池信息"""
    return {
        'nasdaq100_count': len(NASDAQ100),
        'bluechip_count': len(BLUECHIP_NON_TECH),
        'total_count': len(ALL_STOCKS),
    }
