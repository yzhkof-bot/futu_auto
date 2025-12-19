#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Futu OpenAPI Demo - 简易程序示例
功能：
1. 获取行情快照
2. 模拟交易下单
"""

from futu import *


def get_market_snapshot_demo():
    """
    获取港股腾讯控股(HK.00700)的快照数据
    """
    print("=" * 50)
    print("1. 获取行情快照示例")
    print("=" * 50)
    
    # 创建行情连接对象
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    
    try:
        # 获取港股 HK.00700 的快照数据
        ret_code, ret_data = quote_ctx.get_market_snapshot('HK.00700')
        
        if ret_code == RET_OK:
            print("获取快照数据成功：")
            print(ret_data)
            print("\n关键数据：")
            if not ret_data.empty:
                row = ret_data.iloc[0]
                print(f"  股票代码: {row['code']}")
                print(f"  最新价: {row.get('last_price', 'N/A')}")
                print(f"  开盘价: {row.get('open_price', 'N/A')}")
                print(f"  最高价: {row.get('high_price', 'N/A')}")
                print(f"  最低价: {row.get('low_price', 'N/A')}")
        else:
            print(f"获取快照数据失败: {ret_data}")
            
    except Exception as e:
        print(f"发生异常: {e}")
    finally:
        # 关闭连接，避免占用资源
        quote_ctx.close()
        print("\n行情连接已关闭")


def place_order_demo():
    """
    模拟交易下单示例：买入100股腾讯控股
    """
    print("\n" + "=" * 50)
    print("2. 模拟交易下单示例")
    print("=" * 50)
    
    # 创建交易连接对象
    trd_ctx = OpenSecTradeContext(host='127.0.0.1', port=11111)
    
    try:
        # 模拟交易下单：买入100股，价格500元
        ret_code, ret_data = trd_ctx.place_order(
            price=500.0,
            qty=100,
            code="HK.00700",
            trd_side=TrdSide.BUY,
            trd_env=TrdEnv.SIMULATE  # 模拟交易环境
        )
        
        if ret_code == RET_OK:
            print("下单成功：")
            print(ret_data)
            if not ret_data.empty:
                row = ret_data.iloc[0]
                print(f"\n订单详情：")
                print(f"  订单ID: {row.get('order_id', 'N/A')}")
                print(f"  订单状态: {row.get('order_status', 'N/A')}")
                print(f"  股票代码: {row.get('code', 'N/A')}")
                print(f"  数量: {row.get('qty', 'N/A')}")
                print(f"  价格: {row.get('price', 'N/A')}")
        else:
            print(f"下单失败: {ret_data}")
            
    except Exception as e:
        print(f"发生异常: {e}")
    finally:
        # 关闭连接
        trd_ctx.close()
        print("\n交易连接已关闭")


def main():
    """
    主函数
    """
    print("\n" + "=" * 50)
    print("Futu OpenAPI Demo 启动")
    print("=" * 50)
    print("\n注意事项：")
    print("1. 确保已安装 futu-api: pip install futu-api")
    print("2. 确保 OpenD 已启动并登录（默认端口：11111）")
    print("3. 本示例使用模拟交易环境\n")
    
    try:
        # 1. 获取行情快照
        get_market_snapshot_demo()
        
        # 2. 模拟交易下单
        place_order_demo()
        
    except Exception as e:
        print(f"\n程序执行出错: {e}")
    
    print("\n" + "=" * 50)
    print("Demo 运行完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
