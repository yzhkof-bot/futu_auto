#!/usr/bin/env python3
"""
遗传规划 Alpha 因子挖掘 - 运行脚本

使用方法:
    python gp_alpha/run_mining.py                    # 默认参数快速挖掘
    python gp_alpha/run_mining.py --full             # 完整挖掘（更多代数）
    python gp_alpha/run_mining.py --symbols 30      # 指定股票数量
    python gp_alpha/run_mining.py --generations 50  # 指定进化代数
"""

import argparse
import sys
import os
from datetime import datetime
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gp_alpha.gp_alpha_miner import GPAlphaMiner


def main():
    parser = argparse.ArgumentParser(description='遗传规划 Alpha 因子挖掘')
    
    parser.add_argument('--symbols', type=int, default=20,
                        help='使用的股票数量 (默认: 20)')
    parser.add_argument('--population', type=int, default=500,
                        help='种群大小 (默认: 500)')
    parser.add_argument('--generations', type=int, default=20,
                        help='进化代数 (默认: 20)')
    parser.add_argument('--top', type=int, default=10,
                        help='返回最佳因子数量 (默认: 10)')
    parser.add_argument('--forward', type=int, default=1,
                        help='预测未来收益天数 (默认: 1)')
    parser.add_argument('--full', action='store_true',
                        help='完整挖掘模式 (种群1000, 代数50)')
    parser.add_argument('--quick', action='store_true',
                        help='快速测试模式 (种群200, 代数10)')
    parser.add_argument('--save', type=str, default=None,
                        help='保存模型路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出结果 JSON 路径')
    
    args = parser.parse_args()
    
    # 模式覆盖
    if args.full:
        args.population = 1000
        args.generations = 50
        args.symbols = 50
        args.top = 20
    elif args.quick:
        args.population = 200
        args.generations = 10
        args.symbols = 10
        args.top = 5
    
    print("\n" + "=" * 70)
    print("🧬 遗传规划 Alpha 因子挖掘")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"参数:")
    print(f"  - 股票数量: {args.symbols}")
    print(f"  - 种群大小: {args.population}")
    print(f"  - 进化代数: {args.generations}")
    print(f"  - 预测天数: {args.forward}")
    print(f"  - 返回因子: {args.top}")
    print("=" * 70)
    
    # 创建挖掘器
    miner = GPAlphaMiner(
        population_size=args.population,
        generations=args.generations,
        verbose=1
    )
    
    # 执行挖掘
    factors = miner.mine(
        n_symbols=args.symbols,
        forward_days=args.forward,
        top_n=args.top
    )
    
    # 保存模型
    if args.save:
        miner.save(args.save)
    
    # 输出结果
    if args.output:
        # 转换为可序列化格式
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'params': {
                'symbols': args.symbols,
                'population': args.population,
                'generations': args.generations,
                'forward_days': args.forward
            },
            'factors': []
        }
        
        for f in factors:
            output_data['factors'].append({
                'rank': f['rank'],
                'formula': f['formula'],
                'ic': float(f['ic']),
                'icir': float(f['icir']),
                'sharpe': float(f['sharpe']),
                'turnover': float(f['turnover']),
                'score': float(f['score']),
                'complexity': {
                    'length': f['length'],
                    'depth': f['depth']
                }
            })
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存至 {args.output}")
    
    # 打印最终汇总
    print("\n" + "=" * 70)
    print("🏆 最佳因子汇总")
    print("=" * 70)
    
    for f in factors[:5]:  # 只显示前5个
        print(f"\n[#{f['rank']}] IC={f['ic']:.4f} | ICIR={f['icir']:.4f} | Score={f['score']:.4f}")
        print(f"    公式: {f['formula']}")
    
    print("\n" + "=" * 70)
    print("挖掘完成！")
    print("=" * 70)
    
    return factors


if __name__ == '__main__':
    main()
