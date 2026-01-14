#!/usr/bin/env python3
"""
遗传规划 Alpha 因子挖掘 V2 - 运行脚本

工业级实现，支持:
- 截面 IC 评估
- 训练/测试集切分
- 完整因子报告

使用方法:
    python gp_alpha/run_mining_v2.py                    # 默认参数
    python gp_alpha/run_mining_v2.py --quick            # 快速测试
    python gp_alpha/run_mining_v2.py --full             # 完整挖掘
    python gp_alpha/run_mining_v2.py --pool nasdaq100   # 指定股票池
    python gp_alpha/run_mining_v2.py --forward 5        # 预测5日收益
"""

import argparse
import sys
import os
from datetime import datetime
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gp_alpha.gp_miner_v2 import GPAlphaMinerV2
from gp_alpha.evaluator import FactorEvaluator
from gp_alpha.visualize_v2 import plot_factor_comparison, plot_full_report


def main():
    parser = argparse.ArgumentParser(description='遗传规划 Alpha 因子挖掘 V2')
    
    # 数据参数
    parser.add_argument('--pool', type=str, default='nasdaq100',
                        choices=['nasdaq100', 'bluechip', 'all'],
                        help='股票池类型 (默认: nasdaq100)')
    parser.add_argument('--start', type=str, default=None,
                        help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                        help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='训练集比例 (默认: 0.7)')
    
    # GP 参数（工业级默认值）
    parser.add_argument('--population', type=int, default=2000,
                        help='种群大小 (默认: 2000)')
    parser.add_argument('--generations', type=int, default=50,
                        help='进化代数 (默认: 50)')
    parser.add_argument('--forward', type=int, default=5,
                        help='预测未来收益天数 (默认: 5)')
    parser.add_argument('--top', type=int, default=20,
                        help='返回最佳因子数量 (默认: 20)')
    
    # 模式
    parser.add_argument('--quick', action='store_true',
                        help='快速测试模式 (种群500, 代数15)')
    parser.add_argument('--full', action='store_true',
                        help='完整挖掘模式 (种群5000, 代数100)')
    
    # 输出
    parser.add_argument('--save', type=str, default=None,
                        help='保存模型路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出结果 JSON 路径')
    parser.add_argument('--plot', action='store_true',
                        help='绘制可视化图表')
    parser.add_argument('--no-cache', action='store_true',
                        help='不使用数据缓存')
    
    args = parser.parse_args()
    
    # 模式覆盖
    if args.quick:
        args.population = 500
        args.generations = 15
        args.top = 10
    elif args.full:
        args.population = 5000
        args.generations = 100
        args.pool = 'all'
        args.top = 50
    
    print("\n" + "=" * 70)
    print("🧬 遗传规划 Alpha 因子挖掘 V2 (工业级)")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n【数据参数】")
    print(f"  股票池:     {args.pool}")
    print(f"  训练集比例: {args.train_ratio:.0%}")
    print(f"\n【GP 参数】")
    print(f"  种群大小:   {args.population}")
    print(f"  进化代数:   {args.generations}")
    print(f"  预测天数:   {args.forward}")
    print(f"  返回因子:   {args.top}")
    print("=" * 70)
    
    # 创建挖掘器
    miner = GPAlphaMinerV2(
        population_size=args.population,
        generations=args.generations,
        verbose=1
    )
    
    # 加载数据
    miner.load_data(
        pool_type=args.pool,
        start_date=args.start,
        end_date=args.end,
        train_ratio=args.train_ratio,
        use_cache=not args.no_cache
    )
    
    # 执行挖掘
    factors = miner.mine(
        forward_days=args.forward,
        top_n=args.top
    )
    
    # 打印摘要
    miner.print_summary(top_n=5)
    
    # 保存模型
    if args.save:
        miner.save(args.save)
    
    # 输出 JSON
    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'params': {
                'pool': args.pool,
                'population': args.population,
                'generations': args.generations,
                'forward_days': args.forward,
                'train_ratio': args.train_ratio,
            },
            'factors': []
        }
        
        for f in factors:
            factor_data = {
                'rank': f['rank'],
                'formula': f['formula'],
                'complexity': {
                    'length': f['length'],
                    'depth': f['depth']
                },
                'train': {
                    'ic': float(f['train_metrics'].get('ic_mean', 0) or 0),
                    'icir': float(f['train_metrics'].get('ic_ir', 0) or 0),
                    'sharpe': float(f['train_metrics'].get('long_short_sharpe', 0) or 0),
                    'score': float(f['train_metrics'].get('composite_score', 0) or 0),
                },
                'test': {
                    'ic': float(f['test_metrics'].get('ic_mean', 0) or 0),
                    'icir': float(f['test_metrics'].get('ic_ir', 0) or 0),
                    'sharpe': float(f['test_metrics'].get('long_short_sharpe', 0) or 0),
                    'score': float(f['test_metrics'].get('composite_score', 0) or 0),
                }
            }
            output_data['factors'].append(factor_data)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存至 {args.output}")
    
    # 可视化
    if args.plot and factors:
        print("\n生成可视化图表...")
        
        # 因子对比图
        plot_factor_comparison(factors[:5], 
                               save_path='reports/gp_factor_comparison.png')
        
        # 最佳因子详细报告
        if factors:
            best_factor = factors[0]
            factor_panel = miner.get_factor_panel(0)
            forward_return = miner.data_manager.get_forward_return(args.forward)
            
            evaluator = FactorEvaluator(factor_panel, forward_return, args.forward)
            plot_full_report(evaluator, 
                             factor_name=f"#{best_factor['rank']}: {best_factor['formula'][:40]}...",
                             save_path='reports/gp_best_factor_report.png')
    
    print("\n" + "=" * 70)
    print("✅ 挖掘完成！")
    print("=" * 70)
    
    return factors


if __name__ == '__main__':
    main()
