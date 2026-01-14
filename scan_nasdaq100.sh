#!/bin/bash
cd /Users/windye/PycharmProjects/FUTU_auto
source .venv/bin/activate

echo "========== 抄底信号扫描 =========="
python scan_nasdaq100.py

echo ""
echo "========== 追涨信号扫描 (Top5因子) =========="
python scan_nasdaq100_top5.py

#抄底回测脚本
#cd /Users/windye/PycharmProjects/FUTU_auto && source .venv/bin/activate && python backtest_random_nasdaq.py 2>&1 | cat
#追涨回测脚本
#cd /Users/windye/PycharmProjects/FUTU_auto && source .venv/bin/activate && python backtest_top5_strategy.py 2>&1 | cat
#卖点回测脚本
#cd /Users/windye/PycharmProjects/FUTU_auto && source .venv/bin/activate && python backtest_ultra_elite_sell_strategy.py AAPL 15 7



mul(neg(sub(delay5(min5(volume)), mean10(std10(return_5)))), mean5(neg(std10(min5(close)))))