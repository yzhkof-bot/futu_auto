#!/bin/bash
cd /Users/windye/PycharmProjects/FUTU_auto
source .venv/bin/activate

# 创建日志目录
LOG_DIR="reports/scan_logs"
mkdir -p "$LOG_DIR"

# 按日期生成日志文件名
DATE=$(date +"%Y-%m-%d")
LOG_FILE="$LOG_DIR/scan_${DATE}.log"

# 执行统一扫描（只下载一次数据）
{
    python -u scan_combined.py
} 2>&1 | stdbuf -oL tee -a "$LOG_FILE"

echo ""
echo "结果已保存至: $LOG_FILE"

#抄底回测脚本
#cd /Users/windye/PycharmProjects/FUTU_auto && source .venv/bin/activate && python backtest_random_nasdaq.py 2>&1 | cat
#追涨回测脚本
#cd /Users/windye/PycharmProjects/FUTU_auto && source .venv/bin/activate && python backtest_top5_strategy.py 2>&1 | cat
#卖点回测脚本
#cd /Users/windye/PycharmProjects/FUTU_auto && source .venv/bin/activate && python backtest_ultra_elite_sell_strategy.py AAPL 15 7


#mul(neg(sub(delay5(min5(volume)), mean10(std10(return_5)))), mean5(neg(std10(min5(close)))))
