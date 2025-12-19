#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
凯利公式下注工具类

凯利公式: f* = (bp - q) / b
其中:
    f* = 最优下注比例
    b  = 赔率 (盈利/本金)
    p  = 获胜概率
    q  = 失败概率 (1 - p)
"""


class KellyCriterion:
    """凯利公式计算器"""
    
    def __init__(self, win_rate: float, win_loss_ratio: float):
        """
        初始化凯利计算器
        
        参数:
            win_rate: 胜率 (0-1之间)
            win_loss_ratio: 盈亏比 (平均盈利 / 平均亏损)
        """
        self.win_rate = win_rate
        self.win_loss_ratio = win_loss_ratio
        self._validate()
    
    def _validate(self):
        """验证参数"""
        if not 0 < self.win_rate < 1:
            raise ValueError("胜率必须在0到1之间")
        if self.win_loss_ratio <= 0:
            raise ValueError("盈亏比必须大于0")
    
    @property
    def lose_rate(self) -> float:
        """失败概率"""
        return 1 - self.win_rate
    
    def kelly_fraction(self) -> float:
        """
        计算凯利最优下注比例
        
        公式: f* = (bp - q) / b
             = p - q/b
             = 胜率 - 败率/盈亏比
        
        返回:
            最优下注比例 (可能为负，表示不应下注)
        """
        b = self.win_loss_ratio
        p = self.win_rate
        q = self.lose_rate
        
        return (b * p - q) / b
    
    def half_kelly(self) -> float:
        """
        半凯利下注比例 (更保守，降低波动)
        """
        return self.kelly_fraction() / 2
    
    def quarter_kelly(self) -> float:
        """
        四分之一凯利下注比例 (非常保守)
        """
        return self.kelly_fraction() / 4
    
    def fractional_kelly(self, fraction: float) -> float:
        """
        分数凯利下注比例
        
        参数:
            fraction: 凯利比例的分数 (如0.5表示半凯利)
        """
        return self.kelly_fraction() * fraction
    
    def should_bet(self) -> bool:
        """
        是否应该下注 (凯利比例 > 0)
        """
        return self.kelly_fraction() > 0
    
    def expected_value(self) -> float:
        """
        期望值 (每单位下注的预期收益)
        
        公式: EV = p * b - q
        """
        return self.win_rate * self.win_loss_ratio - self.lose_rate
    
    def calculate_bet_size(self, bankroll: float, kelly_fraction: float = 1.0) -> float:
        """
        计算实际下注金额
        
        参数:
            bankroll: 总资金
            kelly_fraction: 凯利比例的分数 (默认1.0为全凯利)
        
        返回:
            建议下注金额
        """
        fraction = self.fractional_kelly(kelly_fraction)
        if fraction <= 0:
            return 0
        return bankroll * fraction
    
    def simulate_growth(self, bankroll: float, num_bets: int, kelly_fraction: float = 1.0) -> dict:
        """
        模拟资金增长 (基于期望值)
        
        参数:
            bankroll: 初始资金
            num_bets: 下注次数
            kelly_fraction: 凯利比例分数
        
        返回:
            模拟结果字典
        """
        import math
        
        f = self.fractional_kelly(kelly_fraction)
        if f <= 0:
            return {
                'initial': bankroll,
                'final': bankroll,
                'growth_rate': 0,
                'total_return': 0
            }
        
        # 期望对数增长率
        # g = p * log(1 + f*b) + q * log(1 - f)
        b = self.win_loss_ratio
        p = self.win_rate
        q = self.lose_rate
        
        log_growth = p * math.log(1 + f * b) + q * math.log(1 - f)
        
        # 预期最终资金
        expected_final = bankroll * math.exp(log_growth * num_bets)
        
        return {
            'initial': bankroll,
            'final': round(expected_final, 2),
            'growth_rate': round(log_growth * 100, 4),  # 每次下注的增长率%
            'total_return': round((expected_final / bankroll - 1) * 100, 2)  # 总回报率%
        }
    
    def summary(self) -> str:
        """生成摘要报告"""
        kelly = self.kelly_fraction()
        
        lines = [
            "=" * 50,
            "凯利公式分析报告",
            "=" * 50,
            f"胜率: {self.win_rate * 100:.2f}%",
            f"盈亏比: {self.win_loss_ratio:.2f}",
            f"期望值: {self.expected_value():.4f}",
            "-" * 50,
            f"全凯利比例: {kelly * 100:.2f}%",
            f"半凯利比例: {self.half_kelly() * 100:.2f}%",
            f"1/4凯利比例: {self.quarter_kelly() * 100:.2f}%",
            "-" * 50,
            f"建议下注: {'是' if self.should_bet() else '否'}",
            "=" * 50,
        ]
        return "\n".join(lines)
    
    def __repr__(self):
        return f"KellyCriterion(win_rate={self.win_rate}, win_loss_ratio={self.win_loss_ratio})"


class OptionsKelly(KellyCriterion):
    """
    期权交易专用凯利计算器
    
    针对卖出期权策略:
    - 盈利 = 收取的权利金
    - 亏损 = 止损金额 或 被行权损失
    """
    
    def __init__(self, win_rate: float, avg_premium: float, avg_loss: float):
        """
        初始化期权凯利计算器
        
        参数:
            win_rate: 胜率
            avg_premium: 平均收取权利金 (盈利时)
            avg_loss: 平均亏损金额 (亏损时)
        """
        self.avg_premium = avg_premium
        self.avg_loss = avg_loss
        win_loss_ratio = avg_premium / avg_loss if avg_loss > 0 else float('inf')
        super().__init__(win_rate, win_loss_ratio)
    
    def contracts_to_sell(self, bankroll: float, margin_per_contract: float, 
                          kelly_fraction: float = 0.5) -> int:
        """
        计算应卖出的期权合约数
        
        参数:
            bankroll: 总资金
            margin_per_contract: 每张合约所需保证金
            kelly_fraction: 凯利比例分数 (默认半凯利)
        
        返回:
            建议卖出的合约数
        """
        bet_size = self.calculate_bet_size(bankroll, kelly_fraction)
        if bet_size <= 0 or margin_per_contract <= 0:
            return 0
        return int(bet_size / margin_per_contract)
    
    def max_risk_per_trade(self, bankroll: float, kelly_fraction: float = 0.5) -> float:
        """
        每笔交易最大风险金额
        
        参数:
            bankroll: 总资金
            kelly_fraction: 凯利比例分数
        
        返回:
            最大风险金额
        """
        return self.calculate_bet_size(bankroll, kelly_fraction)


def kelly_from_trades(trades: list) -> KellyCriterion:
    """
    从历史交易记录计算凯利参数
    
    参数:
        trades: 交易记录列表，每个元素是盈亏金额 (正数盈利，负数亏损)
    
    返回:
        KellyCriterion 实例
    """
    if not trades:
        raise ValueError("交易记录不能为空")
    
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    
    win_rate = len(wins) / len(trades)
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 1
    
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    return KellyCriterion(win_rate, win_loss_ratio)


# ============================================================
# 示例用法
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("凯利公式工具类示例")
    print("=" * 60)
    
    # 示例1: 基本用法
    print("\n【示例1: 基本凯利计算】")
    kelly = KellyCriterion(win_rate=0.6, win_loss_ratio=1.5)
    print(kelly.summary())
    
    # 计算下注金额
    bankroll = 100000
    print(f"\n总资金: ${bankroll:,}")
    print(f"全凯利下注: ${kelly.calculate_bet_size(bankroll, 1.0):,.2f}")
    print(f"半凯利下注: ${kelly.calculate_bet_size(bankroll, 0.5):,.2f}")
    print(f"1/4凯利下注: ${kelly.calculate_bet_size(bankroll, 0.25):,.2f}")
    
    # 模拟增长
    print("\n【模拟100次下注后的资金增长】")
    for frac in [1.0, 0.5, 0.25]:
        result = kelly.simulate_growth(bankroll, 100, frac)
        print(f"  {frac}凯利: ${result['initial']:,} -> ${result['final']:,.0f} ({result['total_return']:.1f}%)")
    
    # 示例2: 期权交易
    print("\n" + "=" * 60)
    print("【示例2: 期权卖出策略凯利计算】")
    print("=" * 60)
    
    # 假设回测结果: 胜率70%, 平均盈利$200, 平均亏损$500
    opt_kelly = OptionsKelly(
        win_rate=0.70,
        avg_premium=200,  # 平均收取权利金
        avg_loss=500      # 平均亏损
    )
    print(opt_kelly.summary())
    
    print(f"\n总资金: ${bankroll:,}")
    print(f"每张合约保证金: $5,000")
    print(f"建议卖出合约数 (半凯利): {opt_kelly.contracts_to_sell(bankroll, 5000, 0.5)} 张")
    print(f"每笔最大风险 (半凯利): ${opt_kelly.max_risk_per_trade(bankroll, 0.5):,.2f}")
    
    # 示例3: 从交易记录计算
    print("\n" + "=" * 60)
    print("【示例3: 从历史交易记录计算凯利】")
    print("=" * 60)
    
    # 模拟交易记录 (正数盈利，负数亏损)
    trades = [150, 180, -400, 200, 160, -350, 190, 170, -500, 185,
              175, 195, -420, 165, 180, 200, -380, 190, 175, 160]
    
    kelly_from_history = kelly_from_trades(trades)
    print(f"交易次数: {len(trades)}")
    print(f"盈利次数: {len([t for t in trades if t > 0])}")
    print(f"亏损次数: {len([t for t in trades if t < 0])}")
    print(kelly_from_history.summary())
