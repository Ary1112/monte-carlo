"""
Monte Carlo Trading Strategy Simulator
Clean implementation matching HowToTrade specs (inputs, metrics, and logic loop)
Runs with standard library (no external packages required, optional matplotlib plotting).
"""

import math
import random
from typing import Dict, Any, List

def run_monte_carlo(
    initial_balance: float = 3000.0,
    win_rate_pct: float = 54.0,
    risk_pct_per_trade: float = 1.0,
    reward_ratio: float = 1.5,
    trades_per_month: int = 20,
    total_months: int = 24,
    break_even_pct: float = 0.0,
    iterations: int = 1000,
    ruin_dd_threshold: float = 50.0  # drawdown % considered ruin
) -> Dict[str, Any]:
    """
    Simulates random trade sequence outcomes using compound position sizing.
    """
    p_win = win_rate_pct / 100.0
    p_be = break_even_pct / 100.0
    p_loss = max(0.0, 1.0 - p_win - p_be)
    risk_pct = risk_pct_per_trade / 100.0
    total_trades = trades_per_month * total_months

    # Theoretical Expected Value multiplier per trade
    trade_ev_multiplier = (p_win * (1.0 + risk_pct * reward_ratio)) + (p_be * 1.0) + (p_loss * (1.0 - risk_pct))
    theoretical_final_balance = initial_balance * (trade_ev_multiplier ** total_trades)

    final_balances: List[float] = []
    max_drawdowns: List[float] = []
    ruin_count = 0
    dd20_count = 0
    overall_max_loss_streak = 0
    overall_max_win_streak = 0

    monthly_trajectories: List[List[float]] = []

    for _ in range(iterations):
        balance = initial_balance
        peak = balance
        max_dd = 0.0
        cur_loss_streak = 0
        cur_win_streak = 0
        run_max_loss_streak = 0
        run_max_win_streak = 0

        monthly_bal = [balance]

        for m in range(1, total_months + 1):
            for _ in range(trades_per_month):
                if balance <= 1.0:
                    # Account wiped out
                    cur_loss_streak += 1
                    if cur_loss_streak > run_max_loss_streak:
                        run_max_loss_streak = cur_loss_streak
                    continue

                r = random.random()
                risk_dollar = balance * risk_pct

                if r < p_win:
                    # Win
                    balance += risk_dollar * reward_ratio
                    cur_win_streak += 1
                    cur_loss_streak = 0
                    if cur_win_streak > run_max_win_streak:
                        run_max_win_streak = cur_win_streak
                elif r < p_win + p_be:
                    # Break-even
                    pass
                else:
                    # Loss
                    balance -= risk_dollar
                    cur_loss_streak += 1
                    cur_win_streak = 0
                    if cur_loss_streak > run_max_loss_streak:
                        run_max_loss_streak = cur_loss_streak

                if balance > peak:
                    peak = balance
                dd = ((peak - balance) / peak) * 100.0 if peak > 0 else 100.0
                if dd > max_dd:
                    max_dd = dd

            monthly_bal.append(balance)

        if max_dd >= ruin_dd_threshold:
            ruin_count += 1
        if max_dd >= 20.0:
            dd20_count += 1

        if run_max_loss_streak > overall_max_loss_streak:
            overall_max_loss_streak = run_max_loss_streak
        if run_max_win_streak > overall_max_win_streak:
            overall_max_win_streak = run_max_win_streak

        final_balances.append(balance)
        max_drawdowns.append(max_dd)
        monthly_trajectories.append(monthly_bal)

    # Sort for percentiles
    final_balances.sort()
    max_drawdowns.sort()

    def get_percentile(data: List[float], p: float) -> float:
        idx = int(len(data) * (p / 100.0))
        idx = min(idx, len(data) - 1)
        return data[idx]

    p5 = get_percentile(final_balances, 5)
    p25 = get_percentile(final_balances, 25)
    p50 = get_percentile(final_balances, 50)
    p75 = get_percentile(final_balances, 75)
    p95 = get_percentile(final_balances, 95)
    median_dd = get_percentile(max_drawdowns, 50)
    worst_dd = max_drawdowns[-1]

    # Monthly breakdown for Expected Value
    monthly_ev = []
    curr_ev = initial_balance
    for m in range(1, total_months + 1):
        prev_ev = curr_ev
        curr_ev = prev_ev * (trade_ev_multiplier ** trades_per_month)
        monthly_ev.append({
            "month": m,
            "starting": prev_ev,
            "gain_dollar": curr_ev - prev_ev,
            "gain_pct": ((curr_ev - prev_ev) / prev_ev) * 100.0,
            "ending": curr_ev
        })

    return {
        "initial_balance": initial_balance,
        "theoretical_final_balance": theoretical_final_balance,
        "p5": p5,
        "p25": p25,
        "p50_median": p50,
        "p75": p75,
        "p95": p95,
        "median_max_drawdown": median_dd,
        "worst_max_drawdown": worst_dd,
        "risk_of_ruin_pct": (ruin_count / iterations) * 100.0,
        "prob_dd_20_pct": (dd20_count / iterations) * 100.0,
        "max_loss_streak": overall_max_loss_streak,
        "max_win_streak": overall_max_win_streak,
        "monthly_ev": monthly_ev,
        "total_trades": total_trades
    }

def print_report(res: Dict[str, Any]):
    init = res["initial_balance"]
    p50 = res["p50_median"]
    p95 = res["p95"]
    p5 = res["p5"]

    ev_final = res['theoretical_final_balance']
    ev_profit = ev_final - init
    ev_return_pct = (ev_profit / init) * 100

    print("=" * 65)
    print("      MONTE CARLO TRADING STRATEGY REPORT (1,000 RUNS)")
    print("=" * 65)
    print(f"Initial Starting Capital:        ${init:,.2f}")
    print(f"TOTAL NET PROFIT / LOSS (EV):    {'+' if ev_profit >= 0 else '-'}${abs(ev_profit):,.2f} ({'+' if ev_return_pct >= 0 else ''}{ev_return_pct:,.1f}% Return)")
    print(f"Expected Final Balance (EV):     ${ev_final:,.2f}")
    print(f"Total Trade Executions:          {res['total_trades']}")
    print("-" * 65)
    print("PERCENTILE OUTCOMES (Stress Tested Variance):")
    print(f"  * 95th Percentile (Best Case): ${p95:,.2f}  (Net: +${p95-init:,.2f} / +{((p95-init)/init)*100:,.1f}%)")
    print(f"  * 50th Percentile (Median):    ${p50:,.2f}  (Net: +${p50-init:,.2f} / +{((p50-init)/init)*100:,.1f}%)")
    print(f"  * 5th Percentile (Worst Case):  ${p5:,.2f}  (Net: {'+' if p5>=init else '-'}${abs(p5-init):,.2f} / {((p5-init)/init)*100:,.1f}%)")
    print("-" * 65)
    print("RISK & DRAWDOWN METRICS:")
    print(f"  * Median Maximum Drawdown:      {res['median_max_drawdown']:.1f}%")
    print(f"  * Worst Peak Drawdown (Run):   {res['worst_max_drawdown']:.1f}%")
    print(f"  * Odds of >20% Drawdown:       {res['prob_dd_20_pct']:.1f}%")
    print(f"  * Risk of Ruin (>50% DD):      {res['risk_of_ruin_pct']:.1f}%")
    print(f"  * Longest Losing Streak:       {res['max_loss_streak']} consecutive losses")
    print(f"  * Longest Winning Streak:      {res['max_win_streak']} consecutive wins")
    print("=" * 65)
    print("MONTHLY P&L MATRIX (Expected Trajectory - First 6 & Last 3 Months):")
    print(f"{'Month':<7} | {'Start Balance':<14} | {'Monthly Gain':<13} | {'End Balance':<14}")
    print("-" * 65)

    ev_list = res["monthly_ev"]
    display_indices = list(range(min(6, len(ev_list))))
    if len(ev_list) > 6:
        display_indices.extend(range(len(ev_list) - 3, len(ev_list)))

    seen = set()
    for idx in display_indices:
        if idx in seen:
            continue
        seen.add(idx)
        if idx == len(ev_list) - 3 and len(ev_list) > 9:
            print("  ...    |      ...       |     ...       |     ...")
        row = ev_list[idx]
        print(f"Month {row['month']:<2} | ${row['starting']:<13,.2f} | +${row['gain_dollar']:<11,.2f} | ${row['ending']:<13,.2f}")
    print("=" * 65)

if __name__ == "__main__":
    results = run_monte_carlo(
        initial_balance=3000.0,
        win_rate_pct=54.0,
        risk_pct_per_trade=1.0,
        reward_ratio=1.5,
        trades_per_month=20,
        total_months=24,
        break_even_pct=0.0,
        iterations=1000
    )
    print_report(results)
