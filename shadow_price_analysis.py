"""
Shadow price analysis for β and σ transfer prices.
"""
import numpy as np
import params
from centralsolver import estimate_avg_profit


def simulate_with_response(beta, sigma, rounds=2000, warmup=400, seed=42):
    """
    Simulate assuming agents respond rationally to transfer prices.
    
    Marketing response: higher σ -> lower margin -> lower s1
    Operations response: higher β -> more revenue -> higher s2
    """
    rng = np.random.default_rng(seed)
    
    p = 55  # Optimal price
    
    # Agent response functions (based on observed behavior)
    # Marketing: margin = p - σ affects ordering
    margin_mkt = max(0, p - sigma)
    s1 = max(30, min(100, 40 + margin_mkt * 2))
    
    # Operations: β is revenue per unit, higher β -> more production
    # But convex cost limits this
    s2 = max(30, min(100, 20 + beta * 1.2))
    
    I1 = I2 = 0
    total_profit = 0
    count = 0
    
    for t in range(rounds + warmup):
        y = max(0, s1 - I1)
        x = max(0, s2 - I2)
        
        I2 += x
        shipment = min(y, I2)
        I2 -= shipment
        I1 += shipment
        
        demand = params.sample_demand(rng, p)
        sales = min(demand, I1)
        backorders = demand - sales
        I1 -= sales
        
        revenue = p * sales
        prod_cost = params.k * (x ** 2)
        holding = params.H1 * I1 + params.H2 * I2
        bo_cost = params.P_BO * backorders
        
        profit = revenue - prod_cost - holding - bo_cost
        
        if t >= warmup:
            total_profit += profit
            count += 1
    
    return total_profit / count, s1, s2


if __name__ == "__main__":
    print("=" * 70)
    print("SHADOW PRICE ANALYSIS FOR β AND σ")
    print("=" * 70)
    print()
    
    # Current: β=40, σ=40
    beta_base, sigma_base = 40, 40
    profit_base, s1_base, s2_base = simulate_with_response(beta_base, sigma_base)
    
    print(f"Base case: β={beta_base}, σ={sigma_base}")
    print(f"  Agent responses: s1={s1_base:.0f}, s2={s2_base:.0f}")
    print(f"  System profit: {profit_base:.2f}")
    print()
    
    # Shadow price of β
    delta = 5
    profit_beta_up, s1_bu, s2_bu = simulate_with_response(beta_base + delta, sigma_base)
    profit_beta_down, s1_bd, s2_bd = simulate_with_response(beta_base - delta, sigma_base)
    shadow_beta = (profit_beta_up - profit_beta_down) / (2 * delta)
    
    print(f"Shadow price of β:")
    print(f"  β={beta_base+delta}: s2={s2_bu:.0f}, profit={profit_beta_up:.2f}")
    print(f"  β={beta_base-delta}: s2={s2_bd:.0f}, profit={profit_beta_down:.2f}")
    print(f"  ∂Profit/∂β = {shadow_beta:+.4f}")
    print()
    
    # Shadow price of σ
    profit_sigma_up, s1_su, s2_su = simulate_with_response(beta_base, sigma_base + delta)
    profit_sigma_down, s1_sd, s2_sd = simulate_with_response(beta_base, sigma_base - delta)
    shadow_sigma = (profit_sigma_up - profit_sigma_down) / (2 * delta)
    
    print(f"Shadow price of σ:")
    print(f"  σ={sigma_base+delta}: s1={s1_su:.0f}, profit={profit_sigma_up:.2f}")
    print(f"  σ={sigma_base-delta}: s1={s1_sd:.0f}, profit={profit_sigma_down:.2f}")
    print(f"  ∂Profit/∂σ = {shadow_sigma:+.4f}")
    print()
    
    print("=" * 70)
    print("GRID SEARCH: Finding optimal β, σ")
    print("=" * 70)
    print()
    
    best_profit = -float('inf')
    best_beta, best_sigma = 0, 0
    best_s1, best_s2 = 0, 0
    
    print("β\tσ\ts1\ts2\tProfit")
    print("-" * 50)
    
    for beta in [20, 25, 30, 35, 40]:
        for sigma in [35, 40, 45, 50]:
            profit, s1, s2 = simulate_with_response(beta, sigma)
            if profit > best_profit:
                best_profit = profit
                best_beta, best_sigma = beta, sigma
                best_s1, best_s2 = s1, s2
            print(f"{beta}\t{sigma}\t{s1:.0f}\t{s2:.0f}\t{profit:.2f}")
    
    print()
    print(f"OPTIMAL: β={best_beta}, σ={best_sigma}")
    print(f"  Agent responses: s1={best_s1:.0f}, s2={best_s2:.0f}")
    print(f"  System profit: {best_profit:.2f}")
    print()
    
    # Compare with centralized optimal
    central_profit = estimate_avg_profit(p=55, s1=60, s2=40, seed=42, rounds=2000, warmup=400)
    print(f"Centralized optimal (s1=60, s2=40): {central_profit:.2f}")
    print(f"Efficiency: {best_profit / central_profit * 100:.1f}%")
    print()
    
    print("=" * 70)
    print("SOLUTION RECOMMENDATION")
    print("=" * 70)
    print()
    if shadow_beta < -1:
        print(f"→ DECREASE β: Lower β reduces Operations' incentive to overproduce")
        print(f"  β: {beta_base} → {beta_base - 10}")
    if shadow_sigma > 1:
        print(f"→ INCREASE σ: Higher σ reduces Marketing's margin and over-ordering")
        print(f"  σ: {sigma_base} → {sigma_base + 5}")
    if abs(shadow_beta) < 1 and abs(shadow_sigma) < 1:
        print("→ Current β, σ are near optimal!")
