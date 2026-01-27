"""
Analyze why efficiency is 97% despite s1/s2 deviations.
Also compute shadow prices for constraints.
"""

import numpy as np
import params
from centralsolver import estimate_avg_profit


def detailed_simulation(p, s1, s2, rounds=5000, warmup=500, seed=42):
    """Run detailed simulation and return all metrics."""
    rng = np.random.default_rng(seed)
    
    I1 = I2 = 0
    
    total_profit = 0
    total_revenue = 0
    total_prod_cost = 0
    total_holding_I1 = 0
    total_holding_I2 = 0
    total_backorder_cost = 0
    total_sales = 0
    total_demand = 0
    total_backorders = 0
    total_I1 = 0
    total_I2 = 0
    count = 0
    
    for t in range(rounds + warmup):
        # Order and production from base-stock
        y = max(0, s1 - I1)
        x = max(0, s2 - I2)
        
        # Production
        I2 += x
        
        # Shipping
        shipment = min(y, I2)
        I2 -= shipment
        I1 += shipment
        
        # Demand
        demand = params.sample_demand(rng, p)
        
        # Sales and backorders
        sales = min(demand, I1)
        backorders = demand - sales
        I1 -= sales
        
        # Costs
        revenue = p * sales
        prod_cost = params.k * (x ** 2)
        holding_I1 = params.H1 * I1
        holding_I2 = params.H2 * I2
        backorder_cost = params.P_BO * backorders
        
        profit = revenue - prod_cost - holding_I1 - holding_I2 - backorder_cost
        
        if t >= warmup:
            total_profit += profit
            total_revenue += revenue
            total_prod_cost += prod_cost
            total_holding_I1 += holding_I1
            total_holding_I2 += holding_I2
            total_backorder_cost += backorder_cost
            total_sales += sales
            total_demand += demand
            total_backorders += backorders
            total_I1 += I1
            total_I2 += I2
            count += 1
    
    return {
        'profit': total_profit / count,
        'revenue': total_revenue / count,
        'prod_cost': total_prod_cost / count,
        'holding_I1': total_holding_I1 / count,
        'holding_I2': total_holding_I2 / count,
        'backorder_cost': total_backorder_cost / count,
        'sales': total_sales / count,
        'demand': total_demand / count,
        'backorders': total_backorders / count,
        'I1': total_I1 / count,
        'I2': total_I2 / count,
    }


def compute_shadow_prices(p_opt, s1_opt, s2_opt, profit_opt):
    """
    Compute shadow prices by perturbing each decision variable.
    Shadow price = marginal value of relaxing a constraint by 1 unit.
    """
    delta = 5  # Perturbation size (matching grid step)
    
    shadow_prices = {}
    
    # Shadow price of s1 (inventory capacity at retailer)
    profit_s1_up = estimate_avg_profit(p=p_opt, s1=s1_opt + delta, s2=s2_opt, 
                                        seed=42, rounds=1000, warmup=200)
    profit_s1_down = estimate_avg_profit(p=p_opt, s1=s1_opt - delta, s2=s2_opt,
                                          seed=42, rounds=1000, warmup=200)
    shadow_prices['s1'] = (profit_s1_up - profit_s1_down) / (2 * delta)
    
    # Shadow price of s2 (inventory capacity at supplier)
    profit_s2_up = estimate_avg_profit(p=p_opt, s1=s1_opt, s2=s2_opt + delta,
                                        seed=42, rounds=1000, warmup=200)
    profit_s2_down = estimate_avg_profit(p=p_opt, s1=s1_opt, s2=s2_opt - delta,
                                          seed=42, rounds=1000, warmup=200)
    shadow_prices['s2'] = (profit_s2_up - profit_s2_down) / (2 * delta)
    
    # Shadow price of p (price flexibility)
    profit_p_up = estimate_avg_profit(p=p_opt + 5, s1=s1_opt, s2=s2_opt,
                                       seed=42, rounds=1000, warmup=200)
    profit_p_down = estimate_avg_profit(p=p_opt - 5, s1=s1_opt, s2=s2_opt,
                                         seed=42, rounds=1000, warmup=200)
    shadow_prices['p'] = (profit_p_up - profit_p_down) / 10
    
    return shadow_prices


if __name__ == "__main__":
    print("=" * 70)
    print("EFFICIENCY ANALYSIS: Why 97% with s1/s2 deviations?")
    print("=" * 70)
    print()
    
    # Optimal policy
    p_opt, s1_opt, s2_opt = 55, 60, 40
    
    # Decentralized policy (from UCB results)
    p_dec, s1_dec, s2_dec = 55, 50, 60
    
    print("DETAILED COST BREAKDOWN")
    print("-" * 70)
    
    print("\n1. OPTIMAL POLICY (p=55, s1=60, s2=40):")
    opt_metrics = detailed_simulation(p_opt, s1_opt, s2_opt)
    print(f"   Revenue:           {opt_metrics['revenue']:8.2f}")
    print(f"   - Production cost: {opt_metrics['prod_cost']:8.2f}")
    print(f"   - Holding I1:      {opt_metrics['holding_I1']:8.2f}  (I1 avg = {opt_metrics['I1']:.2f})")
    print(f"   - Holding I2:      {opt_metrics['holding_I2']:8.2f}  (I2 avg = {opt_metrics['I2']:.2f})")
    print(f"   - Backorder cost:  {opt_metrics['backorder_cost']:8.2f}  (BO avg = {opt_metrics['backorders']:.2f})")
    print(f"   = PROFIT:          {opt_metrics['profit']:8.2f}")
    print(f"   Sales: {opt_metrics['sales']:.2f}, Demand: {opt_metrics['demand']:.2f}")
    
    print("\n2. DECENTRALIZED POLICY (p=55, s1=50, s2=60):")
    dec_metrics = detailed_simulation(p_dec, s1_dec, s2_dec)
    print(f"   Revenue:           {dec_metrics['revenue']:8.2f}")
    print(f"   - Production cost: {dec_metrics['prod_cost']:8.2f}")
    print(f"   - Holding I1:      {dec_metrics['holding_I1']:8.2f}  (I1 avg = {dec_metrics['I1']:.2f})")
    print(f"   - Holding I2:      {dec_metrics['holding_I2']:8.2f}  (I2 avg = {dec_metrics['I2']:.2f})")
    print(f"   - Backorder cost:  {dec_metrics['backorder_cost']:8.2f}  (BO avg = {dec_metrics['backorders']:.2f})")
    print(f"   = PROFIT:          {dec_metrics['profit']:8.2f}")
    print(f"   Sales: {dec_metrics['sales']:.2f}, Demand: {dec_metrics['demand']:.2f}")
    
    print("\n3. DIFFERENCE ANALYSIS:")
    print(f"   Revenue diff:      {dec_metrics['revenue'] - opt_metrics['revenue']:+8.2f}")
    print(f"   Prod cost diff:    {dec_metrics['prod_cost'] - opt_metrics['prod_cost']:+8.2f}")
    print(f"   Holding I1 diff:   {dec_metrics['holding_I1'] - opt_metrics['holding_I1']:+8.2f}")
    print(f"   Holding I2 diff:   {dec_metrics['holding_I2'] - opt_metrics['holding_I2']:+8.2f}")
    print(f"   Backorder diff:    {dec_metrics['backorder_cost'] - opt_metrics['backorder_cost']:+8.2f}")
    print(f"   -" * 35)
    print(f"   PROFIT LOSS:       {dec_metrics['profit'] - opt_metrics['profit']:+8.2f}")
    
    efficiency = dec_metrics['profit'] / opt_metrics['profit'] * 100
    print(f"\n   EFFICIENCY: {efficiency:.1f}%")
    
    print("\n" + "=" * 70)
    print("SHADOW PRICE ANALYSIS")
    print("=" * 70)
    print("\nShadow price = marginal value of increasing decision variable by 1 unit")
    print("(at the optimal point)")
    print()
    
    shadow_prices = compute_shadow_prices(p_opt, s1_opt, s2_opt, opt_metrics['profit'])
    
    print(f"Shadow price of s1 (retailer base-stock):  {shadow_prices['s1']:+.4f} per unit")
    print(f"Shadow price of s2 (supplier base-stock):  {shadow_prices['s2']:+.4f} per unit")
    print(f"Shadow price of p (market price):          {shadow_prices['p']:+.4f} per unit")
    
    print("\nINTERPRETATION:")
    if abs(shadow_prices['s1']) < 0.5:
        print(f"  s1: Near optimal (shadow price ≈ 0), small changes don't matter much")
    else:
        print(f"  s1: {'Increase' if shadow_prices['s1'] > 0 else 'Decrease'} would {'help' if shadow_prices['s1'] > 0 else 'hurt'}")
    
    if abs(shadow_prices['s2']) < 0.5:
        print(f"  s2: Near optimal (shadow price ≈ 0), small changes don't matter much")
    else:
        print(f"  s2: {'Increase' if shadow_prices['s2'] > 0 else 'Decrease'} would {'help' if shadow_prices['s2'] > 0 else 'hurt'}")
    
    if abs(shadow_prices['p']) < 1:
        print(f"  p:  Near optimal (shadow price ≈ 0)")
    else:
        print(f"  p:  {'Increase' if shadow_prices['p'] > 0 else 'Decrease'} would {'help' if shadow_prices['p'] > 0 else 'hurt'}")
    
    print("\n" + "=" * 70)
    print("WHY IS EFFICIENCY 97% DESPITE s1/s2 DEVIATIONS?")
    print("=" * 70)
    print("""
KEY INSIGHT: The objective function is FLAT near the optimum!

1. PRICE (p) is the MOST IMPORTANT decision:
   - p affects demand directly: D = 120 - 1.5p
   - Decentralized found p* = 55 ✓ (optimal!)
   - This alone captures most of the value

2. INVENTORY LEVELS have DIMINISHING RETURNS:
   - s1 = 50 vs 60: only 10 units difference
   - The holding cost tradeoff is gentle near optimum
   - Extra safety stock has low marginal value

3. COSTS OFFSET EACH OTHER:
   - Decentralized has HIGHER I2 (more holding cost)
   - But also LOWER backorders (less penalty)
   - These partially cancel out!

4. CONVEX PRODUCTION COST LIMITS DAMAGE:
   - k × x² means production cost grows quadratically
   - This prevents extreme overproduction
""")
