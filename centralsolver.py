import numpy as np
import params

def _simulate_period(rng, I1, I2, p, s1, s2):
    # Order quantity from base-stock policy
    y = max(0, s1 - I1)
    
    # Production from base-stock policy: x = max(0, s2 - I2)
    x = max(0, s2 - I2)
    
    # Production: I2 increases by x
    I2 += x
    
    # Shipping: Q_ship = min(y, I2)
    shipment = min(y, I2)
    I2 -= shipment
    I1 += shipment
    
    # Demand realization (price-dependent)
    demand = params.sample_demand(rng, p)
    
    # Sales and backorders
    sales = min(demand, I1)
    backorders = demand - sales
    
    # Update inventory
    I1 -= sales
    
    # Calculate profit (negative cost)
    revenue = p * sales
    production_cost = params.k * (x ** 2)  # Convex cost
    # Marketing: (H1-H2) × I1 (incremental cost)
    # Operations: H2 × (I1+I2) (responsible for all downstream)
    # Total = (H1-H2)×I1 + H2×(I1+I2) = H1×I1 + H2×I2 (mathematically equivalent)
    holding_costs = (params.H1 - params.H2) * I1 + params.H2 *  (I2+I1) # with Echelon Inventory
    #holding_costs = (params.H1 - params.H2) * I1 + params.H2 *  (I2) # without Echelon Inventory
    backorder_cost = params.P_BO * backorders
    
    period_profit = revenue - production_cost - holding_costs - backorder_cost
    
    return I1, I2, period_profit, sales, backorders


def estimate_avg_profit(*, p, s1, s2, seed, rounds, warmup):
    rng = np.random.default_rng(seed)
    
    # Initial state
    I1 = I2 = 0
    
    total_profit = 0.0
    count = 0
    
    for t in range(rounds + warmup):
        I1, I2, profit, _, _ = _simulate_period(rng, I1, I2, p, s1, s2)
        
        if t >= warmup:
            total_profit += profit
            count += 1
    
    return total_profit / max(1, count)


def estimate_avg_cost(*, p, s1, s2, seed, rounds, warmup):
    return -estimate_avg_profit(p=p, s1=s1, s2=s2, seed=seed, rounds=rounds, warmup=warmup)


def compute_centralized_optimum(
    *,
    p_range=None,
    s1_range=None,
    s2_range=None,
    seed=None,
    rounds=None,
    warmup=None,
    verbose=True
):
    # Defaults from params
    if p_range is None:
        p_range = params.p_range
    if s1_range is None:
        s1_range = params.s1_range
    if s2_range is None:
        s2_range = params.s2_range
    if seed is None:
        seed = params.SEED
    if rounds is None:
        rounds = params.ROUNDS
    if warmup is None:
        warmup = params.WARMUP
    
    best_p = p_range[0]
    best_s1 = s1_range[0]
    best_s2 = s2_range[0]
    best_profit = float("-inf")
    
    total_combinations = len(p_range) * len(s1_range) * len(s2_range)
    
    if verbose:
        print(f"Computing centralized optimum over {total_combinations} combinations...")
    
    evaluated = 0
    for p in p_range:
        for s1 in s1_range:
            for s2 in s2_range:
                profit = estimate_avg_profit(
                    p=p, s1=s1, s2=s2,
                    seed=seed, rounds=rounds, warmup=warmup
                )
                
                if profit > best_profit:
                    best_profit = profit
                    best_p = p
                    best_s1 = s1
                    best_s2 = s2
                
                evaluated += 1
                if verbose and evaluated % 100 == 0:
                    print(f"  Evaluated {evaluated}/{total_combinations}...")
    
    best_cost = -best_profit
    
    if verbose:
        print(f"\n{'='*50}")
        print("CENTRALIZED OPTIMUM FOUND:")
        print(f"  Price (p*):        {best_p}")
        print(f"  Base-stock s1*:    {best_s1}")
        print(f"  Base-stock s2*:    {best_s2}")
        print(f"  (x derived from s2 dynamically)")
        print(f"  Avg Profit/period: {best_profit:.2f}")
        print(f"  Avg Cost/period:   {best_cost:.2f}")
        print(f"{'='*50}")
    
    return int(best_p), int(best_s1), int(best_s2), float(best_profit), float(best_cost)

if __name__ == "__main__":
    print("Computing Centralized System Optimum")
    print("=" * 50)
    print(f"Parameters: a={params.a}, b={params.b}, σ_d={params.sigma_d}")
    print(f"Costs: h1={params.H1}, h2={params.H2}, k={params.k}, π={params.P_BO}")
    print(f"Backorder allocation: α={params.ALPHA}")
    print()
    
    p_opt, s1_opt, s2_opt, profit_opt, cost_opt = compute_centralized_optimum(verbose=True)
    
    print(f"\nTo use this benchmark, set in params.py:")
    print(f"  CTOT_OPT = {cost_opt:.2f}")