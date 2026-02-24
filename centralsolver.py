"""
Centralized (system-optimal) benchmark solver for the 2-stage serial supply chain.

Computes the optimal combination of:
- Market price (p*)
- Base-stock levels (s1*, s2*)  
- Production level (x*)

Uses Monte Carlo simulation to estimate long-run average system profit.
The centralized planner can set all variables optimally without agency problems.
"""

import numpy as np
import params


def _simulate_period(rng, I1, I2, p, s1, s2):
    """
    Simulate one period under centralized control.
    
    Production is derived from base-stock policy: x = max(0, s2 - I2)
    
    Returns: (new_I1, new_I2, period_profit, sales, backorders)
    """
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
    # ECHELON inventory costs (same as agents.py for consistency)
    # Marketing: (H1-H2) × I1 (incremental cost)
    # Operations: H2 × (I1+I2) (responsible for all downstream)
    # Total = (H1-H2)×I1 + H2×(I1+I2) = H1×I1 + H2×I2 (mathematically equivalent)
    holding_costs = (params.H1 - params.H2) * I1 + params.H2 *  (I2 + I1)
    backorder_cost = params.P_BO * backorders
    
    period_profit = revenue - production_cost - holding_costs - backorder_cost
    
    return I1, I2, period_profit, sales, backorders


def estimate_avg_profit(*, p, s1, s2, seed, rounds, warmup):
    """
    Estimate long-run average profit under fixed policy (p, s1, s2) via Monte Carlo.
    
    Production x is derived from base-stock: x = max(0, s2 - I2)
    
    Args:
        p: Market price
        s1: Marketing base-stock level
        s2: Operations base-stock level
        seed: Random seed
        rounds: Number of simulation rounds
        warmup: Warmup period before measuring
    
    Returns:
        Average profit per period (after warmup)
    """
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
    """
    Estimate long-run average COST (for comparison, cost = -profit).
    """
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
    """
    Find the system-optimal policy by grid search over (p, s1, s2).
    
    Production x is derived from base-stock: x = max(0, s2 - I2)
    
    Returns:
        (p_opt, s1_opt, s2_opt, avg_profit_opt, avg_cost_opt)
    """
    # Defaults from params
    if p_range is None:
        p_range = params.p_range
    if s1_range is None:
        s1_range = params.s_range
    if s2_range is None:
        s2_range = params.s_range
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


def compute_quick_optimum(verbose=True):
    """
    Compute optimum with reduced search space for faster execution.
    Uses the SAME ranges as MARL simulation for fair comparison.
    x is derived from s2.
    """
    # Use same ranges as MARL simulation from params
    p_range = params.p_range  # Same price range as Marketing
    s1_range = params.s1_range  # Same s1 range as Marketing
    s2_range = params.s2_range  # Same s2 range as Operations
    
    return compute_centralized_optimum(
        p_range=p_range,
        s1_range=s1_range,
        s2_range=s2_range,
        rounds=1000,  # Faster
        warmup=200,
        verbose=verbose
    )


# ============================================
# Legacy Functions (for backwards compatibility)
# ============================================

# def _env_step_once(
#         rng, I1, I2, B1, B2,
#         s1_loc, s2_loc, price, h1, h2, p_bo, alpha
# ):
#     """
#     Simulates inventory transitions within one period.
#     Updated to use Normal distribution via params.sample_demand.
#     Lead time = 0, so orders arrive immediately in the same period.
    
#     Returns: (I1, I2, B1, B2, total_cost, period_profit, sales)
#     """
#     # (1) local inventory positions
#     IP1 = I1 - B1
#     IP2 = I2 - B2

#     # (2) order-up-to (local base stock)
#     O1 = max(0, int(s1_loc) - int(IP1))
#     O2 = max(0, int(s2_loc) - int(IP2))

#     # (3) arrivals (lead time = 0, orders arrive immediately)
#     # Operations receives from external source (production)
#     I2 += O2
    
#     # (4) releases (supplier ships to retailer)
#     ship = min(I2, B2 + O1)
#     I2 -= ship
#     B2 = B2 + O1 - ship
#     # Marketing receives shipment immediately
#     I1 += ship

#     # (5) demand at retailer - USES NORMAL DISTRIBUTION
#     D = params.sample_demand(rng, price)
#     sales = min(I1, B1 + D)
#     I1 -= sales
#     B1 = B1 + D - sales

#     # (6) period profit and costs
#     # Revenue
#     revenue = price * sales
    
#     # Production cost: k * x^2 where x = O2 (production quantity)
#     production_cost = params.k * (O2 ** 2)
    
#     # Holding costs (echelon accounting)
#     # Marketing: (h1-h2) * I1 (incremental)
#     # Operations: h2 * (I1+I2) (total downstream)
#     # Total = h1*I1 + h2*I2 (mathematically equivalent)
#     holding_cost_marketing = (h1 - h2) * I1
#     holding_cost_operations = h2 * (I1 + I2)
    
#     # Backorder costs (split according to alpha)
#     backorder_cost_marketing = alpha * p_bo * B1
#     backorder_cost_operations = (1.0 - alpha) * p_bo * B1
    
#     # Total costs for each agent
#     H1 = holding_cost_marketing + backorder_cost_marketing
#     H2 = holding_cost_operations + backorder_cost_operations + production_cost
#     total_cost = float(H1 + H2)
    
#     # System profit
#     period_profit = revenue - production_cost - (holding_cost_marketing + holding_cost_operations) - (backorder_cost_marketing + backorder_cost_operations)

#     return I1, I2, B1, B2, total_cost, period_profit, sales


def estimate_avg_total_cost(
        *, s1_loc, s2_loc, seed,
        rounds, warmup, price, h1, h2, p_bo, alpha
):
    """Estimate long-run average total cost under fixed (s1_loc, s2_loc).
    Updated to use Normal distribution via params.sample_demand.
    Lead time = 0, so orders arrive immediately.
    
    Returns: (avg_cost, avg_profit)
    """
    rng = np.random.default_rng(seed)
    I1 = I2 = 0
    B1 = B2 = 0
    total_cost = 0.0
    total_profit = 0.0
    count = 0

    for t in range(rounds + warmup):
        I1, I2, B1, B2, c, profit, _ = _env_step_once(
            rng, I1, I2, B1, B2,
            s1_loc, s2_loc, price, h1, h2, p_bo, alpha
        )
        if t >= warmup:
            total_cost += c
            total_profit += profit
            count += 1

    avg_cost = total_cost / max(1, count)
    avg_profit = total_profit / max(1, count)
    return avg_cost, avg_profit


def compute_supply_optimum_local(
        *,
        s_lower, s_upper, seed,
        rounds, warmup, price, h1, h2, p_bo, alpha
):
    """Enumerate over (s1, s2) using env_step_once. Updated for Normal demand model."""
    best_s1 = s_lower
    best_s2 = s_lower
    best_cost = float("inf")
    best_profit = float("-inf")

    for s1 in range(s_lower, s_upper + 1):
        for s2 in range(s_lower, s_upper + 1):
            c, profit = estimate_avg_total_cost(
                s1_loc=s1, s2_loc=s2,
                seed=seed, rounds=rounds, warmup=warmup,
                price=price, h1=h1, h2=h2, p_bo=p_bo, alpha=alpha,
            )
            if c < best_cost:
                best_cost = c
                best_profit = profit
                best_s1 = s1
                best_s2 = s2

    return int(best_s1), int(best_s2), float(best_cost), float(best_profit)


def compute_optimum_via_env_step(
    *,
    p_range=None,
    s1_range=None,
    s2_range=None,
    seed=None,
    rounds=None,
    warmup=None,
    verbose=True
):
    """
    Find the system-optimal policy using env_step_once path (with backorders).
    This uses the legacy simulation path for validation.
    
    Returns:
        (p_opt, s1_opt, s2_opt, avg_profit_opt, avg_cost_opt)
    """
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
    best_cost = float("inf")
    
    total_combinations = len(p_range) * len(s1_range) * len(s2_range)
    
    if verbose:
        print(f"Computing optimum via env_step_once over {total_combinations} combinations...")
    
    evaluated = 0
    for p in p_range:
        for s1 in s1_range:
            for s2 in s2_range:
                cost, profit = estimate_avg_total_cost(
                    s1_loc=s1, s2_loc=s2,
                    seed=seed, rounds=rounds, warmup=warmup,
                    price=p, h1=params.H1, h2=params.H2,
                    p_bo=params.P_BO, alpha=params.ALPHA
                )
                
                if profit > best_profit:
                    best_profit = profit
                    best_cost = cost
                    best_p = p
                    best_s1 = s1
                    best_s2 = s2
                
                evaluated += 1
                if verbose and evaluated % 100 == 0:
                    print(f"  Evaluated {evaluated}/{total_combinations}...")
    
    if verbose:
        print(f"\n{'='*50}")
        print("CENTRALIZED OPTIMUM (via env_step_once):")
        print(f"  Price (p*):        {best_p}")
        print(f"  Base-stock s1*:    {best_s1}")
        print(f"  Base-stock s2*:    {best_s2}")
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
    
    # Method 1: Quick optimization using _simulate_period
    print("\n" + "="*50)
    print("METHOD 1: Using _simulate_period (no backorders)")
    print("="*50)
    p_opt1, s1_opt1, s2_opt1, profit_opt1, cost_opt1 = compute_quick_optimum()
    
    # Method 2: Optimization using _env_step_once (with backorders)
    print("\n" + "="*50)
    print("METHOD 2: Using _env_step_once (with backorders)")
    print("="*50)
    p_opt2, s1_opt2, s2_opt2, profit_opt2, cost_opt2 = compute_optimum_via_env_step(
        p_range=params.p_range,
        s1_range=params.s1_range,
        s2_range=params.s2_range,
        # update rounds back to params.ROUNDS after testing
        rounds=5000,
        warmup=1000,
        verbose=True
    )
    
    # Comparison
    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    print(f"Method 1: p*={p_opt1}, s1*={s1_opt1}, s2*={s2_opt1}, Profit={profit_opt1:.2f}")
    print(f"Method 2: p*={p_opt2}, s1*={s1_opt2}, s2*={s2_opt2}, Profit={profit_opt2:.2f}")
    print(f"\nProfit difference: {abs(profit_opt1 - profit_opt2):.2f}")
    
    if abs(profit_opt1 - profit_opt2) < 50:
        print("✓ Both methods agree (difference < $50)")
    else:
        print("⚠ Methods disagree significantly!")
    
    print(f"\nTo use this benchmark, set in params.py:")
    print(f"  CTOT_OPT = {cost_opt2:.2f}  (from env_step_once method)")
