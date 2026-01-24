"""
Centralized (system-optimal) benchmark solver for the 2-stage serial supply chain.
Computes the optimal pair of local base-stock levels (s1*, s2*) by enumeration
and estimates the corresponding long-run average total cost via Monte Carlo.
"""

import numpy as np
#m params import sample_demand
a=120
b=1.5

std_dev = 10
#std_dev = 10
def sample_demand(rng,p):
 
    """One draw of consumer demand D_t ~ Poisson(LAM)"""
    # Demand is depending on the price and follows a normal distribution
    
    mean = a - b * p
    #return int(rng.gauss(mean, std_dev))
    return int(rng.normal(mean, std_dev,1))


def _env_step_once(
        rng, I1, I2, B1, B2, U1_prev, U2_prev,
        s1_loc, s2_loc, lam, h1, h2, p_bo, alpha
):
    """
    Simulates inventory transitions within one period.
    """

    # (1) arrivals
    I1 += U1_prev
    I2 += U2_prev

    # (2) local inventory positions
    IP1 = I1 - B1
    IP2 = I2 - B2

    # (3) order-up-to (local base stock)
    O1 = max(0, int(s1_loc) - int(IP1))
    O2 = max(0, int(s2_loc) - int(IP2))

    # (4) releases (supplier ships to retailer; source is unconstrained)
    ship = min(I2, B2 + O1)
    I2 -= ship
    B2 = B2 + O1 - ship
    U1 = ship
    U2 = O2

    # (5) demand at retailer
    D = sample_demand(rng,p=lam)
    sales = min(I1, B1 + D)
    I1 -= sales
    B1 = B1 + D - sales

    # (6) end-of-period costs (match model.py)
    H1 = (h1 + h2) * I1 + alpha * p_bo * B1
    H2 = h2 * (I2 + U1) + (1.0 - alpha) * p_bo * B1
    total_cost = float(H1 + H2)

    return I1, I2, B1, B2, U1, U2, total_cost


def estimate_avg_total_cost(
        *, s1_loc, s2_loc, seed,
        rounds, warmup, lam, h1, h2, p_bo, alpha
):
    """
    Estimate long-run average total cost under fixed (s1_loc, s2_loc) via Monte Carlo.
    """
    rng = np.random.default_rng(seed)

    # initial state (align with model.py)
    I1 = I2 = 0
    B1 = B2 = 0
    U1_prev = U2_prev = 0

    total = 0.0
    count = 0

    for t in range(rounds + warmup):
        I1, I2, B1, B2, U1_prev, U2_prev, c = _env_step_once(
            rng, I1, I2, B1, B2, U1_prev, U2_prev,
            s1_loc, s2_loc, lam, h1, h2, p_bo, alpha
        )
        if t >= warmup:
            total += c
            count += 1

    return total / max(1, count)


def compute_supply_optimum_local(
        *,
        s_lower, s_upper, seed,
        rounds, warmup, lam, h1, h2, p_bo, alpha
):
    """
    Enumerate over (s1, s2) in {s_lower,...,s_upper}^2 and return:
      (s1_opt, s2_opt, ctot_opt)
    where ctot_opt is the estimated average total cost.
    """
    best_s1 = s_lower
    best_s2 = s_lower
    best_cost = float("inf")

    for s1 in range(s_lower, s_upper + 1):
        for s2 in range(s_lower, s_upper + 1):
            c = estimate_avg_total_cost(
                s1_loc=s1,
                s2_loc=s2,
                seed=seed,
                rounds=rounds,
                warmup=warmup,
                lam=lam,
                h1=h1,
                h2=h2,
                p_bo=p_bo,
                alpha=alpha,
            )
            if c < best_cost:
                best_cost = c
                best_s1 = s1
                best_s2 = s2

    return int(best_s1), int(best_s2), float(best_cost)
