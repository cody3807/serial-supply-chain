"""
Parameter configuration for the two-stage serial supply chain (Local-Inventory game)
"""

import numpy as np
from centralsolver import compute_supply_optimum_local

# Cost structure
H1 = 0.5          
H2 = 0.5          
P_BO = 50       
ALPHA = 0.5       

# End-customer demand
a=100
b=3

std_dev = 10
p = [5,7,9,11,13,15]

# production cost
k = 4
def sample_demand(rng,p):
 
    """One draw of consumer demand D_t ~ Poisson(LAM)"""
    # Demand is depending on the price and follows a normal distribution
    p=11
    mean = a - b * p
    #return int(rng.gauss(mean, std_dev))
    return int(rng.normal(mean, std_dev,1))

# Action space (base-stock levels)
S_LOWER = 0
S_UPPER = 60
s_range = np.arange(S_LOWER, S_UPPER + 1, dtype=int)
def action_space():
    """Return array of discrete base-stock levels"""
    return np.arange(S_LOWER, S_UPPER + 1, dtype=int)
# Action Space sigma, beta
multiplier_p=2
def action_space_principal():
    price_range = np.arange(min(p) * -multiplier_p, max(p) * multiplier_p + 1,3, dtype=int)
    action_space = []
    for i in price_range:
        for j in price_range:
            action_space.append((i,j))
    return np.array(action_space)
# ε-greedy schedule
EPS_START = 0.8
EPS_END   = 0.05
def epsilon_at(t, rounds):
    """Linear decay of epsilon from EPS_START → EPS_END over [0, rounds-1]"""
    if rounds <= 1:
        return EPS_END
    frac = np.clip(t, 0, rounds - 1) / (rounds - 1)
    return (1.0 - frac) * EPS_START + frac * EPS_END

# Simulation control
ROUNDS = 3650
SEED   = 42

#Action space for marketing
def action_space_marketing():
    """Return array of discrete base-stock levels for marketing agent"""
    action_space=[]
    for i in s_range:
        for j in p:
            action_space.append((i,j))
    return np.array(action_space)

#Action space for marketing
def action_space_operation():
    """Return array of discrete base-stock levels for marketing agent"""
    action_space=[]
    for i in s_range:
        for j in p:
            action_space.append((i,j))
    return np.array(action_space)
# Benchmark (Echeleon base stock levels, transferred to local base-stocks)
WARMUP = 200 # start to estimate here
S1_OPT_LOC, S2_OPT_LOC, CTOT_OPT = compute_supply_optimum_local(
    s_lower=S_LOWER, s_upper=S_UPPER, seed=SEED,
    rounds=ROUNDS, warmup=WARMUP, lam=p[0],
    h1=H1, h2=H2,p_bo=P_BO, alpha=ALPHA
)


