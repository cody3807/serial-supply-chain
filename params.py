"""
Parameter configuration for the two-stage serial supply chain (Local-Inventory game)
"""

import numpy as np
from centralsolver import compute_supply_optimum_local

# Cost structure
H1 = 3         
H2 = 5        
P_BO = 15      
ALPHA = 0.5      

# End-customer demand
a=120
b=1.5

std_dev = 10
p = [25,30,35,40,45,50,55,60,65]

# production cost
k = 0.35
def sample_demand(rng,p):
 
    """One draw of consumer demand D_t ~ Poisson(LAM)"""
    # Demand is depending on the price and follows a normal distribution
    
    mean = a - b * p
    #return int(rng.gauss(mean, std_dev))
    return int(rng.normal(mean, std_dev,1).item())

# Action space (base-stock levels)
S_LOWER = 20
S_UPPER = 120
s_range = np.arange(S_LOWER, S_UPPER + 1, 5, dtype=int)
def action_space():
    """Return array of discrete base-stock levels"""
    return np.arange(S_LOWER, S_UPPER + 1, 5, dtype=int)
# Action Space sigma, beta
# σ = β zorunlu (tek transfer fiyatı)
multiplier_p=2
def action_space_principal():
    """Principal sadece σ = β seçebilir (tek transfer price)"""
    price_range = np.arange(0, 71, 5, dtype=int)
    
    # Sadece σ = β kombinasyonları
    action_space = [(p, p) for p in price_range]
    return np.array(action_space)
# ε-greedy schedule
EPS_START = 0.80
EPS_END   = 0.02
def epsilon_at(t, rounds):
    """Linear decay of epsilon from EPS_START → EPS_END over [0, rounds-1]"""
    if rounds <= 1:
        return EPS_END
    frac = np.clip(t, 0, rounds - 1) / (rounds - 1)
    return (1.0 - frac) * EPS_START + frac * EPS_END

# Simulation control
ROUNDS = 10000
SEED   = 42

#Action space for marketing
def action_space_marketing():
    """Return array of discrete base-stock levels for marketing agent"""
    action_space=[]
    for i in s_range:
        for j in p:
            action_space.append((i,j))
    return np.array(action_space)

#Action space for operations
def action_space_operation():
    """Return array of discrete base-stock levels for operations agent"""
    action_space=[]
    for i in s_range:
        
        action_space.append(i)
    return np.array(action_space)
# Benchmark (Echeleon base stock levels, transferred to local base-stocks)
WARMUP = 2000 # start to estimate here

# Placeholder values - comment out the loop below for faster execution
CTOT_OPT = 0.0  # Will be computed if needed

# Uncomment below to compute optimal values (slow)
# for i in p:
#     S1_OPT_LOC, S2_OPT_LOC, CTOT_OPT = compute_supply_optimum_local(
#         s_lower=S_LOWER, s_upper=S_UPPER, seed=SEED,
#         rounds=ROUNDS, warmup=WARMUP, lam=i,
#         h1=H1, h2=H2,p_bo=P_BO, alpha=ALPHA
#     )
#     print(i,S1_OPT_LOC, S2_OPT_LOC, CTOT_OPT)


