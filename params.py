"""
Parameter configuration for the two-stage serial supply chain (Local-Inventory game)
"""

import numpy as np
from centralsolver import compute_supply_optimum_local

# Cost structure
H1 = 0.5          
H2 = 0.5          
P_BO = 5.0        
ALPHA = 0.5       

# End-customer demand
LAM = 20.0       
def sample_demand(rng):
    """One draw of consumer demand D_t ~ Poisson(LAM)"""
    return int(rng.poisson(LAM))

# Action space (base-stock levels)
S_LOWER = 0
S_UPPER = 60
def action_space():
    """Return array of discrete base-stock levels"""
    return np.arange(S_LOWER, S_UPPER + 1, dtype=int)

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
ROUNDS = 365
SEED   = 42

# Benchmark (Echeleon base stock levels, transferred to local base-stocks)
WARMUP = 200 # start to estimate here
S1_OPT_LOC, S2_OPT_LOC, CTOT_OPT = compute_supply_optimum_local(
    s_lower=S_LOWER, s_upper=S_UPPER, seed=SEED,
    rounds=ROUNDS, warmup=WARMUP, lam=LAM,
    h1=H1, h2=H2,p_bo=P_BO, alpha=ALPHA
)