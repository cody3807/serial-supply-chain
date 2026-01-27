"""
Parameter configuration for the two-stage serial supply chain
Split-Principal Architecture (Principal Beta & Principal Sigma)
Based on Cachon & Zipkin (1999) and Kouvelis & Lariviere (2000)
"""

import numpy as np

# ============================================
# Cost Structure
# ============================================
H1 = 3.0          # Retailer (Marketing) holding cost
H2 = 2.0          # Supplier (Operations) holding cost
P_BO = 25         # Total backorder penalty (π)
ALPHA = 0.3       # Penalty split ratio (Marketing pays α, Operations pays 1-α)
k = 0.05          # Convex production cost coefficient

# ============================================
# Demand Parameters (Price-dependent Normal)
# D ~ Normal(a - b*p, sigma_d)
# ============================================
a = 120           # Demand intercept
b = 1.5           # Price sensitivity coefficient  
sigma_d = 10      # Demand standard deviation

def sample_demand(rng, price):
    """
    Generate price-dependent stochastic demand.
    D ~ Normal(mean = a - b*p, std = sigma_d)
    Demand is constrained to be non-negative.
    """
    mean = max(0, a - b * price)
    demand = rng.normal(mean, sigma_d)
    return max(0, int(round(demand)))

# ============================================
# Action Spaces
# ============================================

# Base-stock levels (for Marketing s1 and Operations s2)
S_LOWER = 0
S_UPPER = 80
s_range = np.arange(S_LOWER, S_UPPER + 1, 5, dtype=int)  # Discretized for tractability

# Price range for Marketing (explicit set)
p_range = np.array([30, 35, 40, 45, 50, 55, 60], dtype=int)

# Production range for Operations
X_MIN = 0
X_MAX = 80
x_range = np.arange(X_MIN, X_MAX + 1, 5, dtype=int)

# Transfer price ranges for Principals
BETA_MIN = 0
BETA_MAX = 30
beta_range = np.arange(BETA_MIN, BETA_MAX + 1, 3, dtype=int)

SIGMA_MIN = 0
SIGMA_MAX = 30
sigma_range = np.arange(SIGMA_MIN, SIGMA_MAX + 1, 3, dtype=int)

def action_space():
    """Return array of discrete base-stock levels (legacy)"""
    return s_range

def action_space_principal_beta():
    """Return array of discrete beta (buy price) values"""
    return beta_range

def action_space_principal_sigma():
    """Return array of discrete sigma (sell price) values"""
    return sigma_range

def action_space_marketing():
    """
    Return array of (s1, p) tuples for Marketing agent.
    s1: base-stock level, p: market price
    """
    action_space = []
    for s1 in s_range:
        for p in p_range:
            action_space.append((int(s1), int(p)))
    return np.array(action_space)

def action_space_operation():
    """
    Return array of (s2, x) tuples for Operations agent.
    s2: base-stock level, x: production quantity
    """
    action_space = []
    for s2 in s_range:
        for x in x_range:
            action_space.append((int(s2), int(x)))
    return np.array(action_space)

# ============================================
# ε-greedy Learning Schedule
# ============================================
EPS_START = 0.95
EPS_END = 0.00

def epsilon_at(t, rounds):
    """Linear decay of epsilon from EPS_START → EPS_END over [0, rounds-1]"""
    if rounds <= 1:
        return EPS_END
    frac = np.clip(t, 0, rounds - 1) / (rounds - 1)
    return (1.0 - frac) * EPS_START + frac * EPS_END

# ============================================
# Simulation Control
# ============================================
ROUNDS = 5000     # Number of simulation steps
SEED = 42         # Random seed for reproducibility
WARMUP = 500      # Warmup period for benchmark estimation

# ============================================
# Centralized Benchmark (computed at import time)
# ============================================
# Note: The benchmark will be computed in centralsolver.py
# and imported separately to avoid circular dependencies
CTOT_OPT = 0.0  # Placeholder - will be set after running centralsolver
