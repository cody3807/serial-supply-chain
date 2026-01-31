"""
Parameter configuration for the two-stage serial supply chain
Split-Principal Architecture (Principal Beta & Principal Sigma)
Based on Cachon & Zipkin (1999) and Kouvelis & Lariviere (2000)
"""

import numpy as np

# ============================================
# Cost Structure (Realistic)
# ============================================
# ============================================
# Cost Structure (Realistic)
# ============================================
H1 = 5.0          # Retailer (Marketing) holding cost
H2 = 3.0          # Supplier (Operations) holding cost
P_BO = 1000       # Total backorder penalty (π) - HUGE
ALPHA = 0.5       # Penalty split ratio
k = 0.35          # Convex production cost coefficient

# ============================================
# Demand Parameters (Price-dependent Normal)
# D ~ Normal(a - b*p, sigma_d)
# ============================================
a = 1200          # Demand intercept - HUGE
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
# Action Spaces (WIDE - for realistic testing)
# ============================================

# Base-stock for Marketing (s1) - demand approx 1100
S1_LOWER = 20
S1_UPPER = 120
s1_range = np.arange(S1_LOWER, S1_UPPER + 1, 25, dtype=int)

# Base-stock for Operations (s2)
S2_LOWER = 20
S2_UPPER = 120
s2_range = np.arange(S2_LOWER, S2_UPPER + 1, 25, dtype=int)

# Legacy combined range (for backwards compatibility)
s_range = s1_range

# Price range for Marketing - wide range
p_range = np.array([25, 30, 35, 40, 45, 50, 55, 60, 65], dtype=int)  # 9 values

# Transfer price ranges - VERY WIDE for full exploration
BETA_MIN = 0
BETA_MAX = 50
beta_range = np.arange(BETA_MIN, BETA_MAX + 1, 5, dtype=int)  # 0,5,10,...,50 (11 values)

# Sigma - VERY WIDE range
SIGMA_MIN = 0
SIGMA_MAX = 60
sigma_range = np.arange(SIGMA_MIN, SIGMA_MAX + 1, 5, dtype=int)  # 10,15,...,70 (13 values)

def action_space():
    """Return array of discrete base-stock levels (legacy)"""
    return s_range

def action_space_principal_beta():
    """Return array of discrete beta (buy price) values"""
    return beta_range

def action_space_principal_sigma():
    """Return array of discrete sigma (sell price) values"""
    return sigma_range

def action_space_principal():
    """
    Return array of (beta, sigma) tuples for unified Principal agent.
    Principal controls both transfer prices to minimize total system cost.
    """
    action_space = []
    for beta in beta_range:
        for sigma in sigma_range:
            action_space.append((int(beta), int(sigma)))
    return np.array(action_space)

def action_space_marketing():
    """
    Return array of (s1, p) tuples for Marketing agent.
    s1: base-stock level, p: market price
    """
    action_space = []
    for s1 in s1_range:
        for p in p_range:
            action_space.append((int(s1), int(p)))
    return np.array(action_space)

def action_space_operations():
    """Return array of discrete s2 base-stock levels for Operations agent."""
    return s2_range

# ============================================
# ε-greedy Learning Schedule
# ============================================
EPS_START = 0.80   # More exploration initially
EPS_END = 0.02     # Small residual exploration

def epsilon_at(t, rounds):
    """Linear decay of epsilon from EPS_START → EPS_END over [0, rounds-1]"""
    if rounds <= 1:
        return EPS_END
    frac = np.clip(t, 0, rounds - 1) / (rounds - 1)
    return (1.0 - frac) * EPS_START + frac * EPS_END

# ============================================
# Simulation Control
# ============================================
ROUNDS = 5000     # More rounds for better exploration
SEED = 42         # Random seed for reproducibility
WARMUP = 2000     # Warmup period for benchmark estimation

# ============================================
# Centralized Benchmark (computed at import time)
# ============================================
# Note: The benchmark will be computed in centralsolver.py
# and imported separately to avoid circular dependencies
CTOT_OPT = 0.0  # Placeholder - will be set after running centralsolver
