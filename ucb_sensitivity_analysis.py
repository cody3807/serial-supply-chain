"""
Sensitivity Analysis for UCB Supply Chain Model.

Tests the impact of different hyperparameters on:
- Convergence speed
- Final efficiency
- Stability
- Cumulative regret

Hyperparameters analyzed:
1. Beta range (transfer price to Operations)
2. Sigma range (transfer price from Marketing)
3. Action space granularity (step size)
4. UCB exploration constant (c in sqrt(c * ln(t) / n))
5. Number of rounds
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from copy import deepcopy

from model import TwoStageSupplyChainModel
from centralsolver import compute_quick_optimum
import params
from run_ucb import UCBSupplyChainModel, UCBMixin
from agents import PrincipalAgent, MarketingAgent, OperationsAgent


# ============================================
# Configurable UCB Mixin with Exploration Parameter
# ============================================

class ConfigurableUCBMixin(UCBMixin):
    """UCB Mixin with configurable exploration constant."""
    
    def __init__(self, *args, exploration_constant=2.0, **kwargs):
        self.exploration_constant = exploration_constant
        super().__init__(*args, **kwargs)
    
    def select_action_ucb(self):
        """Select action using UCB1 algorithm with configurable exploration."""
        untried = np.flatnonzero(self.counts == 0)
        
        if len(untried) > 0:
            action_idx = int(untried[0])
        else:
            t = self.model.t + 1
            if t > 1:
                confidence_bound = np.sqrt((self.exploration_constant * np.log(t)) / self.counts)
                ucb_values = self.average_reward + confidence_bound
            else:
                ucb_values = self.average_reward
            
            best = np.flatnonzero(ucb_values == ucb_values.max())
            action_idx = int(self.random.choice(best))
        
        self.action_idx = action_idx
        self.action = self.action_space[action_idx]


# ============================================
# Configurable UCB Agents
# ============================================

class ConfigurableUCBPrincipalAgent(PrincipalAgent, ConfigurableUCBMixin):
    """Principal with configurable UCB exploration."""
    
    def __init__(self, model, exploration_constant=2.0):
        self.exploration_constant = exploration_constant
        super().__init__(model)
        self.name = f"Principal (UCB c={exploration_constant})"
    
    def select_action(self):
        self.select_action_ucb()
        if self.action is not None:
            self.beta = float(self.action[0])
            self.sigma = float(self.action[1])


class ConfigurableUCBMarketingAgent(MarketingAgent, ConfigurableUCBMixin):
    """Marketing with configurable UCB exploration."""
    
    def __init__(self, model, exploration_constant=2.0):
        self.exploration_constant = exploration_constant
        super().__init__(model)
        self.name = f"Marketing (UCB c={exploration_constant})"
    
    def select_action(self):
        self.select_action_ucb()
        if self.action is not None:
            self.s1 = int(self.action[0])
            self.p = max(5, int(self.action[1]))


class ConfigurableUCBOperationsAgent(OperationsAgent, ConfigurableUCBMixin):
    """Operations with configurable UCB exploration."""
    
    def __init__(self, model, exploration_constant=2.0):
        self.exploration_constant = exploration_constant
        super().__init__(model)
        self.name = f"Operations (UCB c={exploration_constant})"
    
    def select_action(self):
        self.select_action_ucb()
        if self.action is not None:
            self.s2 = int(self.action)


# ============================================
# Configurable UCB Model
# ============================================

class ConfigurableUCBModel(UCBSupplyChainModel):
    """UCB model with configurable hyperparameters."""
    
    def __init__(self, exploration_constant=2.0):
        from mesa import Model
        from mesa.datacollection import DataCollector
        
        Model.__init__(self)
        
        self.rng = np.random.default_rng(params.SEED)
        self.t = 0
        
        # Create agents with custom exploration constant
        ConfigurableUCBPrincipalAgent(self, exploration_constant)
        ConfigurableUCBMarketingAgent(self, exploration_constant)
        ConfigurableUCBOperationsAgent(self, exploration_constant)
        
        self.rewards = {a: 0.0 for a in self.agents}
        
        # Initialize state variables (same as parent)
        self.I1 = 0
        self.I2 = 0
        self.beta = 0.0
        self.sigma = 0.0
        self.p = 0
        self.s1 = 0
        self.s2 = 0
        self.x = 0
        self.y = 0
        self.demand = 0
        self.sales = 0
        self.backorders = 0
        self.shipment = 0
        self.cost_marketing = 0.0
        self.cost_operations = 0.0
        self.total_cost = 0.0
        self.reward_principal = 0.0
        self.reward_beta = 0.0
        self.reward_sigma = 0.0
        self.reward_marketing = 0.0
        self.reward_operations = 0.0
        self.regret_vs_optimal = 0.0
        self.regret_cumulative = 0.0
        
        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Step": lambda m: m.t,
                "Beta": "beta",
                "Sigma": "sigma",
                "Price": "p",
                "S1 (Marketing)": "s1",
                "S2 (Operations)": "s2",
                "X (Production)": "x",
                "Y (Order)": "y",
                "I1 (Marketing Inv)": "I1",
                "I2 (Operations Inv)": "I2",
                "Demand": "demand",
                "Sales": "sales",
                "Backorders": "backorders",
                "Shipment": "shipment",
                "R_Beta": "reward_beta",
                "R_Sigma": "reward_sigma",
                "R_Marketing": "reward_marketing",
                "R_Operations": "reward_operations",
                "Total Cost": "total_cost",
                "Regret vs Optimal": "regret_vs_optimal",
                "Cumulative Regret": "regret_cumulative",
            },
            agent_reporters={
                "Agent Name": lambda a: getattr(a, 'name', a.__class__.__name__),
                "Action": "action",
                "Reward": "reward",
                "Cumulative Reward": "reward_cum",
            },
        )


# ============================================
# Sensitivity Analysis Functions
# ============================================

def run_single_experiment(config, verbose=False):
    """Run a single experiment with given configuration."""
    
    # Temporarily modify params
    original_params = {}
    for key, value in config.items():
        if hasattr(params, key):
            original_params[key] = getattr(params, key)
            setattr(params, key, value)
    
    # Create custom action spaces if needed
    if 'beta_step' in config or 'BETA_MIN' in config or 'BETA_MAX' in config:
        beta_min = config.get('BETA_MIN', params.BETA_MIN)
        beta_max = config.get('BETA_MAX', params.BETA_MAX)
        beta_step = config.get('beta_step', 5)
        params.beta_range = np.arange(beta_min, beta_max + 1, beta_step, dtype=int)
    
    if 'sigma_step' in config or 'SIGMA_MIN' in config or 'SIGMA_MAX' in config:
        sigma_min = config.get('SIGMA_MIN', params.SIGMA_MIN)
        sigma_max = config.get('SIGMA_MAX', params.SIGMA_MAX)
        sigma_step = config.get('sigma_step', 5)
        params.sigma_range = np.arange(sigma_min, sigma_max + 1, sigma_step, dtype=int)
    
    # Rebuild principal action space
    action_space = []
    for beta in params.beta_range:
        for sigma in params.sigma_range:
            action_space.append((int(beta), int(sigma)))
    params._principal_action_space = np.array(action_space)
    
    # Monkey-patch the action_space_principal function
    def custom_action_space_principal():
        return params._principal_action_space
    original_func = params.action_space_principal
    params.action_space_principal = custom_action_space_principal
    
    rounds = config.get('rounds', 5000)
    exploration_constant = config.get('exploration_constant', 2.0)
    
    # Run simulation
    model = ConfigurableUCBModel(exploration_constant=exploration_constant)
    
    for step in range(rounds):
        model.step()
        if verbose and (step + 1) % (rounds // 10) == 0:
            print(f"    Progress: {step + 1}/{rounds}")
    
    # Collect results
    df_model = model.datacollector.get_model_vars_dataframe()
    
    warmup_period = min(1000, rounds // 5)
    df_stable = df_model.iloc[warmup_period:]
    
    # Calculate metrics
    avg_cost = df_stable["Total Cost"].mean()
    avg_profit = -avg_cost
    final_cumulative_regret = df_model["Cumulative Regret"].iloc[-1]
    
    # Get modes for learned policy
    mode_beta = df_stable["Beta"].mode().iloc[0] if len(df_stable["Beta"].mode()) > 0 else 0
    mode_sigma = df_stable["Sigma"].mode().iloc[0] if len(df_stable["Sigma"].mode()) > 0 else 0
    mode_price = df_stable["Price"].mode().iloc[0] if len(df_stable["Price"].mode()) > 0 else 0
    mode_s1 = df_stable["S1 (Marketing)"].mode().iloc[0] if len(df_stable["S1 (Marketing)"].mode()) > 0 else 0
    mode_s2 = df_stable["S2 (Operations)"].mode().iloc[0] if len(df_stable["S2 (Operations)"].mode()) > 0 else 0
    
    # Calculate stability (std of last 20% of data)
    stability_window = df_model.iloc[int(0.8 * len(df_model)):]
    cost_stability = stability_window["Total Cost"].std()
    
    # Efficiency
    efficiency = (avg_profit / config.get('profit_opt', 1)) * 100 if config.get('profit_opt', 1) > 0 else 0
    
    results = {
        'avg_profit': avg_profit,
        'avg_cost': avg_cost,
        'efficiency': efficiency,
        'cumulative_regret': final_cumulative_regret,
        'cost_stability': cost_stability,
        'learned_beta': mode_beta,
        'learned_sigma': mode_sigma,
        'learned_price': mode_price,
        'learned_s1': mode_s1,
        'learned_s2': mode_s2,
        'n_beta_actions': len(params.beta_range),
        'n_sigma_actions': len(params.sigma_range),
        'n_principal_actions': len(params.beta_range) * len(params.sigma_range),
    }
    
    # Restore original params
    for key, value in original_params.items():
        setattr(params, key, value)
    params.action_space_principal = original_func
    
    return results, df_model


def sensitivity_analysis_beta_range():
    """Analyze impact of beta range on performance."""
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: BETA RANGE")
    print("=" * 70)
    
    # Compute benchmark once
    p_opt, s1_opt, s2_opt, profit_opt, cost_opt = compute_quick_optimum(verbose=False)
    params.CTOT_OPT = cost_opt
    
    # Test different beta ranges
    beta_configs = [
        {'name': 'Narrow (10-30)', 'BETA_MIN': 10, 'BETA_MAX': 30, 'beta_step': 5},
        {'name': 'Medium (0-50)', 'BETA_MIN': 0, 'BETA_MAX': 50, 'beta_step': 5},
        {'name': 'Wide (0-80)', 'BETA_MIN': 0, 'BETA_MAX': 80, 'beta_step': 5},
        {'name': 'Fine-grained (10-40)', 'BETA_MIN': 10, 'BETA_MAX': 40, 'beta_step': 2},
        {'name': 'Coarse (0-50)', 'BETA_MIN': 0, 'BETA_MAX': 50, 'beta_step': 10},
    ]
    
    results = []
    for config in beta_configs:
        print(f"\nTesting Beta Range: {config['name']}")
        print(f"  Range: [{config['BETA_MIN']}, {config['BETA_MAX']}], Step: {config['beta_step']}")
        
        full_config = {**config, 'rounds': 5000, 'profit_opt': profit_opt}
        result, _ = run_single_experiment(full_config, verbose=False)
        
        result['config_name'] = config['name']
        results.append(result)
        
        print(f"  Efficiency: {result['efficiency']:.2f}%")
        print(f"  Learned β: {result['learned_beta']}, σ: {result['learned_sigma']}")
        print(f"  Action space size: {result['n_principal_actions']} (β: {result['n_beta_actions']}, σ: {result['n_sigma_actions']})")
    
    return pd.DataFrame(results)


def sensitivity_analysis_sigma_range():
    """Analyze impact of sigma range on performance."""
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: SIGMA RANGE")
    print("=" * 70)
    
    p_opt, s1_opt, s2_opt, profit_opt, cost_opt = compute_quick_optimum(verbose=False)
    params.CTOT_OPT = cost_opt
    
    sigma_configs = [
        {'name': 'Narrow (20-40)', 'SIGMA_MIN': 20, 'SIGMA_MAX': 40, 'sigma_step': 5},
        {'name': 'Medium (0-60)', 'SIGMA_MIN': 0, 'SIGMA_MAX': 60, 'sigma_step': 5},
        {'name': 'Wide (0-100)', 'SIGMA_MIN': 0, 'SIGMA_MAX': 100, 'sigma_step': 5},
        {'name': 'Fine-grained (10-60)', 'SIGMA_MIN': 10, 'SIGMA_MAX': 60, 'sigma_step': 2},
        {'name': 'Coarse (0-60)', 'SIGMA_MIN': 0, 'SIGMA_MAX': 60, 'sigma_step': 10},
    ]
    
    results = []
    for config in sigma_configs:
        print(f"\nTesting Sigma Range: {config['name']}")
        print(f"  Range: [{config['SIGMA_MIN']}, {config['SIGMA_MAX']}], Step: {config['sigma_step']}")
        
        full_config = {**config, 'rounds': 5000, 'profit_opt': profit_opt}
        result, _ = run_single_experiment(full_config, verbose=False)
        
        result['config_name'] = config['name']
        results.append(result)
        
        print(f"  Efficiency: {result['efficiency']:.2f}%")
        print(f"  Learned β: {result['learned_beta']}, σ: {result['learned_sigma']}")
        print(f"  Action space size: {result['n_principal_actions']}")
    
    return pd.DataFrame(results)


def sensitivity_analysis_s1_range():
    """Analyze impact of s1 (Marketing base-stock) range on performance."""
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: S1 RANGE (Marketing Base-Stock)")
    print("=" * 70)
    
    p_opt, s1_opt, s2_opt, profit_opt, cost_opt = compute_quick_optimum(verbose=False)
    params.CTOT_OPT = cost_opt
    
    s1_configs = [
        {'name': 'Narrow (30-70)', 'S1_LOWER': 30, 'S1_UPPER': 70, 's1_step': 10},
        {'name': 'Medium (20-120)', 'S1_LOWER': 20, 'S1_UPPER': 120, 's1_step': 10},
        {'name': 'Wide (10-150)', 'S1_LOWER': 10, 'S1_UPPER': 150, 's1_step': 10},
        {'name': 'Fine-grained (30-80)', 'S1_LOWER': 30, 'S1_UPPER': 80, 's1_step': 5},
    ]
    
    results = []
    for config in s1_configs:
        print(f"\nTesting S1 Range: {config['name']}")
        print(f"  Range: [{config['S1_LOWER']}, {config['S1_UPPER']}], Step: {config['s1_step']}")
        
        # Temporarily modify s1_range
        original_s1 = params.s1_range
        s1_step = config.get('s1_step', 10)
        params.s1_range = np.arange(config['S1_LOWER'], config['S1_UPPER'] + 1, s1_step, dtype=int)
        
        # Rebuild marketing action space
        action_space = []
        for s1 in params.s1_range:
            for p in params.p_range:
                action_space.append((int(s1), int(p)))
        params._marketing_action_space = np.array(action_space)
        
        original_func = params.action_space_marketing
        params.action_space_marketing = lambda: params._marketing_action_space
        
        full_config = {'rounds': 5000, 'profit_opt': profit_opt}
        result, _ = run_single_experiment(full_config, verbose=False)
        
        # Restore
        params.s1_range = original_s1
        params.action_space_marketing = original_func
        
        result['config_name'] = config['name']
        result['n_s1_actions'] = len(params._marketing_action_space)
        results.append(result)
        
        print(f"  Efficiency: {result['efficiency']:.2f}%")
        print(f"  Learned s1: {result['learned_s1']}, p: {result['learned_price']}")
        print(f"  Marketing action space size: {result['n_s1_actions']}")
    
    return pd.DataFrame(results)


def sensitivity_analysis_s2_range():
    """Analyze impact of s2 (Operations base-stock) range on performance."""
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: S2 RANGE (Operations Base-Stock)")
    print("=" * 70)
    
    p_opt, s1_opt, s2_opt, profit_opt, cost_opt = compute_quick_optimum(verbose=False)
    params.CTOT_OPT = cost_opt
    
    s2_configs = [
        {'name': 'Narrow (20-60)', 'S2_LOWER': 20, 'S2_UPPER': 60, 's2_step': 10},
        {'name': 'Medium (10-120)', 'S2_LOWER': 10, 'S2_UPPER': 120, 's2_step': 10},
        {'name': 'Wide (0-150)', 'S2_LOWER': 0, 'S2_UPPER': 150, 's2_step': 10},
        {'name': 'Fine-grained (20-80)', 'S2_LOWER': 20, 'S2_UPPER': 80, 's2_step': 5},
    ]
    
    results = []
    for config in s2_configs:
        print(f"\nTesting S2 Range: {config['name']}")
        print(f"  Range: [{config['S2_LOWER']}, {config['S2_UPPER']}], Step: {config['s2_step']}")
        
        # Temporarily modify s2_range
        original_s2 = params.s2_range
        s2_step = config.get('s2_step', 10)
        params.s2_range = np.arange(config['S2_LOWER'], config['S2_UPPER'] + 1, s2_step, dtype=int)
        
        original_func = params.action_space_operations
        params.action_space_operations = lambda: params.s2_range
        
        full_config = {'rounds': 5000, 'profit_opt': profit_opt}
        result, _ = run_single_experiment(full_config, verbose=False)
        
        # Restore
        params.s2_range = original_s2
        params.action_space_operations = original_func
        
        result['config_name'] = config['name']
        result['n_s2_actions'] = len(params.s2_range)
        results.append(result)
        
        print(f"  Efficiency: {result['efficiency']:.2f}%")
        print(f"  Learned s2: {result['learned_s2']}")
        print(f"  Operations action space size: {result['n_s2_actions']}")
    
    return pd.DataFrame(results)


def sensitivity_analysis_price_range():
    """Analyze impact of price range on performance."""
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: PRICE RANGE")
    print("=" * 70)
    
    p_opt, s1_opt, s2_opt, profit_opt, cost_opt = compute_quick_optimum(verbose=False)
    params.CTOT_OPT = cost_opt
    
    price_configs = [
        {'name': 'Narrow (40-60)', 'p_min': 40, 'p_max': 60, 'p_step': 5},
        {'name': 'Medium (25-65)', 'p_min': 25, 'p_max': 65, 'p_step': 5},
        {'name': 'Wide (15-80)', 'p_min': 15, 'p_max': 80, 'p_step': 5},
        {'name': 'Fine-grained (40-65)', 'p_min': 40, 'p_max': 65, 'p_step': 2},
    ]
    
    results = []
    for config in price_configs:
        print(f"\nTesting Price Range: {config['name']}")
        print(f"  Range: [{config['p_min']}, {config['p_max']}], Step: {config['p_step']}")
        
        # Temporarily modify p_range
        original_p = params.p_range
        params.p_range = np.arange(config['p_min'], config['p_max'] + 1, config['p_step'], dtype=int)
        
        # Rebuild marketing action space
        action_space = []
        for s1 in params.s1_range:
            for p in params.p_range:
                action_space.append((int(s1), int(p)))
        params._marketing_action_space = np.array(action_space)
        
        original_func = params.action_space_marketing
        params.action_space_marketing = lambda: params._marketing_action_space
        
        full_config = {'rounds': 5000, 'profit_opt': profit_opt}
        result, _ = run_single_experiment(full_config, verbose=False)
        
        # Restore
        params.p_range = original_p
        params.action_space_marketing = original_func
        
        result['config_name'] = config['name']
        result['n_price_actions'] = len(params._marketing_action_space)
        results.append(result)
        
        print(f"  Efficiency: {result['efficiency']:.2f}%")
        print(f"  Learned price: {result['learned_price']}")
        print(f"  Marketing action space size: {result['n_price_actions']}")
    
    return pd.DataFrame(results)


def sensitivity_analysis_production_cost():
    """Analyze impact of production cost coefficient (k) on performance."""
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: PRODUCTION COST COEFFICIENT (k)")
    print("=" * 70)
    
    k_values = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    
    results = []
    for k_val in k_values:
        print(f"\nTesting k = {k_val}")
        
        # Temporarily modify k
        original_k = params.k
        params.k = k_val
        
        # Recompute optimal with new k
        p_opt, s1_opt, s2_opt, profit_opt, cost_opt = compute_quick_optimum(verbose=False)
        params.CTOT_OPT = cost_opt
        
        config = {'rounds': 5000, 'profit_opt': profit_opt, 'k': k_val}
        result, _ = run_single_experiment(config, verbose=False)
        
        # Restore
        params.k = original_k
        
        result['k_value'] = k_val
        results.append(result)
        
        print(f"  Optimal profit: {profit_opt:.2f}")
        print(f"  Efficiency: {result['efficiency']:.2f}%")
        print(f"  Cumulative Regret: {result['cumulative_regret']:.2f}")
    
    return pd.DataFrame(results)


def sensitivity_analysis_backorder_cost():
    """Analyze impact of backorder penalty (P_BO) on performance."""
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: BACKORDER PENALTY (P_BO)")
    print("=" * 70)
    
    p_bo_values = [10, 20, 30, 50, 75, 100]
    
    results = []
    for p_bo in p_bo_values:
        print(f"\nTesting P_BO = {p_bo}")
        
        # Temporarily modify P_BO
        original_p_bo = params.P_BO
        params.P_BO = p_bo
        
        # Recompute optimal with new P_BO
        p_opt, s1_opt, s2_opt, profit_opt, cost_opt = compute_quick_optimum(verbose=False)
        params.CTOT_OPT = cost_opt
        
        config = {'rounds': 5000, 'profit_opt': profit_opt, 'P_BO': p_bo}
        result, _ = run_single_experiment(config, verbose=False)
        
        # Restore
        params.P_BO = original_p_bo
        
        result['p_bo_value'] = p_bo
        results.append(result)
        
        print(f"  Optimal profit: {profit_opt:.2f}")
        print(f"  Efficiency: {result['efficiency']:.2f}%")
        print(f"  Cumulative Regret: {result['cumulative_regret']:.2f}")
    
    return pd.DataFrame(results)


def sensitivity_analysis_holding_costs():
    """Analyze impact of holding costs (H1, H2) on performance."""
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: HOLDING COSTS (H1, H2)")
    print("=" * 70)
    
    holding_configs = [
        {'name': 'Low H1=2, H2=1', 'H1': 2.0, 'H2': 1.0},
        {'name': 'Medium H1=5, H2=3', 'H1': 5.0, 'H2': 3.0},
        {'name': 'High H1=10, H2=6', 'H1': 10.0, 'H2': 6.0},
        {'name': 'Very High H1=15, H2=10', 'H1': 15.0, 'H2': 10.0},
        {'name': 'Equal H1=5, H2=5', 'H1': 5.0, 'H2': 5.0},
    ]
    
    results = []
    for config in holding_configs:
        print(f"\nTesting: {config['name']}")
        
        # Temporarily modify holding costs
        original_h1 = params.H1
        original_h2 = params.H2
        params.H1 = config['H1']
        params.H2 = config['H2']
        
        # Recompute optimal with new holding costs
        p_opt, s1_opt, s2_opt, profit_opt, cost_opt = compute_quick_optimum(verbose=False)
        params.CTOT_OPT = cost_opt
        
        full_config = {'rounds': 5000, 'profit_opt': profit_opt, 'H1': config['H1'], 'H2': config['H2']}
        result, _ = run_single_experiment(full_config, verbose=False)
        
        # Restore
        params.H1 = original_h1
        params.H2 = original_h2
        
        result['config_name'] = config['name']
        result['H1'] = config['H1']
        result['H2'] = config['H2']
        results.append(result)
        
        print(f"  Optimal profit: {profit_opt:.2f}")
        print(f"  Efficiency: {result['efficiency']:.2f}%")
        print(f"  Cumulative Regret: {result['cumulative_regret']:.2f}")
    
    return pd.DataFrame(results)


def sensitivity_analysis_exploration_constant():
    """Analyze impact of UCB exploration constant."""
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: UCB EXPLORATION CONSTANT (c)")
    print("=" * 70)
    
    p_opt, s1_opt, s2_opt, profit_opt, cost_opt = compute_quick_optimum(verbose=False)
    params.CTOT_OPT = cost_opt
    
    exploration_constants = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    
    results = []
    for c in exploration_constants:
        print(f"\nTesting Exploration Constant: c = {c}")
        
        config = {'rounds': 5000, 'exploration_constant': c, 'profit_opt': profit_opt}
        result, _ = run_single_experiment(config, verbose=False)
        
        result['exploration_constant'] = c
        results.append(result)
        
        print(f"  Efficiency: {result['efficiency']:.2f}%")
        print(f"  Cumulative Regret: {result['cumulative_regret']:.2f}")
        print(f"  Cost Stability (std): {result['cost_stability']:.2f}")
    
    return pd.DataFrame(results)


def plot_sensitivity_results(results_dict):
    """Create comprehensive visualization of sensitivity analysis."""
    
    n_analyses = len(results_dict)
    fig = plt.figure(figsize=(16, 4 * n_analyses))
    
    row = 0
    for analysis_name, df in results_dict.items():
        
        # Determine x-axis column
        if 'exploration_constant' in df.columns:
            x_col = 'exploration_constant'
            x_label = 'UCB Exploration Constant (c)'
        elif 'k_value' in df.columns:
            x_col = 'k_value'
            x_label = 'Production Cost Coefficient (k)'
        elif 'p_bo_value' in df.columns:
            x_col = 'p_bo_value'
            x_label = 'Backorder Penalty (P_BO)'
        elif 'H1' in df.columns:
            x_col = 'config_name'
            x_label = 'Holding Cost Configuration'
        else:
            x_col = 'config_name'
            x_label = 'Configuration'
        
        is_numeric = x_col in ['exploration_constant', 'k_value', 'p_bo_value']
        
        # Plot 1: Efficiency
        ax1 = plt.subplot(n_analyses, 3, row * 3 + 1)
        if is_numeric:
            ax1.plot(df[x_col], df['efficiency'], marker='o', linewidth=2, markersize=8)
        else:
            ax1.bar(range(len(df)), df['efficiency'])
            ax1.set_xticks(range(len(df)))
            ax1.set_xticklabels(df[x_col], rotation=45, ha='right')
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Efficiency (%)')
        ax1.set_title(f'{analysis_name}: Efficiency')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Optimal')
        ax1.legend()
        
        # Plot 2: Cumulative Regret
        ax2 = plt.subplot(n_analyses, 3, row * 3 + 2)
        if is_numeric:
            ax2.plot(df[x_col], df['cumulative_regret'], marker='o', linewidth=2, 
                    markersize=8, color='red')
        else:
            ax2.bar(range(len(df)), df['cumulative_regret'], color='red', alpha=0.7)
            ax2.set_xticks(range(len(df)))
            ax2.set_xticklabels(df[x_col], rotation=45, ha='right')
        ax2.set_xlabel(x_label)
        ax2.set_ylabel('Cumulative Regret')
        ax2.set_title(f'{analysis_name}: Cumulative Regret')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Action Space Size vs Efficiency (or stability for cost parameters)
        ax3 = plt.subplot(n_analyses, 3, row * 3 + 3)
        if 'n_principal_actions' in df.columns:
            scatter = ax3.scatter(df['n_principal_actions'], df['efficiency'], 
                                s=100, alpha=0.6, c=df['cumulative_regret'], 
                                cmap='RdYlGn_r')
            ax3.set_xlabel('Action Space Size (Principal)')
            ax3.set_ylabel('Efficiency (%)')
            ax3.set_title(f'{analysis_name}: Action Space vs Performance')
            plt.colorbar(scatter, ax=ax3, label='Cumulative Regret')
        else:
            # For cost parameters, show stability
            if is_numeric:
                ax3.plot(df[x_col], df['cost_stability'], marker='o', linewidth=2, 
                        markersize=8, color='purple')
            else:
                ax3.bar(range(len(df)), df['cost_stability'], color='purple', alpha=0.7)
                ax3.set_xticks(range(len(df)))
                ax3.set_xticklabels(df[x_col], rotation=45, ha='right')
            ax3.set_xlabel(x_label)
            ax3.set_ylabel('Cost Stability (std)')
            ax3.set_title(f'{analysis_name}: Cost Stability')
        ax3.grid(True, alpha=0.3)
        
        row += 1
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ucb_sensitivity_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {filename}")
    plt.show()


def run_full_sensitivity_analysis():
    """Run all sensitivity analyses."""
    
    print("\n" + "=" * 70)
    print("UCB SUPPLY CHAIN - COMPREHENSIVE SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    results = {}
    
    # 1. Beta Range
    df_beta = sensitivity_analysis_beta_range()
    results['Beta Range'] = df_beta
    
    # 2. Sigma Range
    df_sigma = sensitivity_analysis_sigma_range()
    results['Sigma Range'] = df_sigma
    
    # 3. S1 Range (Marketing Base-Stock)
    df_s1 = sensitivity_analysis_s1_range()
    results['S1 Range'] = df_s1
    
    # 4. S2 Range (Operations Base-Stock)
    df_s2 = sensitivity_analysis_s2_range()
    results['S2 Range'] = df_s2
    
    # 5. Price Range
    df_price = sensitivity_analysis_price_range()
    results['Price Range'] = df_price
    
    # 6. Exploration Constant
    df_exploration = sensitivity_analysis_exploration_constant()
    results['Exploration Constant'] = df_exploration
    
    # 7. Production Cost Coefficient (k)
    df_k = sensitivity_analysis_production_cost()
    results['Production Cost (k)'] = df_k
    
    # 8. Backorder Penalty (P_BO)
    df_pbo = sensitivity_analysis_backorder_cost()
    results['Backorder Penalty'] = df_pbo
    
    # 9. Holding Costs (H1, H2)
    df_holding = sensitivity_analysis_holding_costs()
    results['Holding Costs'] = df_holding
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name, df in results.items():
        filename = f"sensitivity_{name.replace(' ', '_').lower().replace('(', '').replace(')', '')}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\nSaved: {filename}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_sensitivity_results(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 70)
    
    for name, df in results.items():
        print(f"\n{name}:")
        print(f"  Best Efficiency: {df['efficiency'].max():.2f}%")
        print(f"  Worst Efficiency: {df['efficiency'].min():.2f}%")
        print(f"  Efficiency Range: {df['efficiency'].max() - df['efficiency'].min():.2f}%")
        if 'cumulative_regret' in df.columns:
            print(f"  Best Cumulative Regret: {df['cumulative_regret'].min():.2f}")
            print(f"  Worst Cumulative Regret: {df['cumulative_regret'].max():.2f}")
    
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = run_full_sensitivity_analysis()
