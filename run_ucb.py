"""
UCB (Upper Confidence Bound) version of the Split-Principal Supply Chain simulation.
Compares UCB1 learning algorithm against epsilon-greedy.
"""

import numpy as np
import pandas as pd
from copy import deepcopy

from model import TwoStageSupplyChainModel
from centralsolver import compute_quick_optimum
import params


# ============================================
# UCB Agent Mixin
# ============================================

class UCBMixin:
    """Mixin to add UCB1 action selection to any agent."""
    
    def select_action_ucb(self):
        """Select action using UCB1 algorithm."""
        # Try untried actions first
        untried = np.flatnonzero(self.counts == 0)
        
        if len(untried) > 0:
            # Explore: pick first untried action
            action_idx = int(untried[0])
        else:
            # UCB1: balance exploitation and exploration
            # UCB = average_reward + sqrt(2 * ln(t) / n_i)
            t = max(1, sum(self.counts))
            confidence_bound = np.sqrt((2.0 * np.log(t)) / self.counts)
            ucb_values = self.average_reward + confidence_bound
            
            # Pick action with highest UCB value
            best = np.flatnonzero(ucb_values == ucb_values.max())
            action_idx = int(self.random.choice(best))
        
        self.action_idx = action_idx
        self.action = self.action_space[action_idx]


# ============================================
# UCB Agent Classes
# ============================================

from agents import (
    PrincipalAgent, PrincipalBetaAgent, PrincipalSigmaAgent,
    MarketingAgent, OperationsAgent
)


class UCBPrincipalAgent(PrincipalAgent, UCBMixin):
    """Unified Principal with UCB1 learning."""
    
    def __init__(self, model):
        super().__init__(model)
        self.name = "Principal (UCB)"
    
    def select_action(self):
        self.select_action_ucb()
        if self.action is not None:
            self.beta = float(self.action[0])
            self.sigma = float(self.action[1])


class UCBPrincipalBetaAgent(PrincipalBetaAgent, UCBMixin):
    """Principal Beta with UCB1 learning."""
    
    def __init__(self, model):
        super().__init__(model)
        self.name = "Principal Beta (UCB)"
    
    def select_action(self):
        self.select_action_ucb()


class UCBPrincipalSigmaAgent(PrincipalSigmaAgent, UCBMixin):
    """Principal Sigma with UCB1 learning."""
    
    def __init__(self, model):
        super().__init__(model)
        self.name = "Principal Sigma (UCB)"
    
    def select_action(self):
        self.select_action_ucb()


class UCBMarketingAgent(MarketingAgent, UCBMixin):
    """Marketing Agent with UCB1 learning."""
    
    def __init__(self, model):
        super().__init__(model)
        self.name = "Marketing (UCB)"
    
    def select_action(self):
        self.select_action_ucb()
        if self.action is not None:
            self.s1 = int(self.action[0])
            self.p = max(5, int(self.action[1]))


class UCBOperationsAgent(OperationsAgent, UCBMixin):
    """Operations Agent with UCB1 learning."""
    
    def __init__(self, model):
        super().__init__(model)
        self.name = "Operations (UCB)"
    
    def select_action(self):
        self.select_action_ucb()
        if self.action is not None:
            self.s2 = int(self.action[0])
            self.tp = int(self.action[1])


# ============================================
# UCB Model
# ============================================

from mesa import Model
from mesa.datacollection import DataCollector


def create_ucb_agents(model):
    """Create UCB agents with unified Principal."""
    #UCBPrincipalAgent(model)  # Unified principal
    UCBMarketingAgent(model)
    UCBOperationsAgent(model)


class UCBSupplyChainModel(TwoStageSupplyChainModel):
    """Supply chain model using UCB1 learning."""
    
    def __init__(self, seed=None):
        # Skip parent __init__ and set up manually
        # Pass seed to Mesa's Model to make agent random choices reproducible
        Model.__init__(self, seed=seed if seed is not None else params.SEED)
        
        # Simulation control
        self.rng = np.random.default_rng(params.SEED)
        self.t = 0
        
        # Initialize UCB agents
        create_ucb_agents(self)
        self.rewards = {a: 0.0 for a in self.agents}
        
        # Inventory state
        self.I1 = 0
        self.I2 = 0
        
        # Decision variables
        self.beta = 0.0
        self.sigma = 0.0
        self.p = 0
        self.s1 = 0
        self.s2 = 0
        self.x = 0
        self.y = 0
        
        # State variables
        self.demand = 0
        self.sales = 0
        self.backorders = 0
        self.shipment = 0
        
        # Cost components
        self.cost_marketing = 0.0
        self.cost_operations = 0.0
        self.total_cost = 0.0
        
        # Rewards
        self.reward_principal = 0.0
        self.reward_beta = 0.0
        self.reward_sigma = 0.0
        self.reward_marketing = 0.0
        self.reward_operations = 0.0
        
        # Benchmark comparison
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
# Run UCB Simulation
# ============================================

def run_ucb_simulation(rounds=None, verbose=True):
    """Run simulation with UCB learning."""
    if rounds is None:
        rounds = params.ROUNDS
    
    print("=" * 60)
    print("UCB SUPPLY CHAIN MARL SIMULATION")
    print("=" * 60)
    print()
    
    # Compute benchmark
    print("STEP 1: Computing Centralized Benchmark")
    print("-" * 40)
    
    p_opt, s1_opt, s2_opt, profit_opt, cost_opt = compute_quick_optimum(verbose=True)
    params.CTOT_OPT = cost_opt
    
    print()
    
    # Run UCB simulation
    print("STEP 2: Running UCB MARL Simulation")
    print("-" * 40)
    print(f"Agents: All using UCB1 algorithm")
    print(f"Rounds: {rounds}")
    print()
    
    model = UCBSupplyChainModel()
    
    report_interval = max(1, rounds // 10)
    
    for step in range(rounds):
        if (step == 5001):
            print("Re-seeding RNG for extended exploration...")
            
        model.step()
        
        if verbose and (step + 1) % report_interval == 0:
            pct = 100 * (step + 1) / rounds
            print(f"  Progress: {step + 1}/{rounds} ({pct:.0f}%)")
    
    print()
    
    # Analyze results
    print("STEP 3: Analyzing Results")
    print("-" * 40)
    
    df_model = model.datacollector.get_model_vars_dataframe()
    
    warmup_period = rounds // 5
    df_stable = df_model.iloc[warmup_period:]
    
    avg_cost = df_stable["Total Cost"].mean()
    avg_profit = -avg_cost
    avg_backorders = df_stable["Backorders"].mean()
    avg_sales = df_stable["Sales"].mean()
    avg_I1 = df_stable["I1 (Marketing Inv)"].mean()
    avg_I2 = df_stable["I2 (Operations Inv)"].mean()
    final_cumulative_regret = df_model["Cumulative Regret"].iloc[-1]
    
    mode_beta = df_stable["Beta"].mode().iloc[0] if len(df_stable["Beta"].mode()) > 0 else 0
    mode_sigma = df_stable["Sigma"].mode().iloc[0] if len(df_stable["Sigma"].mode()) > 0 else 0
    mode_price = df_stable["Price"].mode().iloc[0] if len(df_stable["Price"].mode()) > 0 else 0
    mode_s1 = df_stable["S1 (Marketing)"].mode().iloc[0] if len(df_stable["S1 (Marketing)"].mode()) > 0 else 0
    mode_s2 = df_stable["S2 (Operations)"].mode().iloc[0] if len(df_stable["S2 (Operations)"].mode()) > 0 else 0
    mode_x = df_stable["X (Production)"].mode().iloc[0] if len(df_stable["X (Production)"].mode()) > 0 else 0
    
    print(f"Centralized Optimal:")
    print(f"  p* = {p_opt}, s1* = {s1_opt}, s2* = {s2_opt}")
    print(f"  Optimal Profit/period: {profit_opt:.2f}")
    print()
    
    print(f"UCB Decentralized (learned, stable period):")
    print(f"  β = {mode_beta}, σ = {mode_sigma}")
    print(f"  p = {mode_price}, s1 = {mode_s1}, s2 = {mode_s2}, x = {mode_x}")
    print(f"  Avg Profit/period: {avg_profit:.2f}")
    print(f"  Avg I1 (Marketing): {avg_I1:.2f}")
    print(f"  Avg I2 (Operations): {avg_I2:.2f}")
    print(f"  Avg Backorders: {avg_backorders:.2f}")
    print(f"  Avg Sales: {avg_sales:.2f}")
    print()
    
    efficiency = (avg_profit / profit_opt * 100) if profit_opt > 0 else 0
    print(f"Performance Comparison:")
    print(f"  Efficiency vs Optimal: {efficiency:.1f}%")
    print(f"  Avg Regret/period: {(avg_cost - cost_opt):.2f}")
    print(f"  Cumulative Regret: {final_cumulative_regret:.2f}")
    print()
    
    # ============================================
    # 4. PLOT DECISION VARIABLES OVER TIME
    # ============================================
    print("STEP 4: Plotting Decision Variables")
    print("-" * 40)
    
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # Rolling window for smoothing
    window = min(500, rounds // 20)
    
    # Decision variables to plot
    decision_vars = {
        "Beta": ("Transfer Price", "blue", None),
        
        "Price": ("p (Market Price)", "green", p_opt),
        "S1 (Marketing)": ("s₁ (Marketing Base-Stock)", "red", s1_opt),
        "S2 (Operations)": ("s₂ (Operations Base-Stock)", "purple", s2_opt),
        "X (Production)": ("x (Production Quantity)", "brown", None),
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    time = df_model.index.values
    
    for idx, (col, (label, color, optimal)) in enumerate(decision_vars.items()):
        ax = axes[idx]
        
        # Plot raw values (semi-transparent)
        ax.plot(time, df_model[col], alpha=0.15, color=color, linewidth=0.5)
        
        # Plot rolling average
        rolling_avg = df_model[col].rolling(window=window, min_periods=1).mean()
        ax.plot(time, rolling_avg, color=color, linewidth=2, label=f"Rolling Avg (w={window})")
        
        # Plot optimal line if available
        if optimal is not None:
            ax.axhline(y=optimal, color='black', linestyle='--', linewidth=1.5, 
                      label=f"Optimal = {optimal}")
        
        ax.set_xlabel("Time (Period)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} Over Time")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"UCB without Principal Decision Variable Convergence (Efficiency: {efficiency:.1f}%)", 
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"ucb_convergence_plots_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    print(f"Plots saved to: {plot_filename}")
    
    # Show plot
    plt.show()
    print()
    
    print("=" * 60)
    print("UCB SIMULATION COMPLETE")
    print("=" * 60)
    
    return model, df_model, {
        'p_opt': p_opt, 's1_opt': s1_opt, 's2_opt': s2_opt,
        'profit_opt': profit_opt, 'efficiency': efficiency
    }


if __name__ == "__main__":
    model, df_model, results = run_ucb_simulation()
