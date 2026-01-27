"""
Run script for the Split-Principal Supply Chain MARL Simulation.

This script:
1. Computes the centralized benchmark (system optimum)
2. Runs the decentralized MARL simulation
3. Compares performance against the benchmark
4. Exports results to Excel
"""

import numpy as np
import pandas as pd
from datetime import datetime

from model import TwoStageSupplyChainModel
from centralsolver import compute_quick_optimum
import params


def run_simulation(rounds=None, verbose=True):
    """
    Run the complete simulation with benchmark comparison.
    """
    if rounds is None:
        rounds = params.ROUNDS
    
    print("=" * 60)
    print("SPLIT-PRINCIPAL SUPPLY CHAIN MARL SIMULATION")
    print("=" * 60)
    print()
    
    # ============================================
    # 1. COMPUTE CENTRALIZED BENCHMARK
    # ============================================
    print("STEP 1: Computing Centralized Benchmark")
    print("-" * 40)
    
    p_opt, s1_opt, s2_opt, profit_opt, cost_opt = compute_quick_optimum(verbose=True)
    
    # Update the benchmark for regret calculation
    params.CTOT_OPT = cost_opt
    
    print()
    
    # ============================================
    # 2. RUN DECENTRALIZED SIMULATION
    # ============================================
    print("STEP 2: Running Decentralized MARL Simulation")
    print("-" * 40)
    print(f"Agents: Principal (unified), Marketing, Operations")
    print(f"Rounds: {rounds}")
    print(f"Learning: ε-greedy with decay ({params.EPS_START} → {params.EPS_END})")
    print()
    
    # Create and run model with unified Principal
    model = TwoStageSupplyChainModel(
        agent_types=("principal", "greedy_m", "greedy_o")
    )
    
    # Progress reporting
    report_interval = max(1, rounds // 10)
    
    for step in range(rounds):
        model.step()
        
        if verbose and (step + 1) % report_interval == 0:
            pct = 100 * (step + 1) / rounds
            print(f"  Progress: {step + 1}/{rounds} ({pct:.0f}%)")
    
    print()
    
    # ============================================
    # 3. COLLECT AND ANALYZE RESULTS
    # ============================================
    print("STEP 3: Analyzing Results")
    print("-" * 40)
    
    df_model = model.datacollector.get_model_vars_dataframe()
    df_agents = model.datacollector.get_agent_vars_dataframe()
    
    # Compute summary statistics
    warmup_period = min(500, rounds // 5)  # Exclude early learning period
    
    df_stable = df_model.iloc[warmup_period:]
    
    avg_cost = df_stable["Total Cost"].mean()
    avg_profit = -avg_cost
    avg_backorders = df_stable["Backorders"].mean()
    avg_sales = df_stable["Sales"].mean()
    avg_I1 = df_stable["I1 (Marketing Inv)"].mean()
    avg_I2 = df_stable["I2 (Operations Inv)"].mean()
    final_cumulative_regret = df_model["Cumulative Regret"].iloc[-1]
    
    # Most common actions in stable period
    mode_beta = df_stable["Beta"].mode().iloc[0] if len(df_stable["Beta"].mode()) > 0 else 0
    mode_sigma = df_stable["Sigma"].mode().iloc[0] if len(df_stable["Sigma"].mode()) > 0 else 0
    mode_price = df_stable["Price"].mode().iloc[0] if len(df_stable["Price"].mode()) > 0 else 0
    mode_s1 = df_stable["S1 (Marketing)"].mode().iloc[0] if len(df_stable["S1 (Marketing)"].mode()) > 0 else 0
    mode_s2 = df_stable["S2 (Operations)"].mode().iloc[0] if len(df_stable["S2 (Operations)"].mode()) > 0 else 0
    mode_x = df_stable["X (Production)"].mode().iloc[0] if len(df_stable["X (Production)"].mode()) > 0 else 0
    
    print(f"Centralized Optimal:")
    print(f"  p* = {p_opt}, s1* = {s1_opt}, s2* = {s2_opt}")
    print(f"  (x derived from s2 using base-stock policy)")
    print(f"  Optimal Profit/period: {profit_opt:.2f}")
    print()
    
    print(f"Decentralized (learned, stable period):")
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
    
    print("=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    
    return model, df_model, df_agents, {
        'p_opt': p_opt, 's1_opt': s1_opt, 's2_opt': s2_opt,
        'profit_opt': profit_opt, 'cost_opt': cost_opt,
        'avg_profit': avg_profit, 'efficiency': efficiency
    }


if __name__ == "__main__":
    # Run with parameters from params.py
    model, df_model, df_agents, results = run_simulation(rounds=params.ROUNDS)
