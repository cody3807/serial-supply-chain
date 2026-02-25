from datetime import datetime
from model import TwoStageSupplyChainModel
from centralsolver import compute_centralized_optimum
import params

def run_simulation(rounds=None, verbose=True):
    if rounds is None:
        rounds = params.ROUNDS
    
    print("=" * 60)
    print("EPSILON-GREEDY SUPPLY CHAIN MARL SIMULATION")
    print("=" * 60)
    print()
    
    # Compute benchmark
    print("STEP 1: Computing Centralized Benchmark")
    print("-" * 40)
    
    p_opt, s1_opt, s2_opt, profit_opt, cost_opt = compute_centralized_optimum(verbose=True)
    params.CTOT_OPT = cost_opt
    
    print()
    
    # Run simulation
    print("STEP 2: Running Epsilon-Greedy MARL Simulation")
    print("-" * 40)
    print(f"Agents: All using ε-greedy algorithm")
    print(f"Rounds: {rounds}")
    print(f"Learning: ε-greedy with decay ({params.EPS_START} → {params.EPS_END})")
    print()
    
    model = TwoStageSupplyChainModel(
        agent_types=("principal", "greedy_m", "greedy_o")
    )
    
    report_interval = max(1, rounds // 10)
    
    for step in range(rounds):
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
    
    print(f"Epsilon-Greedy Decentralized (learned, stable period):")
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
    
    # Rolling window for smoothing
    window = min(500, rounds // 20)
    
    # Decision variables to plot
    decision_vars = {
        "Beta": ("β (Buy Transfer Price)", "blue", None),
        "Sigma": ("σ (Sell Transfer Price)", "orange", None),
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
    
    plt.suptitle(f"Epsilon-Greedy Decision Variable Convergence (Efficiency: {efficiency:.1f}%)", 
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"epsilon_greedy_convergence_plots_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    print(f"Plots saved to: {plot_filename}")
    
    # Show plot
    plt.show()
    print()
    
    print("=" * 60)
    print("EPSILON-GREEDY SIMULATION COMPLETE")
    print("=" * 60)
    
    return model, df_model, {
        'p_opt': p_opt, 's1_opt': s1_opt, 's2_opt': s2_opt,
        'profit_opt': profit_opt, 'efficiency': efficiency
    }

if __name__ == "__main__":
    model, df_model, results = run_simulation()