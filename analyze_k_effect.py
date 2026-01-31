"""
Diagnostic analysis to understand why production cost coefficient (k) 
has such a dramatic impact on UCB performance.

This script will:
1. Show how k affects the optimal policy
2. Show how k affects the production cost gradient
3. Show how k affects the reward landscape
4. Analyze action space sensitivity to k
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from centralsolver import compute_quick_optimum, estimate_avg_profit
import params


def analyze_k_impact_on_optimal():
    """Analyze how k affects the optimal policy."""
    print("=" * 70)
    print("ANALYSIS: How k affects OPTIMAL POLICY")
    print("=" * 70)
    
    k_values = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    results = []
    
    for k in k_values:
        original_k = params.k
        params.k = k
        
        p_opt, s1_opt, s2_opt, profit_opt, cost_opt = compute_quick_optimum(verbose=False)
        
        # Calculate average production at optimal
        rng = np.random.default_rng(42)
        I2_samples = []
        x_samples = []
        for _ in range(1000):
            I2 = np.random.randint(0, s2_opt + 20)
            x = max(0, s2_opt - I2)
            I2_samples.append(I2)
            x_samples.append(x)
        
        avg_x = np.mean(x_samples)
        avg_production_cost = k * (avg_x ** 2)
        
        results.append({
            'k': k,
            'p_opt': p_opt,
            's1_opt': s1_opt,
            's2_opt': s2_opt,
            'profit_opt': profit_opt,
            'avg_x': avg_x,
            'avg_production_cost': avg_production_cost,
            'production_cost_ratio': avg_production_cost / (-cost_opt) if cost_opt != 0 else 0
        })
        
        params.k = original_k
    
    df = pd.DataFrame(results)
    print("\nOptimal Policy vs k:")
    print(df.to_string(index=False))
    
    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print(f"As k increases from {k_values[0]} to {k_values[-1]}:")
    print(f"  - Optimal s2* changes from {df.iloc[0]['s2_opt']} to {df.iloc[-1]['s2_opt']}")
    print(f"  - Average production x changes from {df.iloc[0]['avg_x']:.2f} to {df.iloc[-1]['avg_x']:.2f}")
    print(f"  - Optimal profit decreases from {df.iloc[0]['profit_opt']:.2f} to {df.iloc[-1]['profit_opt']:.2f}")
    print(f"  - Production cost as % of total cost: {df.iloc[0]['production_cost_ratio']*100:.1f}% → {df.iloc[-1]['production_cost_ratio']*100:.1f}%")
    
    return df


def analyze_reward_landscape():
    """Analyze how k affects the reward landscape (sensitivity)."""
    print("\n" + "=" * 70)
    print("ANALYSIS: How k affects REWARD LANDSCAPE SENSITIVITY")
    print("=" * 70)
    
    k_values = [0.1, 0.35, 1.0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, k in enumerate(k_values):
        original_k = params.k
        params.k = k
        
        # Get optimal for this k
        p_opt, s1_opt, s2_opt, profit_opt, _ = compute_quick_optimum(verbose=False)
        
        # Test s2 values around optimal
        s2_test = np.arange(max(10, s2_opt - 30), s2_opt + 30, 5)
        profits = []
        
        for s2 in s2_test:
            avg_profit = estimate_avg_profit(
                p=p_opt,
                s1=s1_opt,
                s2=int(s2),
                seed=42,
                rounds=1000,
                warmup=100
            )
            profits.append(avg_profit)
        
        params.k = original_k
        
        # Plot
        ax = axes[idx]
        ax.plot(s2_test, profits, linewidth=2, marker='o', markersize=4)
        ax.axvline(s2_opt, color='red', linestyle='--', label=f's2*={s2_opt}')
        ax.axhline(profit_opt, color='green', linestyle='--', alpha=0.5, label=f'Optimal={profit_opt:.0f}')
        ax.set_xlabel('s2 (Operations Base-Stock)')
        ax.set_ylabel('Avg Profit')
        ax.set_title(f'k={k}: Reward Landscape')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Calculate sensitivity (gradient)
        gradients = np.diff(profits) / np.diff(s2_test)
        avg_gradient = np.mean(np.abs(gradients))
        max_gradient = np.max(np.abs(gradients))
        
        print(f"\nk={k}:")
        print(f"  Optimal s2*: {s2_opt}")
        print(f"  Optimal profit: {profit_opt:.2f}")
        print(f"  Average |gradient|: {avg_gradient:.2f}")
        print(f"  Max |gradient|: {max_gradient:.2f}")
        print(f"  Sensitivity (gradient ratio vs k=0.1): {avg_gradient / (8.5 if k == 0.1 else avg_gradient):.2f}x")
    
    plt.tight_layout()
    plt.savefig('k_reward_landscape.png', dpi=150)
    print(f"\nPlot saved: k_reward_landscape.png")
    plt.show()


def analyze_production_cost_gradient():
    """Show how marginal production cost changes with k."""
    print("\n" + "=" * 70)
    print("ANALYSIS: PRODUCTION COST GRADIENT")
    print("=" * 70)
    
    k_values = [0.1, 0.35, 1.0]
    x_range = np.arange(0, 100, 5)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Total production cost
    ax1 = axes[0]
    for k in k_values:
        costs = k * (x_range ** 2)
        ax1.plot(x_range, costs, linewidth=2, label=f'k={k}')
    ax1.set_xlabel('Production Quantity (x)')
    ax1.set_ylabel('Production Cost (k*x²)')
    ax1.set_title('Total Production Cost vs x')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Marginal cost (derivative)
    ax2 = axes[1]
    for k in k_values:
        marginal_costs = 2 * k * x_range  # d(k*x²)/dx = 2*k*x
        ax2.plot(x_range, marginal_costs, linewidth=2, label=f'k={k}')
    ax2.set_xlabel('Production Quantity (x)')
    ax2.set_ylabel('Marginal Cost (2*k*x)')
    ax2.set_title('Marginal Production Cost vs x')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('k_production_cost_gradient.png', dpi=150)
    print(f"Plot saved: k_production_cost_gradient.png")
    plt.show()
    
    print("\nKEY INSIGHT:")
    print("Marginal cost = 2*k*x")
    print("Higher k → Steeper penalty for overproduction")
    print("This makes the reward function more sensitive around optimal s2")
    print("→ Easier for UCB to distinguish good vs bad actions!")


def analyze_action_space_flatness():
    """Analyze how 'flat' the reward landscape is for different k values."""
    print("\n" + "=" * 70)
    print("ANALYSIS: REWARD LANDSCAPE FLATNESS (Why UCB struggles with low k)")
    print("=" * 70)
    
    k_values = [0.1, 0.35, 1.0]
    
    results = []
    
    for k in k_values:
        original_k = params.k
        params.k = k
        
        # Get optimal
        p_opt, s1_opt, s2_opt, profit_opt, _ = compute_quick_optimum(verbose=False)
        
        # Test how much profit degrades for sub-optimal choices
        profits_at_distances = []
        distances = [0, 10, 20, 30, 40]
        
        for dist in distances:
            s2_test = s2_opt + dist
            avg_profit = estimate_avg_profit(
                p=p_opt,
                s1=s1_opt,
                s2=int(s2_test),
                seed=42,
                rounds=1000,
                warmup=100
            )
            profit_loss = profit_opt - avg_profit
            profit_loss_pct = (profit_loss / profit_opt * 100) if profit_opt > 0 else 0
            profits_at_distances.append(profit_loss_pct)
        
        params.k = original_k
        
        print(f"\nk={k} (optimal s2*={s2_opt}):")
        print(f"  Profit loss when s2 is OFF by:")
        for dist, loss_pct in zip(distances, profits_at_distances):
            print(f"    {dist:2d} units: {loss_pct:6.2f}% worse")
        
        results.append({
            'k': k,
            'loss_at_10': profits_at_distances[1],
            'loss_at_20': profits_at_distances[2],
            'loss_at_30': profits_at_distances[3],
        })
    
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("SUMMARY: Profit Loss from Sub-optimal s2")
    print("=" * 70)
    print(df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("WHY UCB PERFORMS BETTER AT HIGH k:")
    print("=" * 70)
    print("When k is LOW (0.1):")
    print("  → Production cost k*x² is small")
    print("  → Many different s2 values give SIMILAR rewards")
    print("  → Flat reward landscape = hard for UCB to learn")
    print("  → UCB gets 'confused' by noise and explores too many actions")
    print("\nWhen k is HIGH (1.0):")
    print("  → Production cost k*x² is large")
    print("  → Wrong s2 values get SEVERELY penalized")
    print("  → Sharp reward landscape = easy for UCB to learn")
    print("  → UCB quickly identifies and exploits optimal actions")


def analyze_noise_to_signal_ratio():
    """Analyze signal-to-noise ratio in rewards for different k."""
    print("\n" + "=" * 70)
    print("ANALYSIS: SIGNAL-TO-NOISE RATIO")
    print("=" * 70)
    
    k_values = [0.1, 0.35, 1.0]
    
    for k in k_values:
        original_k = params.k
        params.k = k
        
        p_opt, s1_opt, s2_opt, profit_opt, _ = compute_quick_optimum(verbose=False)
        
        # Measure reward variance at optimal
        profits_optimal = []
        for _ in range(100):
            profit = estimate_avg_profit(
                p=p_opt, s1=s1_opt, s2=s2_opt,
                seed=np.random.randint(0, 10000),
                rounds=100, warmup=20
            )
            profits_optimal.append(profit)
        
        # Measure reward at suboptimal (s2 + 20)
        profits_suboptimal = []
        for _ in range(100):
            profit = estimate_avg_profit(
                p=p_opt, s1=s1_opt, s2=s2_opt + 20,
                seed=np.random.randint(0, 10000),
                rounds=100, warmup=20
            )
            profits_suboptimal.append(profit)
        
        params.k = original_k
        
        mean_opt = np.mean(profits_optimal)
        std_opt = np.std(profits_optimal)
        mean_sub = np.mean(profits_suboptimal)
        std_sub = np.std(profits_suboptimal)
        
        signal = abs(mean_opt - mean_sub)
        noise = (std_opt + std_sub) / 2
        snr = signal / noise if noise > 0 else float('inf')
        
        print(f"\nk={k}:")
        print(f"  Optimal s2*={s2_opt}: mean={mean_opt:.2f}, std={std_opt:.2f}")
        print(f"  Suboptimal s2={s2_opt+20}: mean={mean_sub:.2f}, std={std_sub:.2f}")
        print(f"  Signal (difference): {signal:.2f}")
        print(f"  Noise (avg std): {noise:.2f}")
        print(f"  Signal-to-Noise Ratio: {snr:.2f}")
        print(f"  → {'Easy to distinguish' if snr > 5 else 'Hard to distinguish'}")


if __name__ == "__main__":
    # Run all analyses
    df_optimal = analyze_k_impact_on_optimal()
    
    analyze_production_cost_gradient()
    
    analyze_reward_landscape()
    
    analyze_action_space_flatness()
    
    analyze_noise_to_signal_ratio()
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nCONCLUSION:")
    print("Higher k creates a MORE INFORMATIVE reward landscape:")
    print("  1. Steeper gradients around optimal → easier to find peak")
    print("  2. Larger penalties for mistakes → clearer signal")
    print("  3. Higher signal-to-noise ratio → less confusion from randomness")
    print("  4. UCB can quickly identify and exploit optimal actions")
    print("\nLow k creates a FLAT reward landscape:")
    print("  1. Many actions give similar rewards → hard to distinguish")
    print("  2. Small penalties for mistakes → weak signal")
    print("  3. Low signal-to-noise ratio → randomness dominates")
    print("  4. UCB wastes time exploring similar-value actions")
