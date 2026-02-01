"""
Shadow Price Calculation and Comparison with MARL Transfer Prices

Based on Kouvelis & Lariviere (2000) "Decentralizing Cross-Functional Decisions:
Coordination Through Internal Markets"

KEY THEORETICAL INSIGHTS:
- σ* (sell price to Marketing) = λ(s, Φ) = realized shadow price of inventory
- β* (buy price from Operations) = E[λ] = expected shadow price

The shadow price λ equals the Lagrangian multiplier of the resource constraint
in the centralized optimization. Internal markets coordinate decentralized
agents by having them face these shadow prices.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

import params
from centralsolver import compute_quick_optimum, estimate_avg_profit


# ============================================
# SHADOW PRICE CALCULATION (Kouvelis & Lariviere)
# ============================================

def compute_realized_shadow_price(p, s, demand):
    """
    Compute the REALIZED shadow price λ(s, Φ) for a single period.
    
    This is the Lagrangian multiplier of the constraint: sales ≤ inventory
    
    From K&L (2000), Section 2:
    λ(s, Φ) = ∂R(y*)/∂s
    - If demand < s: Constraint slack → λ = 0 (marginal unit worthless)
    - If demand ≥ s: Constraint binding → λ = p (marginal unit could be sold)
    
    Args:
        p: Market price
        s: Available inventory
        demand: Realized demand Φ
        
    Returns:
        λ: Realized shadow price (marginal value of inventory)
    """
    if demand < s:
        # Inventory exceeds demand - extra unit has no value
        return 0.0
    else:
        # Demand exceeds inventory - scarce!
        # Marginal value = price (we could sell one more at price p)
        return float(p)


def compute_expected_shadow_price(p, s, mu_d, sigma_d):
    """
    Compute the EXPECTED shadow price E[λ] analytically.
    
    From K&L (2000), this is what Operations should be paid (β*)
    since they act BEFORE knowing demand.
    
    E[λ] = E[p × 1{D ≥ s}]
         = p × P(D ≥ s)
         = p × (1 - Φ((s - μ_d) / σ_d))
    
    This is the famous Newsvendor critical fractile formula!
    
    Args:
        p: Market price
        s: Inventory level
        mu_d: Mean demand = a - b*p
        sigma_d: Demand standard deviation
        
    Returns:
        E[λ]: Expected shadow price
    """
    # Standardize
    z = (s - mu_d) / sigma_d
    
    # P(D ≥ s) = 1 - Φ(z)
    prob_stockout = 1 - stats.norm.cdf(z)
    
    # E[λ] = p × P(D ≥ s)
    expected_lambda = p * prob_stockout
    
    return expected_lambda


def compute_shadow_price_monte_carlo(p, s, n_samples=10000, seed=42):
    """
    Estimate shadow prices via Monte Carlo simulation.
    
    Validates the analytical formulas.
    
    Returns:
        E[λ]: Estimated expected shadow price
        std[λ]: Standard deviation
        samples: Array of realized shadow prices
    """
    rng = np.random.default_rng(seed)
    
    mu_d = params.a - params.b * p
    lambdas = []
    
    for _ in range(n_samples):
        # Sample demand
        demand = max(0, rng.normal(mu_d, params.sigma_d))
        
        # Compute realized shadow price
        lam = compute_realized_shadow_price(p, s, demand)
        lambdas.append(lam)
    
    lambdas = np.array(lambdas)
    
    return lambdas.mean(), lambdas.std(), lambdas


# ============================================
# OPTIMAL TRANSFER PRICES (K&L Internal Market)
# ============================================

def compute_optimal_beta(p, s1, s2, verbose=False):
    """
    Compute the OPTIMAL buy price β* from K&L framework.
    
    β* has TWO components:
    
    1. Shadow Price Component (K&L):
       β_shadow = E[λ] = p × P(D ≥ s₁)
       This is the "coordination value" - marginal value of inventory
       
    2. Production Incentive Component:
       With convex cost C(x) = k × x², Operations' FOC is:
       β = C'(x) = 2k × x
       To induce production x = s₂, we need:
       β_incentive = 2k × s₂
       
    The effective optimal β must satisfy BOTH:
       β* = max(β_shadow, β_incentive)
    
    Args:
        p: Market price
        s1: Marketing base-stock level (determines stockout prob)
        s2: Operations base-stock level (determines production)
        
    Returns:
        dict with beta_shadow, beta_incentive, beta_effective
    """
    mu_d = params.a - params.b * p
    sigma_d = params.sigma_d
    
    # Component 1: K&L Shadow Price
    beta_shadow = compute_expected_shadow_price(p, s1, mu_d, sigma_d)
    
    # Component 2: Production Incentive
    # Operations FOC: β = C'(x) = 2k × x
    # To induce x = s2 (target base-stock level), we need:
    beta_incentive = 2 * params.k * s2
    
    # Effective β must be at least the incentive level
    beta_effective = max(beta_shadow, beta_incentive)
    
    if verbose:
        prob_stockout = 1 - stats.norm.cdf((s1 - mu_d) / sigma_d)
        print(f"  Mean demand μ_d = {mu_d:.2f}")
        print(f"  P(stockout) = P(D ≥ {s1}) = {prob_stockout:.4f}")
        print(f"  β_shadow = p × P(stockout) = {p} × {prob_stockout:.4f} = {beta_shadow:.2f}")
        print(f"  β_incentive = 2k×s₂ = 2×{params.k}×{s2} = {beta_incentive:.2f}")
        print(f"  β* = max({beta_shadow:.2f}, {beta_incentive:.2f}) = {beta_effective:.2f}")
    
    return {
        'beta_shadow': beta_shadow,
        'beta_incentive': beta_incentive,
        'beta_effective': beta_effective,
        'prob_stockout': 1 - stats.norm.cdf((s1 - mu_d) / sigma_d)
    }


def compute_optimal_sigma(p, s1, verbose=False):
    """
    Compute the OPTIMAL sell price σ* from K&L framework.
    
    From K&L, the sell price to Marketing should be the realized shadow price:
    σ(s, Φ) = λ(s, Φ) = p if D ≥ s, else 0
    
    However, since σ is set BEFORE demand realization in our model,
    we use the expected shadow price:
    σ* = E[λ] = p × P(D ≥ s₁)
    
    This creates the right incentives for Marketing to order optimally.
    
    Additionally, σ must satisfy: σ < p for Marketing to have positive margin.
    
    Returns:
        dict with sigma_shadow, sigma_effective
    """
    mu_d = params.a - params.b * p
    sigma_d = params.sigma_d
    
    # Expected shadow price
    sigma_shadow = compute_expected_shadow_price(p, s1, mu_d, sigma_d)
    
    # Must be less than p for Marketing to order
    sigma_effective = min(sigma_shadow, p - 1)
    
    if verbose:
        prob_stockout = 1 - stats.norm.cdf((s1 - mu_d) / sigma_d)
        print(f"  σ_shadow = E[λ] = {sigma_shadow:.2f}")
        print(f"  σ* = min(σ_shadow, p-1) = {sigma_effective:.2f}")
    
    return {
        'sigma_shadow': sigma_shadow,
        'sigma_effective': sigma_effective
    }


# ============================================
# NUMERICAL SHADOW PRICES FROM OPTIMIZATION
# ============================================

def compute_numerical_shadow_prices(p_opt, s1_opt, s2_opt, delta=1, seed=42):
    """
    Compute shadow prices numerically by perturbation analysis.
    
    Shadow price = ∂(optimal profit)/∂(constraint relaxation)
    
    For the inventory constraint (sales ≤ inventory):
    λ_s1 ≈ (Profit(s1+δ) - Profit(s1-δ)) / (2δ)
    
    For the production/supply constraint:
    λ_s2 ≈ (Profit(s2+δ) - Profit(s2-δ)) / (2δ)
    
    Returns:
        λ_s1: Shadow price of marketing inventory constraint
        λ_s2: Shadow price of operations inventory constraint
    """
    # Perturb s1 (Marketing's inventory)
    profit_s1_up = estimate_avg_profit(p=p_opt, s1=s1_opt + delta, s2=s2_opt, 
                                        seed=seed, rounds=5000, warmup=500)
    profit_s1_down = estimate_avg_profit(p=p_opt, s1=s1_opt - delta, s2=s2_opt, 
                                          seed=seed, rounds=5000, warmup=500)
    lambda_s1 = (profit_s1_up - profit_s1_down) / (2 * delta)
    
    # Perturb s2 (Operations' inventory)
    profit_s2_up = estimate_avg_profit(p=p_opt, s1=s1_opt, s2=s2_opt + delta, 
                                        seed=seed, rounds=5000, warmup=500)
    profit_s2_down = estimate_avg_profit(p=p_opt, s1=s1_opt, s2=s2_opt - delta, 
                                          seed=seed, rounds=5000, warmup=500)
    lambda_s2 = (profit_s2_up - profit_s2_down) / (2 * delta)
    
    return lambda_s1, lambda_s2


# ============================================
# RUN MARL SIMULATION AND EXTRACT RESULTS
# ============================================

def run_simulation_and_extract_prices(rounds=50000, use_ucb=True, verbose=True):
    """
    Run MARL simulation and extract learned β and σ values.
    
    Returns:
        dict with learned prices and convergence info
    """
    from run_ucb import UCBSupplyChainModel
    from model import TwoStageSupplyChainModel
    
    if verbose:
        print(f"\nRunning {'UCB' if use_ucb else 'ε-greedy'} simulation ({rounds} rounds)...")
    
    # Create model
    if use_ucb:
        model = UCBSupplyChainModel()
    else:
        model = TwoStageSupplyChainModel(
            agent_types=("principal", "greedy_m", "greedy_o")
        )
    
    # Run simulation
    for t in range(rounds):
        model.step()
        if verbose and (t + 1) % (rounds // 5) == 0:
            print(f"  Progress: {t + 1}/{rounds}")
    
    # Get data
    df = model.datacollector.get_model_vars_dataframe()
    
    # Use last 20% as stable period
    stable_start = int(rounds * 0.8)
    df_stable = df.iloc[stable_start:]
    
    # Extract learned values
    learned_beta_mode = df_stable["Beta"].mode().iloc[0]
    learned_sigma_mode = df_stable["Sigma"].mode().iloc[0]
    learned_beta_mean = df_stable["Beta"].mean()
    learned_sigma_mean = df_stable["Sigma"].mean()
    learned_p = df_stable["Price"].mode().iloc[0]
    learned_s1 = df_stable["S1 (Marketing)"].mode().iloc[0]
    learned_s2 = df_stable["S2 (Operations)"].mode().iloc[0]
    
    return {
        'beta_mode': learned_beta_mode,
        'sigma_mode': learned_sigma_mode,
        'beta_mean': learned_beta_mean,
        'sigma_mean': learned_sigma_mean,
        'p': learned_p,
        's1': learned_s1,
        's2': learned_s2,
        'df': df,
        'df_stable': df_stable
    }


# ============================================
# COMPREHENSIVE COMPARISON
# ============================================

def run_shadow_price_comparison(run_simulation=True, simulation_rounds=50000, verbose=True):
    """
    Complete shadow price analysis comparing theory with MARL results.
    
    1. Computes centralized optimal solution
    2. Calculates theoretical shadow prices (K&L formulas)
    3. Computes numerical shadow prices via perturbation
    4. Runs MARL simulation (optional)
    5. Compares and analyzes
    
    Returns:
        Comprehensive results dictionary
    """
    print("=" * 70)
    print("KOUVELIS & LARIVIERE (2000) SHADOW PRICE COMPARISON")
    print("=" * 70)
    print()
    
    # ========================================
    # STEP 1: Centralized Optimal Solution
    # ========================================
    print("STEP 1: Computing Centralized Optimal Solution")
    print("-" * 50)
    
    p_opt, s1_opt, s2_opt, profit_opt, cost_opt = compute_quick_optimum(verbose=True)
    
    print()
    
    # ========================================
    # STEP 2: Theoretical Shadow Prices (K&L)
    # ========================================
    print("STEP 2: Theoretical Shadow Prices (Kouvelis & Lariviere)")
    print("-" * 50)
    
    # Beta calculation
    print("\nOptimal β* (Buy Price to Operations):")
    beta_results = compute_optimal_beta(p_opt, s1_opt, s2_opt, verbose=True)
    
    # Sigma calculation
    print("\nOptimal σ* (Sell Price to Marketing):")
    sigma_results = compute_optimal_sigma(p_opt, s1_opt, verbose=True)
    
    print()
    
    # ========================================
    # STEP 3: Monte Carlo Validation
    # ========================================
    print("STEP 3: Monte Carlo Validation of Shadow Price Formulas")
    print("-" * 50)
    
    mc_mean, mc_std, mc_samples = compute_shadow_price_monte_carlo(p_opt, s1_opt, n_samples=10000)
    
    analytical = compute_expected_shadow_price(
        p_opt, s1_opt, 
        params.a - params.b * p_opt, 
        params.sigma_d
    )
    
    print(f"\nE[λ] (analytical) = {analytical:.4f}")
    print(f"E[λ] (Monte Carlo, n=10000) = {mc_mean:.4f} ± {mc_std:.4f}")
    print(f"Difference = {abs(analytical - mc_mean):.4f} ({abs(analytical - mc_mean)/analytical*100:.2f}%)")
    print("✓ Formula validated!" if abs(analytical - mc_mean) < 1 else "⚠ Check formula!")
    print()
    
    # ========================================
    # STEP 4: Numerical Shadow Prices
    # ========================================
    print("STEP 4: Numerical Shadow Prices (Perturbation Analysis)")
    print("-" * 50)
    
    lambda_s1, lambda_s2 = compute_numerical_shadow_prices(p_opt, s1_opt, s2_opt)
    
    print(f"\nλ_s1 (∂Profit/∂s1) = {lambda_s1:.4f}")
    print(f"λ_s2 (∂Profit/∂s2) = {lambda_s2:.4f}")
    print(f"\nInterpretation:")
    print(f"  - One more unit at Marketing (s1): ${lambda_s1:.2f}/period additional profit")
    print(f"  - One more unit at Operations (s2): ${lambda_s2:.2f}/period additional profit")
    print()
    
    # ========================================
    # STEP 5: MARL Simulation Results
    # ========================================
    simulation_results = None
    
    if run_simulation:
        print("STEP 5: MARL Simulation Results")
        print("-" * 50)
        
        simulation_results = run_simulation_and_extract_prices(
            rounds=simulation_rounds, 
            use_ucb=True, 
            verbose=True
        )
        
        print(f"\nLearned Transfer Prices (stable period mode):")
        print(f"  β (learned) = {simulation_results['beta_mode']}")
        print(f"  σ (learned) = {simulation_results['sigma_mode']}")
        print(f"\nLearned Policy:")
        print(f"  p = {simulation_results['p']}")
        print(f"  s1 = {simulation_results['s1']}")
        print(f"  s2 = {simulation_results['s2']}")
        print()
    
    # ========================================
    # STEP 6: Comparison and Analysis
    # ========================================
    print("=" * 70)
    print("COMPARISON: THEORETICAL vs. LEARNED TRANSFER PRICES")
    print("=" * 70)
    print()
    
    print(f"{'Metric':<35} {'Theoretical':>15} {'Learned':>15} {'Gap':>15}")
    print("-" * 80)
    
    # β comparison
    beta_theoretical = beta_results['beta_effective']
    print(f"{'β* (Buy Price)':<35} {beta_theoretical:>15.2f}", end="")
    
    if simulation_results:
        beta_learned = simulation_results['beta_mode']
        beta_gap = abs(beta_theoretical - beta_learned)
        print(f" {beta_learned:>15.2f} {beta_gap:>15.2f}")
    else:
        print()
    
    # σ comparison
    sigma_theoretical = sigma_results['sigma_effective']
    print(f"{'σ* (Sell Price)':<35} {sigma_theoretical:>15.2f}", end="")
    
    if simulation_results:
        sigma_learned = simulation_results['sigma_mode']
        sigma_gap = abs(sigma_theoretical - sigma_learned)
        print(f" {sigma_learned:>15.2f} {sigma_gap:>15.2f}")
    else:
        print()
    
    # Component breakdown
    print()
    print(f"{'β Components:':<35}")
    print(f"{'  Shadow Price β_shadow':<35} {beta_results['beta_shadow']:>15.2f}")
    print(f"{'  Production Incentive β_incentive':<35} {beta_results['beta_incentive']:>15.2f}")
    print(f"{'  P(stockout)':<35} {beta_results['prob_stockout']:>15.4f}")
    
    print()
    print(f"{'Numerical Shadow Prices:':<35}")
    print(f"{'  λ_s1 (Marketing)':<35} {lambda_s1:>15.4f}")
    print(f"{'  λ_s2 (Operations)':<35} {lambda_s2:>15.4f}")
    
    # ========================================
    # STEP 7: K&L Interpretation
    # ========================================
    print()
    print("=" * 70)
    print("INTERPRETATION (Kouvelis & Lariviere Framework)")
    print("=" * 70)
    print()
    
    print("THEORETICAL INSIGHT:")
    print("-" * 50)
    print("""
From K&L (2000), Section 2-3:

1. SHADOW PRICE λ(s, Φ):
   - Equals Lagrangian multiplier of resource constraint
   - λ = p if demand ≥ supply (scarcity), else 0
   
2. BUY PRICE β*:
   - Should equal EXPECTED shadow price: β* = E[λ]
   - In our model: β* = max(p × P(D ≥ s₁), 2k × s₂)
   - First term = coordination value (K&L)
   - Second term = production cost coverage
   
3. SELL PRICE σ*:
   - Should equal shadow price for Marketing's constraint
   - σ* = E[λ] ensures Marketing orders optimally
   
4. DECENTRALIZATION PRINCIPLE:
   - If β = β* and σ = σ*, decentralized = centralized
   - "The market maker's job is to allow price to adapt
      to the environment, not to turn a profit."
""")
    
    if simulation_results:
        print("YOUR SIMULATION RESULTS:")
        print("-" * 50)
        
        # Analyze alignment
        beta_alignment = 100 * (1 - abs(beta_theoretical - simulation_results['beta_mode']) / 
                                max(beta_theoretical, 1))
        sigma_alignment = 100 * (1 - abs(sigma_theoretical - simulation_results['sigma_mode']) / 
                                 max(sigma_theoretical, 1))
        
        print(f"\nβ alignment with theory: {beta_alignment:.1f}%")
        print(f"σ alignment with theory: {sigma_alignment:.1f}%")
        
        print(f"\nFindings:")
        
        if beta_alignment > 80:
            print(f"  ✓ β closely matches theoretical prediction")
            print(f"    Agents learned the shadow price + production cost tradeoff")
        elif simulation_results['beta_mode'] > beta_theoretical:
            print(f"  ⚠ β > β*: Agents over-compensate Operations")
            print(f"    This may cause over-production and excess inventory")
        else:
            print(f"  ⚠ β < β*: Agents under-compensate Operations")
            print(f"    This may cause under-production and stockouts")
        
        if sigma_alignment > 80:
            print(f"  ✓ σ closely matches theoretical prediction")
        elif simulation_results['sigma_mode'] > sigma_theoretical:
            print(f"  ⚠ σ > σ*: Marketing faces high costs, may under-order")
        else:
            print(f"  ⚠ σ < σ*: Marketing faces low costs, may over-order")
        
        # Efficiency analysis
        df_stable = simulation_results['df_stable']
        avg_profit = -df_stable['Total Cost'].mean()
        efficiency = avg_profit / profit_opt * 100
        
        print(f"\n  System Efficiency: {efficiency:.1f}%")
        print(f"  (Decentralized profit / Centralized optimal)")
        
        if efficiency > 95:
            print(f"  ✓ Near-optimal coordination achieved!")
        elif efficiency > 85:
            print(f"  ~ Moderate coordination - some efficiency loss")
        else:
            print(f"  ⚠ Significant coordination failure")
            print(f"    This is the 'price of anarchy' from decentralization")
    
    # Return comprehensive results
    results = {
        'centralized': {
            'p_opt': p_opt,
            's1_opt': s1_opt,
            's2_opt': s2_opt,
            'profit_opt': profit_opt
        },
        'theoretical': {
            'beta_shadow': beta_results['beta_shadow'],
            'beta_incentive': beta_results['beta_incentive'],
            'beta_effective': beta_results['beta_effective'],
            'sigma_effective': sigma_results['sigma_effective'],
            'prob_stockout': beta_results['prob_stockout']
        },
        'numerical': {
            'lambda_s1': lambda_s1,
            'lambda_s2': lambda_s2
        },
        'monte_carlo': {
            'mean': mc_mean,
            'std': mc_std
        }
    }
    
    if simulation_results:
        results['simulation'] = {
            'beta': simulation_results['beta_mode'],
            'sigma': simulation_results['sigma_mode'],
            'beta_alignment': beta_alignment,
            'sigma_alignment': sigma_alignment,
            'efficiency': efficiency
        }
    
    return results


def plot_shadow_price_analysis(results, save=True):
    """
    Generate visualization of shadow price comparison.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Transfer Price Comparison
    ax1 = axes[0, 0]
    
    labels = ['β (Buy)', 'σ (Sell)']
    theoretical = [results['theoretical']['beta_effective'], 
                   results['theoretical']['sigma_effective']]
    learned = [results['simulation']['beta'], 
               results['simulation']['sigma']] if 'simulation' in results else [0, 0]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, theoretical, width, label='Theoretical (K&L)', color='blue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, learned, width, label='Learned (MARL)', color='orange', alpha=0.7)
    
    ax1.set_ylabel('Price')
    ax1.set_title('Transfer Prices: Theoretical vs. Learned')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # Plot 2: β Components
    ax2 = axes[0, 1]
    
    components = ['β_shadow\n(K&L)', 'β_incentive\n(Production)', 'β_effective\n(Max)']
    values = [results['theoretical']['beta_shadow'],
              results['theoretical']['beta_incentive'],
              results['theoretical']['beta_effective']]
    
    colors = ['lightblue', 'lightgreen', 'blue']
    bars = ax2.bar(components, values, color=colors, edgecolor='black')
    
    ax2.set_ylabel('Price')
    ax2.set_title('β* Decomposition (Kouvelis & Lariviere)')
    ax2.grid(True, alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # Plot 3: Numerical Shadow Prices
    ax3 = axes[1, 0]
    
    numerical_labels = ['λ_s1\n(Marketing)', 'λ_s2\n(Operations)']
    numerical_values = [results['numerical']['lambda_s1'], results['numerical']['lambda_s2']]
    
    colors = ['red', 'purple']
    bars = ax3.bar(numerical_labels, numerical_values, color=colors, alpha=0.7, edgecolor='black')
    
    ax3.set_ylabel('Shadow Price (∂Profit/∂s)')
    ax3.set_title('Numerical Shadow Prices (Perturbation Analysis)')
    ax3.grid(True, alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # Plot 4: Summary Text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary = f"""
    KOUVELIS & LARIVIERE (2000) SHADOW PRICE ANALYSIS
    {'=' * 50}
    
    Centralized Optimal:
        p* = {results['centralized']['p_opt']}
        s1* = {results['centralized']['s1_opt']}
        s2* = {results['centralized']['s2_opt']}
        Profit* = ${results['centralized']['profit_opt']:.2f}
    
    Theoretical Transfer Prices (K&L):
        β* = {results['theoretical']['beta_effective']:.2f}
            (shadow: {results['theoretical']['beta_shadow']:.2f}, 
             incentive: {results['theoretical']['beta_incentive']:.2f})
        σ* = {results['theoretical']['sigma_effective']:.2f}
        P(stockout) = {results['theoretical']['prob_stockout']:.4f}
    """
    
    if 'simulation' in results:
        summary += f"""
    Learned Transfer Prices (MARL):
        β = {results['simulation']['beta']:.2f}
        σ = {results['simulation']['sigma']:.2f}
    
    Alignment:
        β alignment: {results['simulation']['beta_alignment']:.1f}%
        σ alignment: {results['simulation']['sigma_alignment']:.1f}%
        System efficiency: {results['simulation']['efficiency']:.1f}%
    """
    
    summary += f"""
    Key Insight:
        Internal markets coordinate decentralized agents
        by pricing resources at their shadow values.
        β* = E[λ] ensures Operations produces optimally.
        σ* = E[λ] ensures Marketing orders optimally.
    """
    
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Shadow Price Analysis: Kouvelis & Lariviere Framework', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"shadow_price_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {filename}")
    
    plt.show()
    
    return fig


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Run full comparison with simulation
    results = run_shadow_price_comparison(
        run_simulation=True,
        simulation_rounds=50000,  # Enough for convergence
        verbose=True
    )
    
    # Generate plots
    plot_shadow_price_analysis(results, save=True)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
