"""
Kouvelis and Lariviere (2000) Shadow Price Validator

This module implements the theoretical shadow price calculations from the paper
and compares them against the prices LEARNED by the MARL agents.

KEY INSIGHT:
- Learning agents (UCB, ε-greedy) do NOT know these formulas
- They learn by trial and error
- This validator calculates what prices SHOULD be (ground truth)
- We compare learned vs. calculated to validate learning

SHADOW PRICE THEORY:
1. Marketing Shadow Price λ(s, Φ):
   - The marginal value of one more unit of inventory
   - λ = 0 if inventory > demand (no scarcity)
   - λ = marginal revenue if demand ≥ inventory (scarcity)

2. Operations Expected Shadow Price β*:
   - β* = E[λ] = Expected value of the shadow price
   - Operations acts BEFORE seeing demand, so uses expectations
   
COORDINATION:
- Optimal σ (sell price) = E[λ | realized state] = Realized Shadow Price
- Optimal β (buy price) = E[λ] = Expected Shadow Price
- When Principal sets (β*, σ*), decentralized decisions = centralized optimum
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

import params
from centralsolver import compute_quick_optimum
from run_ucb import UCBSupplyChainModel
from model import TwoStageSupplyChainModel


# ============================================
# THEORETICAL SHADOW PRICE CALCULATIONS
# ============================================

def compute_realized_shadow_price(p, s, demand):
    """
    Compute the REALIZED shadow price λ(s, Φ) for a single period.
    
    This is what Marketing experiences AFTER seeing actual inventory and demand.
    
    Args:
        p: Market price
        s: Available inventory (stock)
        demand: Realized demand Φ
    
    Returns:
        λ: Realized shadow price (marginal value of inventory)
    
    Theory (Kouvelis & Lariviere):
        λ(s, Φ) = ∂R(y*)/∂s
        - If demand < s: Constraint y ≤ s is slack → λ = 0
        - If demand ≥ s: Constraint is binding → λ = p (marginal revenue)
    """
    if demand < s:
        # Inventory exceeds demand - extra unit has no value
        # (or salvage value if we had one)
        return 0.0
    else:
        # Demand exceeds inventory - scarce!
        # Marginal value = price (we could sell one more at price p)
        return float(p)


def compute_expected_shadow_price(p, s, mu_d, sigma_d):
    """
    Compute the EXPECTED shadow price E[λ] analytically.
    
    This is what Operations should be paid (β*) since they act
    BEFORE knowing demand.
    
    Args:
        p: Market price
        s: Expected inventory level
        mu_d: Mean demand = a - b*p
        sigma_d: Demand standard deviation
    
    Returns:
        E[λ]: Expected shadow price
    
    Theory:
        E[λ] = p × P(D ≥ s) + 0 × P(D < s)
             = p × P(D ≥ s)
             = p × (1 - Φ((s - μ_d) / σ_d))
        
        where Φ is the standard normal CDF.
    """
    # Standardize
    z = (s - mu_d) / sigma_d
    
    # P(D ≥ s) = 1 - Φ(z)
    prob_stockout = 1 - stats.norm.cdf(z)
    
    # E[λ] = p × P(D ≥ s)
    expected_lambda = p * prob_stockout
    
    return expected_lambda


def compute_optimal_transfer_prices(p, s1, s2, verbose=False):
    """
    Compute the OPTIMAL transfer prices (β*, σ*) for given policy.
    
    IMPORTANT: In our model with convex production cost C(x) = k*x²,
    the transfer prices serve TWO purposes:
    
    1. K&L Shadow Price Component: Value of inventory coordination
       β_shadow = E[λ] = p × P(D ≥ s)
       
    2. Production Incentive Component: Must cover marginal production cost
       β_incentive = C'(x) = 2*k*x for Operations to produce x units
    
    The EFFECTIVE optimal β must satisfy BOTH:
       β* = max(β_shadow, β_incentive)
    
    Or in a Nash equilibrium, β is set so that Operations' FOC holds:
       β = 2*k*x  →  x = β/(2k)
    
    Args:
        p: Market price
        s1: Marketing base-stock level (inventory available for sales)
        s2: Operations base-stock level
    
    Returns:
        β_shadow: Pure shadow price component (K&L)
        β_incentive: Production incentive component
        β_effective: Effective required β
        σ*: Optimal sell price
    """
    # Mean demand at price p
    mu_d = params.a - params.b * p
    sigma_d = params.sigma_d
    
    # K&L Shadow Price: Expected marginal value of inventory
    beta_shadow = compute_expected_shadow_price(p, s1, mu_d, sigma_d)
    
    # Production Incentive: What β is needed to induce production x = s2?
    # Operations FOC: β = C'(x) = 2*k*x
    # If we want x = s2 (base-stock), we need:
    x_target = s2  # Target production level
    beta_incentive = 2 * params.k * x_target
    
    # Effective β must be at least the incentive level
    beta_effective = max(beta_shadow, beta_incentive)
    
    # Sigma: In our model, σ affects Marketing's ordering incentive
    # Optimal σ is such that Marketing orders optimally given p
    # If σ is too high, Marketing won't order; if too low, over-orders
    # Rule: σ < p for positive margin
    sigma_star = min(beta_shadow + params.H1, p - 1)  # Simple heuristic
    
    if verbose:
        prob_stockout = 1 - stats.norm.cdf((s1 - mu_d) / sigma_d)
        print(f"  Mean demand μ_d = {mu_d:.2f}")
        print(f"  P(stockout) = P(D ≥ {s1}) = {prob_stockout:.4f}")
        print(f"  Shadow Price β_shadow = p × P(stockout) = {p} × {prob_stockout:.4f} = {beta_shadow:.2f}")
        print(f"  Production Incentive β_incentive = 2k×s2 = 2×{params.k}×{s2} = {beta_incentive:.2f}")
        print(f"  Effective β* = max({beta_shadow:.2f}, {beta_incentive:.2f}) = {beta_effective:.2f}")
    
    return beta_shadow, beta_incentive, beta_effective, sigma_star


def compute_optimal_production_given_beta(beta, k):
    """
    Compute optimal production x* given buy price β.
    
    Operations Problem:
        max β × x - C(x)
        max β × x - k × x²
        
    FOC: β = C'(x) = 2kx
    Solution: x* = β / (2k)
    
    Args:
        beta: Buy price from Principal
        k: Production cost coefficient
        
    Returns:
        x*: Optimal production quantity
    """
    if k <= 0:
        return float('inf')  # No cost, infinite production
    
    x_star = beta / (2 * k)
    return max(0, x_star)


def compute_optimal_order_given_sigma(sigma, p, s1, I1):
    """
    Compute optimal order y* given sell price σ.
    
    Marketing Problem:
        max p × sales - σ × order - holding costs
        
    If σ < p (margin positive): Order up to base-stock s1
    If σ ≥ p (margin negative): Don't order
    
    Args:
        sigma: Sell price to Principal
        p: Market price
        s1: Base-stock level
        I1: Current inventory
        
    Returns:
        y*: Optimal order quantity
    """
    if sigma >= p:
        # Negative margin, don't order
        return 0
    else:
        # Positive margin, order up to base-stock
        return max(0, s1 - I1)


# ============================================
# MONTE CARLO SHADOW PRICE ESTIMATION
# ============================================

def estimate_shadow_prices_monte_carlo(p, s1, n_samples=10000, seed=42):
    """
    Estimate shadow prices via Monte Carlo simulation.
    
    This validates the analytical formulas by simulation.
    
    Returns:
        E[λ]: Estimated expected shadow price
        std[λ]: Standard deviation of shadow price
        samples: Array of realized shadow prices
    """
    rng = np.random.default_rng(seed)
    
    mu_d = params.a - params.b * p
    lambdas = []
    
    for _ in range(n_samples):
        # Sample demand
        demand = max(0, rng.normal(mu_d, params.sigma_d))
        
        # Compute realized shadow price
        lam = compute_realized_shadow_price(p, s1, demand)
        lambdas.append(lam)
    
    lambdas = np.array(lambdas)
    
    return lambdas.mean(), lambdas.std(), lambdas


# ============================================
# FIRST-BEST (CENTRALIZED) BENCHMARK
# ============================================

def compute_first_best_solution(verbose=True):
    """
    Compute the First-Best (centralized) solution.
    
    The centralized planner maximizes total system profit:
        max E[p × min(D, s) - k × x² - H1 × I1 - H2 × I2 - π × backorders]
        
    Returns:
        dict with p*, s1*, s2*, β_shadow, β_incentive, β_effective, σ*, profit*
    """
    if verbose:
        print("\n" + "=" * 70)
        print("FIRST-BEST (CENTRALIZED) SOLUTION")
        print("=" * 70)
    
    # Get centralized optimum from existing solver
    p_opt, s1_opt, s2_opt, profit_opt, cost_opt = compute_quick_optimum(verbose=False)
    
    # Compute corresponding optimal transfer prices
    beta_shadow, beta_incentive, beta_effective, sigma_star = compute_optimal_transfer_prices(
        p_opt, s1_opt, s2_opt, verbose=verbose
    )
    
    if verbose:
        print(f"\n  Optimal Policy:")
        print(f"    p* = {p_opt}")
        print(f"    s1* = {s1_opt}")
        print(f"    s2* = {s2_opt}")
        print(f"  Transfer Price Analysis:")
        print(f"    β_shadow (K&L) = {beta_shadow:.2f} (pure coordination value)")
        print(f"    β_incentive = {beta_incentive:.2f} (production cost coverage)")
        print(f"    β_effective = {beta_effective:.2f} (what agents should learn)")
        print(f"    σ* = {sigma_star:.2f}")
        print(f"  First-Best Profit: {profit_opt:.2f}")
    
    return {
        'p_opt': p_opt,
        's1_opt': s1_opt,
        's2_opt': s2_opt,
        'beta_shadow': beta_shadow,
        'beta_incentive': beta_incentive,
        'beta_effective': beta_effective,
        'sigma_star': sigma_star,
        'profit_opt': profit_opt
    }


# ============================================
# LEARNING vs. THEORETICAL COMPARISON
# ============================================

class ShadowPriceValidator:
    """
    Validates that learning agents converge to theoretical shadow prices.
    
    Usage:
        validator = ShadowPriceValidator()
        validator.run_comparison(model, df_model)
        validator.plot_convergence()
    """
    
    def __init__(self):
        # Compute first-best solution
        fb = compute_first_best_solution(verbose=False)
        self.p_opt = fb['p_opt']
        self.s1_opt = fb['s1_opt']
        self.s2_opt = fb['s2_opt']
        self.beta_shadow = fb['beta_shadow']
        self.beta_incentive = fb['beta_incentive']
        self.beta_star = fb['beta_effective']  # What agents should learn
        self.sigma_star = fb['sigma_star']
        self.profit_opt = fb['profit_opt']
        
        # Storage for period-by-period comparison
        self.periods = []
        self.learned_betas = []
        self.learned_sigmas = []
        self.theoretical_betas = []
        self.theoretical_sigmas = []
        self.realized_lambdas = []
    
    def compute_theoretical_for_period(self, p, s1, s2, demand):
        """
        Compute theoretical shadow prices for a single period.
        """
        # Realized shadow price (what σ SHOULD be this period)
        lambda_realized = compute_realized_shadow_price(p, s1, demand)
        
        # Expected shadow price (K&L component)
        mu_d = params.a - params.b * p
        beta_shadow = compute_expected_shadow_price(p, s1, mu_d, params.sigma_d)
        
        # Production incentive component
        beta_incentive = 2 * params.k * s2
        
        # Effective theoretical beta
        beta_theoretical = max(beta_shadow, beta_incentive)
        
        return beta_theoretical, lambda_realized
    
    def validate_period(self, t, learned_beta, learned_sigma, p, s1, s2, demand):
        """
        Compare learned vs theoretical for a single period.
        """
        beta_theo, lambda_realized = self.compute_theoretical_for_period(p, s1, s2, demand)
        
        self.periods.append(t)
        self.learned_betas.append(learned_beta)
        self.learned_sigmas.append(learned_sigma)
        self.theoretical_betas.append(beta_theo)
        self.realized_lambdas.append(lambda_realized)
        # For sigma, theoretical is the realized lambda (varies each period)
        self.theoretical_sigmas.append(lambda_realized)
    
    def run_learning_simulation(self, rounds=5000, use_ucb=True, verbose=True):
        """
        Run a learning simulation and collect data for validation.
        """
        if verbose:
            print("\n" + "=" * 70)
            print(f"RUNNING {'UCB' if use_ucb else 'EPSILON-GREEDY'} LEARNING SIMULATION")
            print("=" * 70)
        
        # Create model
        if use_ucb:
            model = UCBSupplyChainModel()
        else:
            model = TwoStageSupplyChainModel(
                agent_types=("principal", "greedy_m", "greedy_o")
            )
        
        # Run simulation and collect data
        for t in range(rounds):
            model.step()
            
            # Extract values after step
            self.validate_period(
                t=t,
                learned_beta=model.beta,
                learned_sigma=model.sigma,
                p=model.p,
                s1=model.s1,
                s2=model.s2,
                demand=model.demand
            )
            
            if verbose and (t + 1) % (rounds // 10) == 0:
                print(f"  Progress: {t + 1}/{rounds}")
        
        # Get final dataframe
        df_model = model.datacollector.get_model_vars_dataframe()
        
        return model, df_model
    
    def compute_convergence_metrics(self, window=500):
        """
        Compute metrics showing if learning converges to theory.
        """
        n = len(self.periods)
        if n < window:
            window = n // 2
        
        # Early period averages
        early_beta_error = np.mean(np.abs(
            np.array(self.learned_betas[:window]) - np.array(self.theoretical_betas[:window])
        ))
        
        # Late period averages
        late_beta_error = np.mean(np.abs(
            np.array(self.learned_betas[-window:]) - np.array(self.theoretical_betas[-window:])
        ))
        
        # Rolling average of learned beta (last 20%)
        late_learned_beta = np.mean(self.learned_betas[-window:])
        late_theoretical_beta = np.mean(self.theoretical_betas[-window:])
        
        return {
            'early_beta_error': early_beta_error,
            'late_beta_error': late_beta_error,
            'late_learned_beta': late_learned_beta,
            'late_theoretical_beta': late_theoretical_beta,
            'beta_convergence_ratio': late_beta_error / max(early_beta_error, 0.01),
            'beta_gap': abs(late_learned_beta - late_theoretical_beta),
            'first_best_beta': self.beta_star,
        }
    
    def plot_convergence(self, save=True):
        """
        Plot learning convergence toward theoretical shadow prices.
        """
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        
        periods = np.array(self.periods)
        window = max(100, len(periods) // 50)
        
        # ========================================
        # Plot 1: Learned β vs Theoretical β*
        # ========================================
        ax1 = axes[0, 0]
        
        # Raw learned beta (faint)
        ax1.plot(periods, self.learned_betas, alpha=0.2, color='blue', linewidth=0.5)
        
        # Rolling average of learned beta
        learned_beta_rolling = pd.Series(self.learned_betas).rolling(window=window, min_periods=1).mean()
        ax1.plot(periods, learned_beta_rolling, color='blue', linewidth=2, label='Learned β (rolling avg)')
        
        # Rolling average of theoretical beta
        theo_beta_rolling = pd.Series(self.theoretical_betas).rolling(window=window, min_periods=1).mean()
        ax1.plot(periods, theo_beta_rolling, color='green', linewidth=2, linestyle='--', 
                label='Theoretical E[λ] (rolling avg)')
        
        # First-best beta
        ax1.axhline(y=self.beta_star, color='red', linestyle=':', linewidth=2,
                   label=f'First-Best β* = {self.beta_star:.2f}')
        
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Buy Price (β)')
        ax1.set_title('β Convergence: Learned vs. Theoretical (Kouvelis & Lariviere)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ========================================
        # Plot 2: Learned σ vs Realized λ
        # ========================================
        ax2 = axes[0, 1]
        
        # Rolling average of learned sigma
        learned_sigma_rolling = pd.Series(self.learned_sigmas).rolling(window=window, min_periods=1).mean()
        ax2.plot(periods, learned_sigma_rolling, color='orange', linewidth=2, label='Learned σ (rolling avg)')
        
        # Rolling average of realized lambda
        realized_lambda_rolling = pd.Series(self.realized_lambdas).rolling(window=window, min_periods=1).mean()
        ax2.plot(periods, realized_lambda_rolling, color='green', linewidth=2, linestyle='--',
                label='Realized λ (rolling avg)')
        
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Sell Price (σ) / Shadow Price (λ)')
        ax2.set_title('σ Convergence: Learned vs. Realized Shadow Price')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # ========================================
        # Plot 3: β Error Over Time
        # ========================================
        ax3 = axes[1, 0]
        
        beta_errors = np.abs(np.array(self.learned_betas) - np.array(self.theoretical_betas))
        beta_error_rolling = pd.Series(beta_errors).rolling(window=window, min_periods=1).mean()
        
        ax3.plot(periods, beta_error_rolling, color='red', linewidth=2)
        ax3.set_xlabel('Period')
        ax3.set_ylabel('|Learned β - Theoretical E[λ]|')
        ax3.set_title('β Error Over Time (Should Decrease if Learning Works)')
        ax3.grid(True, alpha=0.3)
        
        # ========================================
        # Plot 4: Distribution of Realized λ
        # ========================================
        ax4 = axes[1, 1]
        
        # Histogram of realized shadow prices
        ax4.hist(self.realized_lambdas, bins=50, density=True, alpha=0.7, color='green',
                label='Realized λ distribution')
        ax4.axvline(x=np.mean(self.realized_lambdas), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean λ = {np.mean(self.realized_lambdas):.2f}')
        ax4.axvline(x=self.beta_star, color='red', linestyle=':', linewidth=2,
                   label=f'First-Best β* = {self.beta_star:.2f}')
        
        ax4.set_xlabel('Shadow Price (λ)')
        ax4.set_ylabel('Density')
        ax4.set_title('Distribution of Realized Shadow Prices λ(s, Φ)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # ========================================
        # Plot 5: Theoretical Shadow Price Surface
        # ========================================
        ax5 = axes[2, 0]
        
        # Compute theoretical beta for different (p, s1) combinations
        p_vals = np.arange(35, 65, 2)
        s1_vals = np.arange(40, 100, 5)
        
        for s1 in [50, 70, 90]:
            betas = []
            for p in p_vals:
                mu_d = params.a - params.b * p
                beta = compute_expected_shadow_price(p, s1, mu_d, params.sigma_d)
                betas.append(beta)
            ax5.plot(p_vals, betas, label=f's1 = {s1}', linewidth=2)
        
        ax5.axvline(x=self.p_opt, color='black', linestyle='--', alpha=0.5, label=f'p* = {self.p_opt}')
        ax5.set_xlabel('Market Price (p)')
        ax5.set_ylabel('Expected Shadow Price E[λ]')
        ax5.set_title('Theoretical β* = E[λ] vs. Price (Kouvelis & Lariviere)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # ========================================
        # Plot 6: Summary Statistics
        # ========================================
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        metrics = self.compute_convergence_metrics()
        
        summary_text = f"""
        KOUVELIS & LARIVIERE SHADOW PRICE VALIDATION
        {'=' * 50}
        
        First-Best Solution:
            p* = {self.p_opt}
            s1* = {self.s1_opt}
            s2* = {self.s2_opt}
            β* = {self.beta_star:.2f}
            Profit* = {self.profit_opt:.2f}
        
        Learning Performance:
            Learned β (late avg): {metrics['late_learned_beta']:.2f}
            Theoretical E[λ] (late avg): {metrics['late_theoretical_beta']:.2f}
            β Gap: {metrics['beta_gap']:.2f}
            
        Convergence:
            Early β Error: {metrics['early_beta_error']:.2f}
            Late β Error: {metrics['late_beta_error']:.2f}
            Convergence Ratio: {metrics['beta_convergence_ratio']:.3f}
            (< 1 means learning is working)
        
        Theory:
            β* = E[λ] = p × P(D ≥ s)
            λ = p if D ≥ s, else 0
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Kouvelis & Lariviere (2000) Shadow Price Validation', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shadow_price_validation_{timestamp}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {filename}")
        
        plt.show()
        
        return metrics


# ============================================
# MAIN EXECUTION
# ============================================

def run_full_validation(rounds=5000, use_ucb=True):
    """
    Run complete shadow price validation.
    """
    print("\n" + "=" * 70)
    print("KOUVELIS & LARIVIERE (2000) SHADOW PRICE VALIDATION")
    print("=" * 70)
    print("\nPurpose: Validate that learning agents converge to theoretical shadow prices")
    print("The agents do NOT know the formulas - they learn by trial and error")
    print("We compare what they LEARNED vs. what the MATH says is optimal")
    
    # Step 1: Compute First-Best Solution
    print("\n" + "-" * 70)
    print("STEP 1: Computing First-Best (Theoretical) Solution")
    print("-" * 70)
    
    fb = compute_first_best_solution(verbose=True)
    p_opt = fb['p_opt']
    s1_opt = fb['s1_opt']
    s2_opt = fb['s2_opt']
    beta_shadow = fb['beta_shadow']
    beta_incentive = fb['beta_incentive']
    beta_star = fb['beta_effective']
    profit_opt = fb['profit_opt']
    
    # Step 2: Validate analytical formula with Monte Carlo
    print("\n" + "-" * 70)
    print("STEP 2: Validating K&L Shadow Price Formula (Monte Carlo)")
    print("-" * 70)
    
    mc_mean, mc_std, mc_samples = estimate_shadow_prices_monte_carlo(p_opt, s1_opt)
    print(f"\n  Monte Carlo Estimate (n=10000):")
    print(f"    E[λ] = {mc_mean:.2f} ± {mc_std:.2f}")
    print(f"    Analytical E[λ] = {beta_shadow:.2f}")
    print(f"    Difference: {abs(mc_mean - beta_shadow):.4f}")
    
    print(f"\n  Note: This is the PURE K&L shadow price (coordination value)")
    print(f"        But agents also need β to cover production cost!")
    print(f"        Effective β* = max(E[λ], 2k×s2) = max({beta_shadow:.2f}, {beta_incentive:.2f}) = {beta_star:.2f}")
    
    # Step 3: Run Learning Simulation
    print("\n" + "-" * 70)
    print(f"STEP 3: Running {'UCB' if use_ucb else 'Epsilon-Greedy'} Learning Simulation")
    print("-" * 70)
    
    validator = ShadowPriceValidator()
    model, df_model = validator.run_learning_simulation(rounds=rounds, use_ucb=use_ucb)
    
    # Step 4: Analyze Convergence
    print("\n" + "-" * 70)
    print("STEP 4: Analyzing Convergence")
    print("-" * 70)
    
    metrics = validator.compute_convergence_metrics()
    
    print(f"\n  Convergence Metrics:")
    print(f"    Target β* = {metrics['first_best_beta']:.2f}")
    print(f"    Learned β (late): {metrics['late_learned_beta']:.2f}")
    print(f"    Theoretical E[β] (late): {metrics['late_theoretical_beta']:.2f}")
    print(f"    β Gap: {metrics['beta_gap']:.2f}")
    print(f"    Convergence Ratio: {metrics['beta_convergence_ratio']:.3f}")
    
    if metrics['beta_convergence_ratio'] < 0.5:
        print("\n    ✓ LEARNING IS WORKING: Error decreased significantly!")
    elif metrics['beta_convergence_ratio'] < 1.0:
        print("\n    ~ LEARNING IS PARTIALLY WORKING: Some convergence observed")
    else:
        print("\n    ⚠ LEARNING MAY NOT BE CONVERGING: Error did not decrease")
    
    # Step 5: Plot Results
    print("\n" + "-" * 70)
    print("STEP 5: Generating Validation Plots")
    print("-" * 70)
    
    validator.plot_convergence(save=True)
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    return validator, model, df_model, metrics


def compare_ucb_vs_greedy():
    """
    Compare UCB vs Epsilon-Greedy convergence to shadow prices.
    """
    print("\n" + "=" * 70)
    print("COMPARING UCB vs EPSILON-GREEDY CONVERGENCE TO SHADOW PRICES")
    print("=" * 70)
    
    rounds = 5000
    
    # Run UCB
    print("\n--- UCB Learning ---")
    validator_ucb = ShadowPriceValidator()
    _, _, = validator_ucb.run_learning_simulation(rounds=rounds, use_ucb=True, verbose=False)
    metrics_ucb = validator_ucb.compute_convergence_metrics()
    
    # Run Epsilon-Greedy
    print("\n--- Epsilon-Greedy Learning ---")
    validator_greedy = ShadowPriceValidator()
    _, _ = validator_greedy.run_learning_simulation(rounds=rounds, use_ucb=False, verbose=False)
    metrics_greedy = validator_greedy.compute_convergence_metrics()
    
    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'UCB':>15} {'ε-Greedy':>15}")
    print("-" * 60)
    print(f"{'First-Best β*':<30} {metrics_ucb['first_best_beta']:>15.2f} {metrics_greedy['first_best_beta']:>15.2f}")
    print(f"{'Learned β (late)':<30} {metrics_ucb['late_learned_beta']:>15.2f} {metrics_greedy['late_learned_beta']:>15.2f}")
    print(f"{'β Gap':<30} {metrics_ucb['beta_gap']:>15.2f} {metrics_greedy['beta_gap']:>15.2f}")
    print(f"{'Convergence Ratio':<30} {metrics_ucb['beta_convergence_ratio']:>15.3f} {metrics_greedy['beta_convergence_ratio']:>15.3f}")
    
    # Determine winner
    if metrics_ucb['beta_gap'] < metrics_greedy['beta_gap']:
        print("\n✓ UCB converges closer to theoretical shadow price!")
    else:
        print("\n✓ Epsilon-Greedy converges closer to theoretical shadow price!")
    
    return metrics_ucb, metrics_greedy


if __name__ == "__main__":
    # Run full validation with UCB
    validator, model, df_model, metrics = run_full_validation(rounds=5000, use_ucb=True)
    
    # Additional analysis: Why might learned β differ from theoretical?
    print("\n" + "=" * 70)
    print("INTERPRETATION: WHY LEARNED β DIFFERS FROM THEORETICAL")
    print("=" * 70)
    print("""
    Key Insight from Kouvelis & Lariviere (2000):
    
    1. The SHADOW PRICE β_shadow = E[λ] = p × P(stockout) is the
       "coordination value" of inventory - how much one more unit is worth.
       
    2. But in our model, β also must INCENTIVIZE production!
       Operations won't produce if β < marginal cost = 2k×x
       
    3. The FIRST-BEST β* = max(β_shadow, β_incentive)
       ensures both coordination AND production incentives.
       
    4. However, in a DECENTRALIZED game, agents may find a NASH EQUILIBRIUM
       that differs from the first-best. This is the "price of anarchy".
       
    What the agents learned:
    - If learned β > first-best β*: Agents over-compensate Operations,
      potentially causing over-production
    - If learned β < first-best β*: Agents under-compensate Operations,
      potentially causing under-production
      
    The gap between learned and theoretical reveals coordination failures
    that transfer pricing can address but not fully solve.
    """)
