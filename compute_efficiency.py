"""
Shadow Price ve Efficiency Hesaplama
Agent sonuçları vs Optimal karşılaştırması
"""
import numpy as np
from scipy import stats

# Parametreler (params.py'den - P_BO=15)
H1 = 3
H2 = 5
P_BO = 15
ALPHA = 0.5
k = 0.35
a = 120
b = 1.5
std_dev = 10

# Agent sonuçları
agent_sigma = 10
agent_beta = 60
agent_s1 = 40
agent_s2 = 65
agent_price = 55

# Optimal değerler
opt_sigma = 8
opt_beta = 8
opt_s1 = 37.1
opt_s2 = 40.0
opt_price = 55

# Mean demand
mean_demand = a - b * agent_price  # 37.5

print("="*70)
print("SHADOW PRICE VE EFFICIENCY HESAPLAMA")
print("="*70)

# ===== SHADOW PRICE HESAPLAMA =====
print("\n--- SHADOW PRICES (Marjinal Maliyet Etkileri) ---\n")

# S1 shadow price: Holding cost + opportunity cost
# dC/dS1 ≈ h_total * P(D < S1) - p_bo * P(D >= S1)
h_total = H1 + H2
p_bo_retailer = ALPHA * P_BO

# P(D < S1) for agent vs optimal
prob_agent_s1 = stats.norm.cdf(agent_s1, loc=mean_demand, scale=std_dev)
prob_opt_s1 = stats.norm.cdf(opt_s1, loc=mean_demand, scale=std_dev)

shadow_s1_agent = h_total * prob_agent_s1 - p_bo_retailer * (1 - prob_agent_s1)
shadow_s1_opt = h_total * prob_opt_s1 - p_bo_retailer * (1 - prob_opt_s1)

print(f"S1 Shadow Price:")
print(f"  Agent (S1={agent_s1}): {shadow_s1_agent:.4f}")
print(f"  Optimal (S1={opt_s1:.1f}): {shadow_s1_opt:.4f}")
print(f"  → Optimal'de shadow price ≈ 0 olmalı (FOC)")

# S2 shadow price
p_bo_supplier = (1 - ALPHA) * P_BO
prob_agent_s2 = stats.norm.cdf(agent_s2, loc=mean_demand, scale=std_dev)
prob_opt_s2 = stats.norm.cdf(opt_s2, loc=mean_demand, scale=std_dev)

shadow_s2_agent = H2 * prob_agent_s2 - p_bo_supplier * (1 - prob_agent_s2)
shadow_s2_opt = H2 * prob_opt_s2 - p_bo_supplier * (1 - prob_opt_s2)

print(f"\nS2 Shadow Price:")
print(f"  Agent (S2={agent_s2}): {shadow_s2_agent:.4f}")
print(f"  Optimal (S2={opt_s2:.1f}): {shadow_s2_opt:.4f}")

# Sigma/Beta shadow price
# dC/dσ = E[U1] (Marketing için maliyet artışı)
# dC/dβ = -E[U1] (Operations için maliyet azalışı)
# Net sistem etkisi: (σ - β) * E[U1]
E_U1 = mean_demand  # Yaklaşık olarak

print(f"\nTransfer Price Shadow Prices:")
print(f"  dC/dσ ≈ E[U1] = {E_U1:.2f} (her birim σ artışı maliyeti bu kadar artırır)")
print(f"  dC/dβ ≈ -E[U1] = {-E_U1:.2f} (her birim β artışı maliyeti bu kadar azaltır)")
print(f"  Net: d(H1+H2)/d(σ-β) = E[U1] = {E_U1:.2f}")

# ===== COST HESAPLAMA =====
print("\n" + "="*70)
print("EXPECTED COST HESAPLAMA")
print("="*70)

def compute_expected_cost(s1, s2, sigma, beta, price, n_sim=50000, include_transfer=True):
    """Monte Carlo ile expected cost hesapla"""
    np.random.seed(42)
    
    total_cost = 0
    total_real_cost = 0  # Transfer hariç
    I1 = I2 = 0
    B1 = B2 = 0
    U1_prev = U2_prev = 0
    
    warmup = 2000
    
    for t in range(n_sim + warmup):
        # Arrivals
        I1 += U1_prev
        I2 += U2_prev
        
        # IP
        IP1 = I1 - B1
        IP2 = I2 - B2
        
        # Orders
        O1 = max(0, s1 - IP1)
        O2 = max(0, s2 - IP2)
        
        # Releases
        ship = min(I2, B2 + O1)
        I2 -= ship
        B2 = B2 + O1 - ship
        U1 = ship
        U2 = O2
        
        # Demand
        mean = a - b * price
        D = max(0, int(np.random.normal(mean, std_dev)))
        sales = min(I1, B1 + D)
        I1 -= sales
        B1 = B1 + D - sales
        
        # Costs (ONLY OPERATIONAL - no revenue, no transfer)
        # Holding costs + backorder costs + production cost
        H1_t = (H1 + H2) * I1 + ALPHA * P_BO * B1  # Marketing operational cost
        H2_t = H2 * (I2 + U1) + (1 - ALPHA) * P_BO * B1 + P_BO * B2 + k * (U2)**2  # Operations operational cost
        
        if t >= warmup:
            total_cost += (H1_t + H2_t)
        
        U1_prev, U2_prev = U1, U2
    
    return total_cost / n_sim

print("\nHesaplanıyor (Monte Carlo)...")

# Agent cost (operasyonel maliyet, revenue hariç)
agent_cost = compute_expected_cost(agent_s1, agent_s2, 0, 0, agent_price)

# Optimal cost (operasyonel maliyet)
optimal_cost = compute_expected_cost(int(opt_s1), int(opt_s2), 0, 0, opt_price)

print(f"\n--- Expected Costs (per period, sadece operasyonel) ---")
print(f"  Agent Cost (S1={agent_s1}, S2={agent_s2}, Price={agent_price}): {agent_cost:.2f}")
print(f"  Optimal Cost (S1={int(opt_s1)}, S2={int(opt_s2)}, Price={opt_price:.1f}): {optimal_cost:.2f}")

# ===== EFFICIENCY HESAPLAMA =====
print("\n" + "="*70)
print("EFFICIENCY HESAPLAMA")
print("="*70)

# Agent cost > Optimal cost olmalı (optimal en düşük)
# Efficiency = Optimal Cost / Agent Cost (100% = optimal)

if agent_cost != 0:
    efficiency = (optimal_cost / agent_cost) * 100
else:
    efficiency = 0

# Loss calculation
loss = agent_cost - optimal_cost
loss_pct = (loss / optimal_cost) * 100 if optimal_cost != 0 else 0

print(f"\n--- Efficiency Metrics ---")
print(f"  Efficiency (Agent/Optimal): {efficiency:.2f}%")
print(f"  Absolute Loss:              {loss:.2f} per period")
print(f"  Relative Loss:              {loss_pct:.2f}%")

# ===== DECOMPOSITION =====
print("\n" + "="*70)
print("LOSS DECOMPOSITION (Kaynak Analizi)")
print("="*70)

# Her bir sapmanın etkisini hesapla (tümü σ=β=0 ile)
# S1 etkisi
cost_s1_only = compute_expected_cost(agent_s1, int(opt_s2), 0, 0, opt_price)
s1_effect = cost_s1_only - optimal_cost

# S2 etkisi
cost_s2_only = compute_expected_cost(int(opt_s1), agent_s2, 0, 0, opt_price)
s2_effect = cost_s2_only - optimal_cost

print(f"\n  S1 Sapma Etkisi ({agent_s1} vs {int(opt_s1)}): {s1_effect:.2f}")
print(f"  S2 Sapma Etkisi ({agent_s2} vs {int(opt_s2)}): {s2_effect:.2f}")
print(f"  Toplam (yaklaşık):            {s1_effect + s2_effect:.2f}")
print(f"  Gerçek Fark:                  {loss:.2f}")

from scipy.optimize import minimize

# ===== NUMERICAL OPTIMIZATION İLE GERÇEK OPTİMAL =====
print("\n" + "="*70)
print("NUMERICAL OPTIMIZATION İLE GERÇEK OPTİMAL")
print("="*70)

def cost_function_for_opt(params):
    """Minimize edilecek maliyet fonksiyonu"""
    s1, s2 = params
    # S2 >= S1 constraint (supplier stoku >= retailer)
    if s2 < s1:
        return 1e10
    return compute_expected_cost(int(s1), int(s2), 0, 0, opt_price, n_sim=1000)

# Initial guess - agent values
x0 = [agent_s1, agent_s2]
bounds = [(10, 100), (10, 150)]

print("Optimizing...")
result = minimize(cost_function_for_opt, x0, method='Nelder-Mead', 
                  options={'maxiter': 200, 'xatol': 1, 'fatol': 1})

opt_s1_num = int(result.x[0])
opt_s2_num = int(result.x[1])
opt_cost_num = result.fun

print(f"\n--- Numerical Optimal ---")
print(f"  Optimal S1: {opt_s1_num}")
print(f"  Optimal S2: {opt_s2_num}")
print(f"  Min Cost:   {opt_cost_num:.2f}")

# Karşılaştırma
print("\n" + "="*70)
print("KARŞILAŞTIRMA")
print("="*70)

agent_cost_final = compute_expected_cost(agent_s1, agent_s2, 0, 0, agent_price, n_sim=5000)
optimal_cost_final = compute_expected_cost(opt_s1_num, opt_s2_num, 0, 0, opt_price, n_sim=5000)

print(f"\n  Agent Cost (S1={agent_s1}, S2={agent_s2}):   {agent_cost_final:.2f}")
print(f"  Optimal Cost (S1={opt_s1_num}, S2={opt_s2_num}): {optimal_cost_final:.2f}")

if optimal_cost_final > 0:
    eff = (optimal_cost_final / agent_cost_final) * 100 if agent_cost_final > 0 else 0
    loss = agent_cost_final - optimal_cost_final
    loss_pct = (loss / optimal_cost_final) * 100
    print(f"\n  Efficiency: {eff:.2f}%")
    print(f"  Loss: {loss:.2f} ({loss_pct:.2f}%)")
