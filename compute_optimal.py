"""
Matematiksel optimal hesaplama - Price=65 için
"""
import numpy as np
from scipy import stats

# Parametreler (params.py'den)
H1 = 3          # Marketing holding cost
H2 = 5          # Operations holding cost  
P_BO = 15       # Backorder cost (yeni değer)
ALPHA = 0.5     # Backorder cost allocation
k = 0.35        # Production cost coefficient
a = 120         # Demand intercept
b = 1.5         # Price sensitivity
std_dev = 10    # Demand std deviation

# Agent'ın seçtiği price
price = 55

# Mean demand for price=65
mean_demand = a - b * price
print(f"="*60)
print(f"MATEMATIKSEL OPTIMAL HESAPLAMA (Price = {price})")
print(f"="*60)
print(f"\nMean Demand: {mean_demand}")
print(f"Std Deviation: {std_dev}")

# Newsvendor kritik oranı (critical ratio) - Retailer (Marketing)
# Retailer toplam holding cost: h1 + h2 = 8
# Retailer backorder cost payı: ALPHA * P_BO = 15
h_total = H1 + H2
p_bo_retailer = ALPHA * P_BO

# Critical ratio for retailer
CR_retailer = p_bo_retailer / (h_total + p_bo_retailer)
print(f"\n--- Retailer (Marketing) ---")
print(f"Holding cost: {h_total}")
print(f"Backorder cost (share): {p_bo_retailer}")
print(f"Critical Ratio: {CR_retailer:.4f}")

# Optimal base stock (z-value from normal distribution)
z_retailer = stats.norm.ppf(CR_retailer)
S1_optimal = mean_demand + z_retailer * std_dev
print(f"z-value: {z_retailer:.4f}")
print(f"Optimal S1: {S1_optimal:.2f}")

# Supplier (Operations)
# Holding cost: h2 = 5
# Backorder cost: (1-ALPHA)*P_BO + internal = 15 (yaklaşık)
p_bo_supplier = (1 - ALPHA) * P_BO

CR_supplier = p_bo_supplier / (H2 + p_bo_supplier)
print(f"\n--- Supplier (Operations) ---")
print(f"Holding cost: {H2}")
print(f"Backorder cost (share): {p_bo_supplier}")
print(f"Critical Ratio: {CR_supplier:.4f}")

z_supplier = stats.norm.ppf(CR_supplier)
S2_optimal = mean_demand + z_supplier * std_dev
print(f"z-value: {z_supplier:.4f}")
print(f"Optimal S2: {S2_optimal:.2f}")

# Transfer price (sigma = beta)
# Optimal transfer price: holding cost echelon farkı
# w* = h1 + h2 - h2 = h1 veya marginal cost bazlı hesaplanabilir
# Literatürde: w* ≈ (h1 + h2) + (underage cost adjustment)
# Basit yaklaşım: w* = holding cost gradient = H1 + H2 = 8
# Veya: w* = (h1+h2) * (1 + critical ratio adjustment)

# Daha doğru hesap: Transfer price = echelon holding cost = h1 + h2
w_optimal = H1 + H2  # Basit yaklaşım
print(f"\n--- Transfer Price (Principal) ---")
print(f"Optimal w (sigma = beta): {w_optimal}")

# Alternatif: Clark-Scarf optimal transfer price
# w* = h1 (upstream'e aktarılan echelon cost)
w_clark_scarf = H1
print(f"Clark-Scarf w*: {w_clark_scarf}")

print(f"\n" + "="*60)
print(f"KARŞILAŞTIRMA")
print(f"="*60)
print(f"\n{'Değişken':<20} {'Optimal':<15} {'Agent':<15} {'Fark':<15}")
print(f"{'-'*60}")
print(f"{'S1 (Marketing)':<20} {S1_optimal:<15.1f} {45:<15} {abs(S1_optimal-45):<15.1f}")
print(f"{'S2 (Operations)':<20} {S2_optimal:<15.1f} {55:<15} {abs(S2_optimal-55):<15.1f}")
print(f"{'Sigma (Principal)':<20} {w_optimal:<15} {10:<15} {abs(w_optimal-10):<15}")
print(f"{'Beta (Principal)':<20} {w_optimal:<15} {70:<15} {abs(w_optimal-70):<15}")
print(f"{'Price':<20} {price:<15} {65:<15} {0:<15}")

print(f"\n" + "="*60)
print(f"YORUM")
print(f"="*60)
print(f"""
1. Optimal transfer price (σ = β) yaklaşık {w_optimal} olmalı
2. Agent σ=10, β=70 seçmiş - bu dengesiz!
3. H1+H2 içindeki (σ-β) terimi nedeniyle β>σ Principal için avantajlı görünüyor
4. Ama gerçekte σ = β olmalı (budget-balanced transfer)

Problem: Principal'ın reward fonksiyonu σ=β'yı teşvik etmiyor.
""")
