"""
Two-Stage Serial Supply Chain Model with Split-Principal Architecture.

This model implements a MARL simulation where decentralized agents attempt
to converge to the centralized system optimum through learning.

Agent execution order (strict causal sequence):
1. Principals Update: Beta and Sigma set transfer prices
2. Marketing Decision: Observe I1, select (p, s1), place order y
3. Operations Decision: Observe order y, select (s2, x), produce
4. Physical Flow: Production → Shipping → Inventory updates
5. Market Realization: Demand → Sales → Backorders
6. Reward & Learning: Calculate rewards, update policies
"""

import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector

from agents import (
    PrincipalAgent, PrincipalBetaAgent, PrincipalSigmaAgent,
    MarketingAgent, OperationsAgent,
    GreedyPAgent, GreedyMAgent, GreedyOAgent
)
import params


def create_agents(model, agent_types):
    """Create agents based on specified types."""
    # Ajanları doğrudan oluşturuyoruz (create_agents metodu yerine)
    
    for agent_type in agent_types:
        if agent_type == "principal":
            # Unified Principal agent
            PrincipalAgent(model)
            
        elif agent_type == "greedy_beta":
            # Doğrudan sınıfı çağırarak örnek (instance) oluşturuyoruz
            PrincipalBetaAgent(model) 
            
        elif agent_type == "greedy_sigma":
            PrincipalSigmaAgent(model)
            
        elif agent_type == "greedy_m":
            MarketingAgent(model)
            
        elif agent_type == "greedy_o":
            OperationsAgent(model)
            
        # Legacy types
        elif agent_type == "greedy_p":
            GreedyPAgent(model)
class TwoStageSupplyChainModel(Model):
    """
    Two-Stage Serial Supply Chain with Split-Principal Architecture.
    
    Agents:
    - Principal: Unified agent controlling both β and σ (buy and sell transfer prices)
    - Marketing: Retailer managing inventory I1, sets price p and base-stock s1
    - Operations: Supplier managing inventory I2, sets base-stock s2 and production x
    """
    
    def __init__(self, agent_types=("principal", "greedy_m", "greedy_o")):
        super().__init__()
        
        # Simulation control
        self.rng = np.random.default_rng(params.SEED)
        self.t = 0
        
        # Initialize agents
        create_agents(self, agent_types)
        self.rewards = {a: 0.0 for a in self.agents}
        
        # Inventory state
        self.I1 = 0  # Marketing (retailer) inventory
        self.I2 = 0  # Operations (supplier) inventory
        
        # Decision variables (for reporting)
        self.beta = 0.0   # Buy price from Principal Beta
        self.sigma = 0.0  # Sell price from Principal Sigma
        self.p = 0        # Market price from Marketing
        self.s1 = 0       # Base-stock level for Marketing
        self.s2 = 0       # Base-stock level for Operations
        self.x = 0        # Production quantity
        self.y = 0        # Order quantity
        
        # State variables (for reporting)
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
    
    def step(self):
        """
        Execute one simulation step with strict causal ordering.
        """
        # Get agents by type
        principal = None  # Unified principal
        principal_beta = None
        principal_sigma = None
        marketing = None
        operations = None
        
        for agent in self.agents:
            if isinstance(agent, PrincipalAgent):
                principal = agent
            elif isinstance(agent, PrincipalBetaAgent):
                principal_beta = agent
            elif isinstance(agent, PrincipalSigmaAgent):
                principal_sigma = agent
            elif isinstance(agent, (MarketingAgent, GreedyMAgent)):
                marketing = agent
            elif isinstance(agent, (OperationsAgent, GreedyOAgent)):
                operations = agent
        
        # ============================================
        # 1. PRINCIPAL UPDATE (set transfer prices)
        # ============================================
        if principal:
            # Unified principal sets both beta and sigma
            principal.select_action()
            self.beta = principal.get_beta()
            self.sigma = principal.get_sigma()
        else:
            # Fallback to split principals
            if principal_beta:
                principal_beta.select_action()
                self.beta = principal_beta.get_beta()
            
            if principal_sigma:
                principal_sigma.select_action()
            self.sigma = principal_sigma.get_sigma()
        
        # ============================================
        # 2. MARKETING DECISION
        # ============================================
        if marketing:
            marketing.select_action()
            self.p = marketing.p
            self.s1 = marketing.s1
            self.y = marketing.compute_order(self.I1)  # y = max(0, s1 - I1)
        
        # ============================================
        # 3. OPERATIONS DECISION
        # ============================================
        if operations:
            operations.select_action()
            self.s2 = operations.s2
            self.x = operations.compute_production(self.I2)  # x = max(0, s2 - I2)
        # ============================================
        # 4. PHYSICAL FLOW (Production & Shipping)
        # ============================================
        # 
        # x is already computed from base-stock: x = max(0, s2 - I2)
        # Now apply profitability constraint from transfer price β
        #
        
        # Production gating: Only produce if profitable
        # Marginal cost = d(k*x²)/dx = 2*k*x
        # Profitable if: β >= 2*k*x → x <= β/(2*k)
        # if params.k > 0:
        #     max_profitable_x = int(self.beta / (2 * params.k))
        #     self.x = min(self.x, max_profitable_x)
        
        # Production: I2 increases by x
        self.I2 += self.x
        
        # Shipment gating: Marketing only orders if market price covers transfer cost
        # Profitable if: p >= σ
        # if self.p >= self.sigma:
        #     self.shipment = min(self.y, self.I2)
        # else:
        #     # Not profitable to order - shipment blocked
        #     self.shipment = 0
        #     self.y = 0  # Reflect that no order was placed
        self.shipment = min(self.y, self.I2)
        self.I2 -= self.shipment
        self.I1 += self.shipment
        
        # ============================================
        # 5. MARKET REALIZATION
        # ============================================
        # Generate demand based on price
        self.demand = params.sample_demand(self.rng, self.p)
        
        # Sales = min(D, I1)
        self.sales = min(self.demand, self.I1)
        
        # Backorders = D - Sales (lost sales)
        self.backorders = self.demand - self.sales
        
        # Update inventory
        self.I1 -= self.sales
        
        # ============================================
        # 6. REWARD CALCULATION (Echelon Holding Costs per Cachon & Zipkin 1999)
        # ============================================
        # Marketing: (h1-h2)×I1 - incremental cost only
        self.reward_marketing = MarketingAgent.compute_reward(
            p=self.p,
            sales=self.sales,
            sigma=self.sigma,
            shipment=self.shipment,  # Pay for goods RECEIVED, not ordered
            h1=params.H1,
            h2=params.H2,
            I1=self.I1,  # Echelon: (h1-h2)×I1
            alpha=params.ALPHA,
            pi=params.P_BO,
            backorders=self.backorders
        )
        
        # Operations: h2×(I1+I2) - echelon inventory = all downstream
        # Gets paid only for shipped units, not all production!
        self.reward_operations = OperationsAgent.compute_reward(
            beta=self.beta,
            x=self.x,
            shipped=self.shipment,  # Only get paid for what's shipped!
            k=params.k,
            h2=params.H2,
            I1=self.I1,  # Echelon: h2×(I1+I2)
            I2=self.I2,
            alpha=params.ALPHA,
            pi=params.P_BO,
            backorders=self.backorders
        )
        
        # Principal reward: system profit (minimize total cost)
        # Total cost = all costs - revenue
        revenue = self.p * self.sales
        production_cost = params.k * (self.x ** 2)
        holding_costs = params.H2 * (self.I2) + (params.H1-params.H2) * self.I1
        backorder_cost = params.P_BO * self.backorders
        
        self.total_cost = production_cost + holding_costs + backorder_cost - revenue
        system_profit = -self.total_cost
        
        # Principal reward: system profit + margin incentive
        # (σ - β) × shipment = Principal's per-unit margin on transfers
        # This incentivizes Principal to:
        # - Increase σ (charge more to Marketing) → reduces Marketing over-ordering
        # - Decrease β (pay less to Operations) → reduces Operations over-production
        # But balanced by system profit which needs sales to happen
        # Weight = 0.5 for balanced incentive
        
        #principal_margin = 1.0 * (self.sigma - self.beta) * self.shipment
        
        self.reward_principal = system_profit #+ principal_margin
        self.reward_beta = system_profit  # For backwards compatibility
        self.reward_sigma = system_profit
        
        # ============================================
        # 7. LEARNING UPDATE
        # ============================================
        # Assign rewards to agents
        if principal:
            self.rewards[principal] = self.reward_principal
            principal.reward = self.reward_principal
            principal.reward_cum += self.reward_principal
        
        if principal_beta:
            self.rewards[principal_beta] = self.reward_beta
            principal_beta.reward = self.reward_beta
            principal_beta.reward_cum += self.reward_beta
        
        if principal_sigma:
            self.rewards[principal_sigma] = self.reward_sigma
            principal_sigma.reward = self.reward_sigma
            principal_sigma.reward_cum += self.reward_sigma
        
        if marketing:
            self.rewards[marketing] = self.reward_marketing
            marketing.reward = self.reward_marketing
            marketing.reward_cum += self.reward_marketing
        
        if operations:
            self.rewards[operations] = self.reward_operations
            operations.reward = self.reward_operations
            operations.reward_cum += self.reward_operations
        
        # Update beliefs (learning)
        self.agents.shuffle_do("update_belief")
        
        # ============================================
        # 8. METRICS & DATA COLLECTION
        # ============================================
        # Regret vs optimal (will be meaningful after benchmark is computed)
        self.regret_vs_optimal = self.total_cost - params.CTOT_OPT
        self.regret_cumulative += self.regret_vs_optimal
        
        # Advance time
        self.t += 1
        
        # Collect data
        
        self.datacollector.collect(self)