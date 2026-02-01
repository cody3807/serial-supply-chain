"""
Agent classes for the Split-Principal Supply Chain MARL Simulation.

Agents:
- PrincipalBetaAgent: Controls supply-side buy price (β)
- PrincipalSigmaAgent: Controls demand-side sell price (σ)  
- MarketingAgent: Retailer managing inventory and market pricing
- OperationsAgent: Supplier managing production and upstream inventory
"""

import numpy as np
from mesa import Agent
import params


class BaseGreedyAgent(Agent):
    """Base class for ε-greedy bandit agents."""
    
    def __init__(self, model, action_space):
        super().__init__(model)
        
        # Action space
        self.action_space = action_space
        self.n_actions = len(self.action_space)
        
        # Bandit statistics
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.average_reward = np.zeros(self.n_actions, dtype=float)
        
        # Epsilon for ε-greedy
        self.eps = params.epsilon_at(model.t, params.ROUNDS)
        
        # Selected action + reward
        self.action_idx = None
        self.action = None
        self.reward = 0.0
        
        # Cumulative tracking
        self.reward_cum = 0.0
    
    def select_action(self):
        """Select an action using ε-greedy strategy."""
        if self.random.random() < self.eps:
            # Exploration: random action
            action_idx = self.random.randrange(self.n_actions)
        else:
            # Exploitation: best known action
            action_idx = int(np.argmax(self.average_reward))
        
        self.action_idx = action_idx
        self.action = self.action_space[action_idx]
    
    def update_belief(self):
        """Update reward estimates using incremental mean update."""
        a_idx = self.action_idx
        r = self.model.rewards[self]
        
        # Incremental update: μ_new = μ_old + (r - μ_old) / n
        n = self.counts[a_idx] + 1
        self.counts[a_idx] = n
        self.average_reward[a_idx] += (r - self.average_reward[a_idx]) / n
        
        # Update epsilon for next round
        self.eps = params.epsilon_at(self.model.t + 1, params.ROUNDS)


# ============================================
# Principal Agents (Split Architecture)
# ============================================

class PrincipalAgent(BaseGreedyAgent):
    """
    Unified Principal Agent - Controls both β and σ.
    
    Goal: Minimize total system cost by setting transfer prices that
    incentivize Marketing and Operations to behave optimally.
    
    Decisions:
    - Beta (β): Buy price paid to Operations for each unit produced
    - Sigma (σ): Sell price charged to Marketing for each unit ordered
    
    Reward: System profit = Revenue - Total Costs
    """
    
    def __init__(self, model):
        action_space = params.action_space_principal()
        super().__init__(model, action_space)
        self.name = "Principal"
        
        # Current decisions
        self.beta = 0.0
        self.sigma = 0.0
    
    def select_action(self):
        """Select (beta, sigma) action."""
        super().select_action()
        if self.action is not None:
            self.beta = float(self.action[0])
            self.sigma = float(self.action[1])
    
    def get_beta(self):
        """Return the selected beta value."""
        return self.beta
    
    def get_sigma(self):
        """Return the selected sigma value."""
        return self.sigma


class PrincipalBetaAgent(BaseGreedyAgent):
    """
    Principal Beta Agent - Supply Coordinator.
    Controls the internal buy price (β).
    Goal: Incentivize Operations agent to produce enough to avoid stockouts.
    """
    
    def __init__(self, model):
        action_space = params.action_space_principal_beta()
        super().__init__(model, action_space)
        self.name = "Principal Beta"
    
    def get_beta(self):
        """Return the selected beta value."""
        return float(self.action) if self.action is not None else 0.0


class PrincipalSigmaAgent(BaseGreedyAgent):
    """
    Principal Sigma Agent - Demand Coordinator.
    Controls the internal sell price (σ).
    Goal: Prevent double marginalization by finding transfer price
    that encourages Marketing to set optimal market price.
    """
    
    def __init__(self, model):
        action_space = params.action_space_principal_sigma()
        super().__init__(model, action_space)
        self.name = "Principal Sigma"
    
    def get_sigma(self):
        """Return the selected sigma value."""
        return float(self.action) if self.action is not None else 0.0


# ============================================
# Marketing Agent (Retailer)
# ============================================

class MarketingAgent(BaseGreedyAgent):
    """
    Marketing Agent (Retailer).
    
    Role: Manages final stage inventory (I1) and sets market strategy.
    
    Decisions:
    - Price (p): Market price, must be > 0
    - Base-Stock Level (s1): Target inventory level
    
    Order Quantity: y = max(0, s1 - I1)
    
    Reward Function:
    R_M = (p × Sales) - (σ × y) - (h1 × I1) - (α × π × Backorders)
    """
    
    def __init__(self, model):
        action_space = params.action_space_marketing()
        super().__init__(model, action_space)
        self.name = "Marketing"
        
        # Current decisions
        self.s1 = 0  # Base-stock level
        self.p = int(params.p_range.min())  # Market price
        self.y = 0  # Order quantity
    
    def select_action(self):
        """Select (s1, p) action."""
        super().select_action()
        if self.action is not None:
            self.s1 = int(self.action[0])
            self.p = max(5, int(self.action[1]))  # Ensure p > 0
    
    def compute_order(self, I1):
        """Compute order quantity based on base-stock policy."""
        self.y = max(0, self.s1 - I1)
        return self.y
    
    @staticmethod
    def compute_reward(p, sales, sigma, shipment, h1, h2, I1, alpha, pi, backorders):
        """
        Compute Marketing agent reward using ECHELON holding costs.
        
        Per Cachon & Zipkin (1999):
        - Echelon holding cost at retailer = (H1 - H2) × I1
        - This is the INCREMENTAL cost of holding at retailer vs supplier
        
        R_M = (p × Sales) - (σ × shipment) - ((h1-h2) × I1) - (α × π × Backorders)
        """
        revenue = p * sales
        transfer_cost = sigma * shipment  # Pay only for goods actually received
        holding_cost = (h1 - h2) * I1  # Echelon: incremental cost only
        backorder_cost = alpha * pi * backorders
        return revenue - transfer_cost - holding_cost - backorder_cost


# ============================================
# Operations Agent (Supplier/Manufacturer)
# ============================================

class OperationsAgent(BaseGreedyAgent):
    """
    Operations Agent (Supplier/Manufacturer).
    
    Role: Manages production and upstream inventory (I2).
    
    Decisions:
    - Base-Stock Level (s2): Target inventory level
    
    Production is DERIVED from base-stock policy:
    - x = max(0, s2 - I2) — produce up to target level
    
    Reward Function (demand-driven):
    R_O = (β × shipped) - (k × x²) - h2×(I1+I2) - ((1-α) × π × Backorders)
    
    NOTE: Gets paid only for shipped units, not produced!
    This incentivizes matching production to actual demand.
    """
    
    def __init__(self, model):
        # Only s2 in action space now (not s2, x)
        action_space = params.action_space_operations()  # Just base-stock levels for Operations
        super().__init__(model, action_space)
        self.name = "Operations"
        
        # Current decisions
        self.s2 = 0  # Base-stock level
        self.x = 0   # Production quantity (derived)
        self.tp = 0  # Transfer price (for reference)
    
    def select_action(self):
        """Select s2 action (base-stock level only)."""
        super().select_action()
        if self.action is not None:
            #self.s2 = int(self.action)  # Single action, not tuple
            self.s2 = int(self.action[0])
            self.tp = max(5, int(self.action[1]))  # Ensure p > 0
    
    def compute_production(self, I2):
        """
        Compute production quantity using base-stock policy.
        x = max(0, s2 - I2) — produce up to target level
        """
        self.x = max(0, self.s2 - I2)
        return self.x
    
    @staticmethod
    def compute_reward(beta, x, shipped, k, h2, I1, I2, alpha, pi, backorders):
        """
        Compute Operations agent reward using ECHELON holding costs.
        
        Per Cachon & Zipkin (1999):
        - Echelon inventory at supplier = I1 + I2 (all inventory downstream)
        - Echelon holding cost = h2 × (I1 + I2)
        
        IMPORTANT: Transfer revenue is based on SHIPPED units, not produced!
        This incentivizes Operations to produce only what Marketing needs.
        
        R_O = (β × shipped) - (k × x²) - (h2 × (I1+I2)) - ((1-α) × π × Backorders)
        """
        transfer_revenue = beta * shipped  # Only get paid for what's shipped!
        production_cost = k * (x ** 2)  # Convex cost for ALL production
        holding_cost = h2 * (I1 + I2)  # Echelon: all downstream inventory
        backorder_cost = (1 - alpha) * pi * backorders
        return transfer_revenue - production_cost - holding_cost - backorder_cost


# ============================================
# Legacy Agent Classes (for backwards compatibility)
# ============================================

class GreedyAgent(BaseGreedyAgent):
    """ε-greedy bandit agent (legacy)."""
    def __init__(self, model):
        super().__init__(model, params.action_space())

class GreedyPAgent(BaseGreedyAgent):
    """ε-greedy principal agent (legacy - deprecated)."""
    def __init__(self, model):
        # Legacy: controls both sigma and beta as tuple
        from itertools import product
        beta_vals = params.action_space_principal_beta()
        sigma_vals = params.action_space_principal_sigma()
        action_space = np.array(list(product(sigma_vals, beta_vals)))
        super().__init__(model, action_space)

class GreedyMAgent(MarketingAgent):
    """Alias for MarketingAgent."""
    pass

class GreedyOAgent(OperationsAgent):
    """Alias for OperationsAgent."""
    pass

class UcbAgent(BaseGreedyAgent):
    """UCB1 bandit agent (legacy)."""
    def __init__(self, model):
        super().__init__(model, params.action_space())
    
    def select_action(self):
        """Select action using UCB1 algorithm."""
        untried = np.flatnonzero(self.counts == 0)
        
        if len(untried) > 0:
            action_idx = int(untried[0])
        else:
            confidence_bound = np.sqrt((2.0 * np.log(params.ROUNDS)) / self.counts)
            ucb = self.average_reward + confidence_bound
            best = np.flatnonzero(ucb == ucb.max())
            action_idx = int(self.random.choice(best))
        
        self.action_idx = action_idx
        self.action = self.action_space[action_idx]
