import numpy as np
from mesa import Agent
import params

class BaseGreedyAgent(Agent):
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
        if self.random.random() < self.eps:
            action_idx = self.random.randrange(self.n_actions)
        else:
            action_idx = int(np.argmax(self.average_reward))
        
        self.action_idx = action_idx
        self.action = self.action_space[action_idx]
    
    def update_belief(self):
        a_idx = self.action_idx
        r = self.model.rewards[self]
        
        # Incremental update: μ_new = μ_old + (r - μ_old) / n
        n = self.counts[a_idx] + 1
        self.counts[a_idx] = n
        self.average_reward[a_idx] += (r - self.average_reward[a_idx]) / n
        
        # Update epsilon for next round
        self.eps = params.epsilon_at(self.model.t + 1, params.ROUNDS)


# ============================================
# Principal Agent
# ============================================

class PrincipalAgent(BaseGreedyAgent):
    def __init__(self, model):
        action_space = params.action_space_principal()
        super().__init__(model, action_space)
        self.name = "Principal"
        
        # Current decisions
        self.beta = 0.0
        self.sigma = 0.0
    
    def select_action(self):
        super().select_action()
        if self.action is not None:
            self.beta = float(self.action[0])
            self.sigma = float(self.action[1])
    
    def get_beta(self):
        return self.beta
    
    def get_sigma(self):
        return self.sigma

# ============================================
# Marketing Agent (Retailer)
# ============================================

class MarketingAgent(BaseGreedyAgent):
    def __init__(self, model):
        action_space = params.action_space_marketing()
        super().__init__(model, action_space)
        self.name = "Marketing"
        
        # Current decisions
        self.s1 = 0  # Base-stock level
        self.p = int(params.p_range.min())  # Market price
        self.y = 0  # Order quantity
    
    def select_action(self):
        super().select_action()
        if self.action is not None:
            self.s1 = int(self.action[0])
            self.p = max(5, int(self.action[1]))  # Ensure p > 0
    
    def compute_order(self, I1):
        self.y = max(0, self.s1 - I1)
        return self.y
    
    @staticmethod
    def compute_reward(p, sales, sigma, shipment, h1, h2, I1, alpha, pi, backorders):
        revenue = p * sales
        transfer_cost = sigma * shipment 
        holding_cost = (h1 - h2) * I1
        backorder_cost = alpha * pi * backorders
        return revenue - transfer_cost - holding_cost - backorder_cost

# ============================================
# Operations Agent (Supplier/Manufacturer)
# ============================================

class OperationsAgent(BaseGreedyAgent):
    def __init__(self, model):
        action_space = params.s2_range
        super().__init__(model, action_space)
        self.name = "Operations"
        
        # Current decisions
        self.s2 = 0  # Base-stock level
        self.x = 0   # Production quantity (derived)
        self.tp = 0  # Transfer price (for reference)
    
    def select_action(self):
        super().select_action()
        if self.action is not None:
            #self.s2 = int(self.action)  # Single action, not tuple
            self.s2 = int(self.action[0])
            self.tp = max(5, int(self.action[1]))  # Ensure p > 0
    
    def compute_production(self, I2):
        self.x = max(0, self.s2 - I2)
        return self.x
    
    @staticmethod
    def compute_reward(beta, x, shipped, k, h2, I1, I2, alpha, pi, backorders):
        transfer_revenue = beta * shipped  
        production_cost = k * (x ** 2)  # Convex cost for ALL production
        holding_cost = h2 * (I1 + I2)  # with Echelon inventory
        #holding_cost = h2 * (I2)  # without Echelon inventory
        backorder_cost = (1 - alpha) * pi * backorders
        return transfer_revenue - production_cost - holding_cost - backorder_cost

class GreedyAgent(BaseGreedyAgent):
    def __init__(self, model):
        super().__init__(model, params.action_space())

class GreedyPAgent(BaseGreedyAgent):
    def __init__(self, model):
        from itertools import product
        beta_vals = params.action_space_principal_beta()
        sigma_vals = params.action_space_principal_sigma()
        action_space = np.array(list(product(sigma_vals, beta_vals)))
        super().__init__(model, action_space)

class GreedyMAgent(MarketingAgent):
    pass

class GreedyOAgent(OperationsAgent):
    pass

class UcbAgent(BaseGreedyAgent):
    def __init__(self, model):
        super().__init__(model, params.action_space())
    
    def select_action(self):
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
