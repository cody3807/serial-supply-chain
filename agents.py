import numpy as np
from scipy.stats import poisson
from mesa import Agent
import params 

class GreedyAgent(Agent):
    """ε-greedy bandit agent"""

    def __init__(self, model):
        """Initialize agent"""
        super().__init__(model)

        # Action space 
        self.action_space = params.action_space()
        self.n_actions = len(self.action_space)

        # Bandit statistics
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.average_reward = np.zeros(self.n_actions, dtype=float)

        # Epsilon for ε-greedy (global decay function)
        self.eps = params.epsilon_at(model.t, params.ROUNDS)

        # Selected action + reward
        self.action_idx = None
        self.action = None
        self.reward = 0.0

        # Evaluation
        self.reward_cum = 0.0
        self.regret_nash = 0.0
        self.regret_nash_cum = 0

    def select_action(self):
        """
        Select an action using ε-greedy
        Update selected action attributes (index and quantity)
        """
        # Exploration
        if self.random.random() < self.eps:
            action_idx = self.random.randrange(self.n_actions)

        # Exploitation
        else:
            action_idx = int(np.argmax(self.average_reward))

        # Store selections
        self.action_idx = action_idx
        self.action = int(self.action_space[action_idx])


    def update_belief(self):
        """
        Update reward estimates using incremental mean update:
            μ_new = μ_old + (r - μ_old) / n
        Update ε for the next round t+1
        """
        a_idx = self.action_idx

        # Reward from environment
        r = self.model.rewards[self]         

        # Incremental update
        n = self.counts[a_idx] + 1
        self.counts[a_idx] = n
        self.average_reward[a_idx] += (r - self.average_reward[a_idx]) / n

        # Update epsilon for the next round t+1
        self.eps = params.epsilon_at(self.model.t + 1, params.ROUNDS)

class UcbAgent(Agent):
    """UCB1 bandit agent"""

    def __init__(self, model):
        super().__init__(model)

        # Action space
        self.action_space = params.action_space()
        self.n_actions = len(self.action_space)

        # Estimates
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.average_reward = np.zeros(self.n_actions, dtype=float)

        # Selected action + reward
        self.action_idx = None
        self.action = None
        self.reward = 0.0

        # Evaluation
        self.reward_cum = 0.0
        self.regret_nash = 0.0
        self.regret_nash_cum = 0.0

    def select_action(self):
        """
        Select an action using UCB1
        Update selected action attributes (index and quantity)
        """
        
        # Ensure each arm is tried at least once
        untried = np.flatnonzero(self.counts == 0)
        
        if len(untried) > 0:
            # Select untried arm, order doesn’t matter
            action_idx = int(untried[0]) 
        else:
            # Select arm with highest UCB
            confidence_bound = np.sqrt((2.0 * np.log(params.ROUNDS)) / self.counts)
            ucb = self.average_reward + confidence_bound
            best = np.flatnonzero(ucb == ucb.max())
            action_idx = int(self.random.choice(best))

        self.action_idx = action_idx
        self.action = int(self.action_space[action_idx])

    def update_belief(self):
        """
        Update reward estimates using incremental mean update:
            μ_new = μ_old + (r - μ_old) / n
        """
        a_idx = self.action_idx

        # Reward from environment 
        r = float(self.model.rewards[self])

        # Incremental updates
        n = self.counts[a_idx] + 1
        self.counts[a_idx] = n
        self.average_reward[a_idx] += (r - self.average_reward[a_idx]) / n


class GreedyPAgent(Agent):
    """ε-greedy principal agent"""

    def __init__(self, model):
        """Initialize agent"""
        super().__init__(model)

        # Action space 
        self.action_space = params.action_space_principal()
        self.n_actions = len(self.action_space)

        # Bandit statistics
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.average_reward = np.zeros(self.n_actions, dtype=float)

        # Epsilon for ε-greedy (global decay function)
        self.eps = params.epsilon_at(model.t, params.ROUNDS)

        # Selected action + reward
        self.action_idx = None
        self.action = None
        self.reward = 0.0

        # Evaluation
        self.reward_cum = 0.0
        self.regret_nash = 0.0
        self.regret_nash_cum = 0

    def select_action(self):
        """
        Select an action using ε-greedy
        Update selected action attributes (index and quantity)
        """
        # Exploration
        if self.random.random() < self.eps:
            action_idx = self.random.randrange(self.n_actions)

        # Exploitation
        else:
            action_idx = int(np.argmax(self.average_reward))

        # Store selections
        self.action_idx = action_idx
        self.action = self.action_space[action_idx]


    def update_belief(self):
        """
        Update reward estimates using incremental mean update:
            μ_new = μ_old + (r - μ_old) / n
        Update ε for the next round t+1
        """
        a_idx = self.action_idx

        # Reward from environment
        r = self.model.rewards[self]         

        # Incremental update
        n = self.counts[a_idx] + 1
        self.counts[a_idx] = n
        self.average_reward[a_idx] += (r - self.average_reward[a_idx]) / n

        # Update epsilon for the next round t+1
        self.eps = params.epsilon_at(self.model.t + 1, params.ROUNDS)


class GreedyMAgent(Agent):
    """ε-greedy bandit agent"""

    def __init__(self, model):
        """Initialize agent"""
        super().__init__(model)

        # Action space 
        self.action_space = params.action_space_marketing()
        self.n_actions = len(self.action_space)

        # Bandit statistics
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.average_reward = np.zeros(self.n_actions, dtype=float)

        # Epsilon for ε-greedy (global decay function)
        self.eps = params.epsilon_at(model.t, params.ROUNDS)

        # Selected action + reward
        self.action_idx = None
        self.action = None
        self.reward = 0.0

        # Evaluation
        self.reward_cum = 0.0
        self.regret_nash = 0.0
        self.regret_nash_cum = 0

    def select_action(self):
        """
        Select an action using ε-greedy
        Update selected action attributes (index and quantity)
        """
        # Exploration
        if self.random.random() < self.eps:
            action_idx = self.random.randrange(self.n_actions)

        # Exploitation
        else:
            action_idx = int(np.argmax(self.average_reward))

        # Store selections
        self.action_idx = action_idx
        self.action = self.action_space[action_idx]


    def update_belief(self):
        """
        Update reward estimates using incremental mean update:
            μ_new = μ_old + (r - μ_old) / n
        Update ε for the next round t+1
        """
        a_idx = self.action_idx

        # Reward from environment
        r = self.model.rewards[self]         

        # Incremental update
        n = self.counts[a_idx] + 1
        self.counts[a_idx] = n
        self.average_reward[a_idx] += (r - self.average_reward[a_idx]) / n

        # Update epsilon for the next round t+1
        self.eps = params.epsilon_at(self.model.t + 1, params.ROUNDS)



class GreedyOAgent(Agent):
    """ε-greedy bandit agent"""

    def __init__(self, model):
        """Initialize agent"""
        super().__init__(model)

        # Action space 
        self.action_space = params.action_space_operation()
        self.n_actions = len(self.action_space)

        # Bandit statistics
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.average_reward = np.zeros(self.n_actions, dtype=float)

        # Epsilon for ε-greedy (global decay function)
        self.eps = params.epsilon_at(model.t, params.ROUNDS)

        # Selected action + reward
        self.action_idx = None
        self.action = None
        self.reward = 0.0

        # Evaluation
        self.reward_cum = 0.0
        self.regret_nash = 0.0
        self.regret_nash_cum = 0

    def select_action(self):
        """
        Select an action using ε-greedy
        Update selected action attributes (index and quantity)
        """
        # Exploration
        if self.random.random() < self.eps:
            action_idx = self.random.randrange(self.n_actions)

        # Exploitation
        else:
            action_idx = int(np.argmax(self.average_reward))

        # Store selections
        self.action_idx = action_idx
        self.action =self.action_space[action_idx]


    def update_belief(self):
        """
        Update reward estimates using incremental mean update:
            μ_new = μ_old + (r - μ_old) / n
        Update ε for the next round t+1
        """
        a_idx = self.action_idx

        # Reward from environment
        r = self.model.rewards[self]         

        # Incremental update
        n = self.counts[a_idx] + 1
        self.counts[a_idx] = n
        self.average_reward[a_idx] += (r - self.average_reward[a_idx]) / n

        # Update epsilon for the next round t+1
        self.eps = params.epsilon_at(self.model.t + 1, params.ROUNDS)



class UcbPAgent(Agent):
    """UCB1 bandit agent"""

    def __init__(self, model):
        super().__init__(model)

        # Action space
        self.action_space = params.action_space_principal()
        self.n_actions = len(self.action_space)

        # Estimates
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.average_reward = np.zeros(self.n_actions, dtype=float)

        # Selected action + reward
        self.action_idx = None
        self.action = None
        self.reward = 0.0

        # Evaluation
        self.reward_cum = 0.0
        self.regret_nash = 0.0
        self.regret_nash_cum = 0.0

    def select_action(self):
        """
        Select an action using UCB1
        Update selected action attributes (index and quantity)
        """
        
        # Ensure each arm is tried at least once
        untried = np.flatnonzero(self.counts == 0)
        
        if len(untried) > 0:
            # Select untried arm, order doesn’t matter
            action_idx = int(untried[0]) 
        else:
            # Select arm with highest UCB
            confidence_bound = np.sqrt((2.0 * np.log(params.ROUNDS)) / self.counts)
            ucb = self.average_reward + confidence_bound
            best = np.flatnonzero(ucb == ucb.max())
            action_idx = int(self.random.choice(best))

        self.action_idx = action_idx
        self.action = self.action_space[action_idx]

    def update_belief(self):
        """
        Update reward estimates using incremental mean update:
            μ_new = μ_old + (r - μ_old) / n
        """
        a_idx = self.action_idx

        # Reward from environment 
        r = float(self.model.rewards[self])

        # Incremental updates
        n = self.counts[a_idx] + 1
        self.counts[a_idx] = n
        self.average_reward[a_idx] += (r - self.average_reward[a_idx]) / n


class UcbOAgent(Agent):
    """UCB1 bandit agent"""

    def __init__(self, model):
        super().__init__(model)

        # Action space
        self.action_space = params.action_space_operation()
        self.n_actions = len(self.action_space)

        # Estimates
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.average_reward = np.zeros(self.n_actions, dtype=float)

        # Selected action + reward
        self.action_idx = None
        self.action = None
        self.reward = 0.0

        # Evaluation
        self.reward_cum = 0.0
        self.regret_nash = 0.0
        self.regret_nash_cum = 0.0

    def select_action(self):
        """
        Select an action using UCB1
        Update selected action attributes (index and quantity)
        """
        
        # Ensure each arm is tried at least once
        untried = np.flatnonzero(self.counts == 0)
        
        if len(untried) > 0:
            # Select untried arm, order doesn’t matter
            action_idx = int(untried[0]) 
        else:
            # Select arm with highest UCB
            confidence_bound = np.sqrt((2.0 * np.log(params.ROUNDS)) / self.counts)
            ucb = self.average_reward + confidence_bound
            best = np.flatnonzero(ucb == ucb.max())
            action_idx = int(self.random.choice(best))

        self.action_idx = action_idx
        self.action = self.action_space[action_idx]

    def update_belief(self):
        """
        Update reward estimates using incremental mean update:
            μ_new = μ_old + (r - μ_old) / n
        """
        a_idx = self.action_idx

        # Reward from environment 
        r = float(self.model.rewards[self])

        # Incremental updates
        n = self.counts[a_idx] + 1
        self.counts[a_idx] = n
        self.average_reward[a_idx] += (r - self.average_reward[a_idx]) / n



class UcbMAgent(Agent):
    """UCB1 bandit agent"""

    def __init__(self, model):
        super().__init__(model)

        # Action space
        self.action_space = params.action_space_marketing()
        self.n_actions = len(self.action_space)

        # Estimates
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.average_reward = np.zeros(self.n_actions, dtype=float)

        # Selected action + reward
        self.action_idx = None
        self.action = None
        self.reward = 0.0

        # Evaluation
        self.reward_cum = 0.0
        self.regret_nash = 0.0
        self.regret_nash_cum = 0.0

    def select_action(self):
        """
        Select an action using UCB1
        Update selected action attributes (index and quantity)
        """
        
        # Ensure each arm is tried at least once
        untried = np.flatnonzero(self.counts == 0)
        
        if len(untried) > 0:
            # Select untried arm, order doesn’t matter
            action_idx = int(untried[0]) 
        else:
            # Select arm with highest UCB
            confidence_bound = np.sqrt((2.0 * np.log(params.ROUNDS)) / self.counts)
            ucb = self.average_reward + confidence_bound
            best = np.flatnonzero(ucb == ucb.max())
            action_idx = int(self.random.choice(best))

        self.action_idx = action_idx
        self.action = self.action_space[action_idx]

    def update_belief(self):
        """
        Update reward estimates using incremental mean update:
            μ_new = μ_old + (r - μ_old) / n
        """
        a_idx = self.action_idx

        # Reward from environment 
        r = float(self.model.rewards[self])

        # Incremental updates
        n = self.counts[a_idx] + 1
        self.counts[a_idx] = n
        self.average_reward[a_idx] += (r - self.average_reward[a_idx]) / n
