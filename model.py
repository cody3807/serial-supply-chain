import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector

from agents import GreedyAgent, GreedyMAgent, GreedyOAgent, GreedyPAgent, UcbAgent
import params


def create_agents(model, agent_types):
    """Create agents given list of agent types."""
    for i in range(len(agent_types)):
        if agent_types[i] == "greedy_m":
            GreedyMAgent.create_agents(model, n=1)
        elif agent_types[i] == "ucb":
            UcbAgent.create_agents(model, n=1)
        elif agent_types[i] == "greedy_p":
            GreedyPAgent.create_agents(model, n=1)
        elif agent_types[i] == "greedy_o":
            GreedyOAgent.create_agents(model, n=1)

class TwoStageSupplyChainModel(Model):
    def __init__(self, agent_types=("greedy_p", "greedy_m","greedy_o")):
        super().__init__()

        # Simulation & learning
        self.rng = np.random.default_rng(params.SEED)
        self.t = 0

        # Initialize agents
        create_agents(self, agent_types)

        self.rewards = {a: 0.0 for a in self.agents}

        # Inventory positions
        # I1: Operation Inventory, I2: Marketing Inventory
        self.I1 = self.I2 = 0
        self.B1 = self.B2 = 0
        self.U1_prev = self.U2_prev = 0

        # Reporters
        self.last_total_cost = 0.0
        self.joint_reg_opt = 0.0
        self.joint_reg_opt_cum = 0.0

        # Initialize DataCollector
        self.datacollector = DataCollector(
            model_reporters={
                "Total Cost": "last_total_cost",
                "Joint Regret Opt": "joint_reg_opt",
                "Cumulative Regret Opt": "joint_reg_opt_cum",
            },
            agent_reporters={
                "Base Stock Level": "action",
                "Reward": "reward",
                "Cumulative Reward": "reward_cum",
            },
        )

    # Inventory transitions
    def env_step(self, s1_loc, s2_loc):
        """
        Simulates one period of the 2-stage serial supply chain env given chosen base stock policies.
        Returns negative inventory costs as feedback signal for each agent.
        """
        I1, I2, B1, B2 = self.I1, self.I2, self.B1, self.B2
        U1_prev, U2_prev = self.U1_prev, self.U2_prev

        # (1) arrivals
        I1 += U1_prev
        I2 += U2_prev

        # (2) local IP
        IP1 = I1 - B1
        IP2 = I2 - B2

        # (3) order-up-to
        O1 = max(0, s1_loc - IP1)
        O2 = max(0, s2_loc - IP2)

        # (4) releases
        ship = min(I2, B2 + O1)
        I2 -= ship
        B2 = B2 + O1 - ship
        U1 = ship
        U2 = O2

        # (5) demand
        D = params.sample_demand(self.rng,p=10)
        sales = min(I1, B1 + D)
        I1 -= sales
        B1 = B1 + D - sales

        # (6) costs & rewards
        H1 = (params.H1 + params.H2) * I1 + params.ALPHA * params.P_BO * B1
        H2 = params.H2 * (I2 + U1) + (1.0 - params.ALPHA) * params.P_BO * B1
        total_cost = float(H1 + H2)

        # Commit
        self.I1, self.I2 = I1, I2
        self.B1, self.B2 = B1, B2
        self.U1_prev, self.U2_prev = U1, U2
        self.last_total_cost = total_cost

        return -float(H1), -float(H2)

    # Simulate one round
    def step(self):

        # (1) Agents choose base stock levels
        #self.agents.shuffle_do("select_action")
        self.agents[0].select_action()
        sigma_principal= self.agents[0].action[0]
        beta_principal= self.agents[0].action[1]
        self.agents[1].select_action()
        self.agents[2].select_action()
        # Get selected actions from agents
        #1 marketing ,2 operations
        s1,p = int(self.agents[1].action[0]),int(self.agents[1].action[1])
        s2,x = int(self.agents[2].action[0]),int(self.agents[2].action[1])

        # (2) Market dynamics
        r1, r2 = self.env_step(s1, s2)

        # (3) Agents receive reward signals
        self.rewards[self.agents[0]] = r1
        self.agents[0].reward = r1
        self.rewards[self.agents[1]] = r2
        self.agents[1].reward = r2

        # (4) Learning
        self.agents.shuffle_do("update_belief")
        self.t += 1

        # Agent reporters
        self.agents[0].reward_cum = self.agents[0].reward_cum + r1
        self.agents[1].reward_cum = self.agents[1].reward_cum + r2

        # Model reporters
        self.joint_reg_opt = self.last_total_cost - params.CTOT_OPT
        self.joint_reg_opt_cum += self.joint_reg_opt

        # Collect data
        self.datacollector.collect(self)
