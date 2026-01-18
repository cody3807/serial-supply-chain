# Imports 
from model import TwoStageSupplyChainModel
import params

# Example model 
model = TwoStageSupplyChainModel(agent_types=("greedy","greedy"))
for _ in range(params.ROUNDS):
    model.step()

# Pull Data
df_model = model.datacollector.get_model_vars_dataframe()
df_agents = model.datacollector.get_agent_vars_dataframe()
print('abc')