# Imports 
from model import TwoStageSupplyChainModel
import params
import pandas as pd
from datetime import datetime

# Example model 
model = TwoStageSupplyChainModel(agent_types=("ucb_p","ucb_m","ucb_o"))
for _ in range(params.ROUNDS):
    model.step()

# Pull Data
df_model = model.datacollector.get_model_vars_dataframe()
df_agents = model.datacollector.get_agent_vars_dataframe()

# Export to Excel
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"simulation_results_{timestamp}.xlsx"

with pd.ExcelWriter(filename, engine='openpyxl') as writer:
    df_model.to_excel(writer, sheet_name='Model Data')
    df_agents.to_excel(writer, sheet_name='Agent Data')

print(f'Results exported to: {filename}')