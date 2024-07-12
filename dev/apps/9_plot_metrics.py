import pandas as pd 
from utils import DataChart
import os 
chart = DataChart()

# Plot das métricas obtidas no conjunto de teste
df_path = f'unet_only_optico_metrics.csv'
chart_title = 'Unet_only_optico'

df = pd.read_csv(df_path, index_col=0)
final_df_melt_general = pd.melt(df, id_vars=['Process Date','Model'], value_vars=['Precision', 'Recall', 'F1', 'Dice'], var_name = 'metrics', value_name = 'scores')    
    
fig = chart.box_plot(final_df_melt_general, x='Model', y='scores', color = 'metrics', title=f'{chart_title}')
fig.write_image(f'{chart_title}_metrics.png')

fig2 = chart.box_plot(final_df_melt_general, x='Model', y='scores', color = 'metrics', title='', x_label=False, y_label=False)
fig2.write_image(f'{chart_title}_metrics.svg')