import pandas as pd 
from utils import DataChart
import os 
chart = DataChart()

# Plot das métricas obtidas no conjunto de teste
df_path = f'validation_metrics_UNet_AFM_6_channels_5_layers.csv'
df = pd.read_csv(df_path, index_col=0)
# final_df_melt_general = pd.melt(df, id_vars=['Process Date','Model'], value_vars=['Precision', 'Recall', 'F1', 'Dice'], var_name = 'metrics', value_name = 'scores')    
    
fig = chart.box_plot(df, x='Model', y='scores', color = 'metrics', title='UNet_AFM_5_channels_3_layers')
fig.write_image(f'UNet_AFM_6_channels_5_layers.png')

fig2 = chart.box_plot(df, x='Model', y='scores', color = 'metrics', title='', x_label=False, y_label=False)
fig2.write_image(f'UNet_AFM_6_channels_5_layers.svg')