import pandas as pd 
from utils import DataChart
import os 
chart = DataChart()

# Plot das métricas obtidas no conjunto de teste
df_path = f'data/output/unet_AFM_predictions1/Unet_AFM_6_channels_metrics.csv'
df = pd.read_csv(df_path, index_col=0)
final_df_melt_general = pd.melt(df, id_vars=['Process Date','Model'], value_vars=['Precision', 'Recall', 'F1', 'Dice'], var_name = 'metrics', value_name = 'scores')    
    
fig = chart.box_plot(final_df_melt_general, x='Model', y='scores', color = 'metrics', title='Unet_6_channels_final_model')
fig.write_image(f'unet_AFM_6_channels_final_metrics.png')
