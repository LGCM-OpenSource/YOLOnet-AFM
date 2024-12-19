import pandas as pd 
from utils import DataChart
import os 
chart = DataChart()

# plot métricas de monitoramento durante o treinamento
df_path = f'models{os.sep}training_final_model.log'
df = pd.read_csv(df_path, index_col=0).reset_index()
df_melt = pd.melt(df, id_vars=['epoch'], value_vars=['precision', 'recall', 'loss', 'dice_coef'], var_name = 'metrics', value_name = 'scores')    

fig = chart.line_plot(df_melt, x='epoch', y='scores', color = 'metrics', title=f'Training metrics: Unet_AFM_6_channels_final_model')
fig.write_image(f'Unet_AFM_6_channels_final_model.png')
    
    