import pandas as pd 
from utils import DataChart
import os 
chart = DataChart()


df_path = f'data\\output\\unet_AFM_predictions\\Unet_AFM_metrics.csv'
df_path_virids = 'data_viridis\\output\\unet_AFM_predictions_viridis\\Unet_AFM_metrics.csv'

df = pd.read_csv(df_path, index_col=0)
df_virids = pd.read_csv(df_path_virids, index_col=0)

df = df.replace({'Model':'unet_prediction'},f'unet_afm_6_channels')
df_virids = df_virids.replace({'Model':'unet_prediction'},f'unet_afm_6_channels_viridis')
df_metrics_list = [df, df_virids]

# df_metrics_list.append(df)
final_df = pd.concat(df_metrics_list, axis=0)
# final_df = df.loc[(df!=0).all(axis=1)]    
final_df_melt = pd.melt(final_df, id_vars=['Model'], value_vars=['Precision','Recall','F1','Dice'], var_name = 'metrics', value_name = 'scores')
    
fig = chart.box_plot(final_df_melt, x='Model', y='scores', color = 'metrics', title='')
# fig.show()
fig.write_image(f'unet_AFM_6_channels_vs_unet_AFM_6_channels_viridis.png')