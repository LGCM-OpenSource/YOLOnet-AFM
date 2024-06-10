import pandas as pd 
from utils import DataChart
import os 
chart = DataChart()


df_metrics_list = []
# df_metrics_general_validation = []
layers_list = ['2', '3', '4', '5']
for fold in layers_list:
    df_path = f'Unet_AFM_6_channels_{fold}_layers_metrics.csv'
    df = pd.read_csv(df_path, index_col=0)
    # df_metrics_general_validation.append(df)
    df = df.replace({'Model':'unet_prediction'},f'Unet_AFM_6_channels_{fold}_layers_metrics')
    if fold == '4':
        df = df.replace({'Model':'unet_prediction'},f'Unet_AFM_6_channels_final_model')
        
    df_metrics_list.append(df)
        
# final_df_general_validation = pd.concat(df_metrics_general_validation, axis=0)
final_df = pd.concat(df_metrics_list, axis=0)


# final_df_melt_general = pd.melt(df, id_vars=['Process Date','Model'], value_vars=['Precision', 'Recall', 'F1', 'Dice'], var_name = 'metrics', value_name = 'scores')    
final_df_melt = pd.melt(final_df, id_vars=['Process Date','Model'], value_vars=['Precision', 'Recall', 'F1', 'Dice'], var_name = 'metrics', value_name = 'scores')
    
# fig = chart.box_plot(final_df_melt_general, x='Model', y='scores', color = 'metrics', title='Unet_6_channels_cross_validation_general')
fig2 = chart.box_plot(final_df_melt, x='Model', y='scores', color = 'metrics',
                      width=800, height = 500,
                      title='Unet_6_channels_diff_layers')

# fig.show()
# fig.write_image(f'unet_AFM_6_channels_final_metrics.png')
fig2.write_image(f'unet_AFM_6_channels_diff_layers.png')
