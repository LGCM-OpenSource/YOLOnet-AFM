import pandas as pd 
from utils import DataChart
import os 
chart = DataChart()


df_metrics_list = []
# df_metrics_general_validation = []
layers_list = ['50_images', '100_images', '150_images', '200_images', 'Unet_AFM_6_channels_4_layers_metrics.csv']
for fold in layers_list:
    df_path = f'unet_AFM_metrics_{fold}.csv'
    replace_name = f'Unet_AFM_6_channels_{fold}'
    if fold == 'Unet_AFM_6_channels_4_layers_metrics.csv':
        df_path = fold
        replace_name = f'Unet_AFM_6_channels_226_images'
        
        
    df = pd.read_csv(df_path, index_col=0)
    # df_metrics_general_validation.append(df)
    df = df.replace({'Model':'unet_prediction'},replace_name)
    # if fold == '4':
    #     df = df.replace({'Model':'unet_prediction'},f'Unet_AFM_6_channels_final_model')
        
    df_metrics_list.append(df)
        
# final_df_general_validation = pd.concat(df_metrics_general_validation, axis=0)
final_df = pd.concat(df_metrics_list, axis=0)


# final_df_melt_general = pd.melt(df, id_vars=['Process Date','Model'], value_vars=['Precision', 'Recall', 'F1', 'Dice'], var_name = 'metrics', value_name = 'scores')    
final_df_melt = pd.melt(final_df, id_vars=['Process Date','Model'], value_vars=['Precision', 'Recall', 'F1', 'Dice'], var_name = 'metrics', value_name = 'scores')
    
# fig = chart.box_plot(final_df_melt_general, x='Model', y='scores', color = 'metrics', title='Unet_6_channels_cross_validation_general')
fig2 = chart.box_plot(final_df_melt, x='Model', y='scores', color = 'metrics',
                      title='')

# fig.show()
# fig.write_image(f'unet_AFM_6_channels_final_metrics.png')
fig2.write_image(f'unet_AFM_6_channels_diff_train_size_without_legends.svg')
