import pandas as pd 
from utils import DataChart
import os 
chart = DataChart()


df_metrics_list = []
# df_metrics_general_validation = []
layers_list = ['50_images', '100_images', '150_images', '200_images', '226_images']

for fold in layers_list:
    
    df_path = f'validation_metrics_unet_afm_6_channels_{fold}.csv'
    replace_name = f'Unet_AFM_6_channels_{fold}'
  
    df = pd.read_csv(df_path, index_col=0)
    # df_metrics_general_validation.append(df)
    df = df.replace({'Model':'unet'},replace_name)
    # if fold == '4':
    #     df = df.replace({'Model':'unet_prediction'},f'Unet_AFM_6_channels_final_model')
        
    df_metrics_list.append(df)
        
# final_df_general_validation = pd.concat(df_metrics_general_validation, axis=0)
final_df = pd.concat(df_metrics_list, axis=0)


# final_df_melt_general = pd.melt(df, id_vars=['Process Date','Model'], value_vars=['Precision', 'Recall', 'F1', 'Dice'], var_name = 'metrics', value_name = 'scores')    
# final_df_melt = pd.melt(final_df, id_vars=['Process Date','Model'], value_vars=['Precision', 'Recall', 'F1', 'Dice'], var_name = 'metrics', value_name = 'scores')
    
fig = chart.box_plot(final_df, x='Model', y='scores', color = 'metrics', title='Unet_AFM_6_channels_diff_train_size')

fig2 = chart.box_plot(final_df, x='Model', y='scores', color = 'metrics',
                      title='', x_label=False, y_label=False)

# fig.show()
fig.write_image(f'unet_AFM_6_channels_diff_train_size.png')
fig2.write_image(f'unet_AFM_6_channels_diff_train_size_without_legends.svg')
