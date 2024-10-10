import pandas as pd 
from utils import DataChart
import os 
chart = DataChart()


df_metrics_list = []
# df_metrics_general_validation = []
layers_list = ['100', '200']
title = 'half_unet_afm_2_channels_only_optical_diff_train_size'
for fold in layers_list:
    
    df_path = f'test_metrics_half_unet_afm_2_channels_only_optical_{fold}_images.csv'
    replace_name = f'half_unet_afm_1_channels_only_AFM_CosHeightSum_{fold}_images'
  
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
    
fig = chart.box_plot(final_df, x='Model', y='scores', color = 'metrics', title=title,
                     width=800, height = 600,)

fig2 = chart.box_plot(final_df, x='Model', y='scores', color = 'metrics',
                      title='', x_label=False, y_label=False,
                      width=800, height = 600,)

# fig.show()
fig.write_image(f'{title}.png')
fig2.write_image(f'{title}.svg')
