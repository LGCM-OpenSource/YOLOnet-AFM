import pandas as pd 
from utils import Charts
import os 
chart = Charts(width=800, height = 500)


df_metrics_list = []

layers_list = [
                
                'test_metrics_unet_afm_2_channels_only_optical_15_samples_stardist_mask_stardist_mask.csv',
                'test_metrics_unet_afm_2_channels_only_optical_30_samples_stardist_mask_stardist_mask.csv',
                'test_metrics_unet_afm_2_channels_only_optical_60_samples_stardist_mask_stardist_mask.csv',
                'test_metrics_unet_afm_2_channels_only_optical_120_samples_stardist_mask_stardist_mask.csv',
                'test_metrics_unet_afm_2_channels_only_optical_234_samples_stardist_mask_stardist_mask.csv'
                
               ]


# layers_list = [f'validation_cv_metrics_fold_{i+1}_unet_afm_crossval_1_channels_only_afm_cosHeightSum_thresh_erode_CORRECTED.csv' for i in range(10)]

# title_chart = 'Unet Optico + AFM: AFM like YOLO (learning curve) balanced (15 - 240)' # unet_AFM_6_channels_diff_layers
# title_chart = 'Unet Only AFM: CHS feature (learning curve) (15 - 234)'
title_chart = 'Test2'
for fold in layers_list:
    df_path = fold
    df = pd.read_csv(df_path, index_col=0)
    # df_metrics_general_validation.append(df)

    df = df.replace({'Model':'unet'},fold[13:-4])
    if fold == 'test_metrics_half_unet_afm_1_channels_only_AFM_CosHeightSum_50_images.csv':
        df = df.replace({'Model':fold[13:-4]},f'half_unet_afm_1_channels_only_AFM_CosHeightSum_400_selected_images')
        
    # if fold == 'test_metrics_unet_afm_pp_1_channels_only_afm_cosHeightSum_thresh_erode.csv':
    #     df = df.replace({'Model':'unet'},f'Unet++_AFM')
        
    # if fold == 'test_metrics_half_unet_afm_batch_normalization_1_channels_cosHeightSum_thresh_erode.csv':
    #     df = df.replace({'Model':'unet'},f'Half_Unet_AFM')
        
    df_metrics_list.append(df)
        
# final_df_general_validation = pd.concat(df_metrics_general_validation, axis=0)
# df_cosHeightSum_final = pd.read_csv('test_metrics_UNet_AFM_1_channels_cosHeightSum_thresh_erode.csv', index_col=0)
# df_cosHeightSum_final = df_cosHeightSum_final.replace({'Model':'unet'},'Final Model')
# df_metrics_list.append(df_cosHeightSum_final)
final_df = pd.concat(df_metrics_list, axis=0)


# final_df_melt_general = pd.melt(df, id_vars=['Process Date','Model'], value_vars=['Precision', 'Recall', 'F1', 'Dice'], var_name = 'metrics', value_name = 'scores')    
# final_df_melt = pd.melt(final_df, id_vars=['Process Date','Model'], value_vars=['Precision', 'Recall', 'F1', 'Dice'], var_name = 'metrics', value_name = 'scores')
    
fig = chart.box_plot(final_df, x='Model', y='scores', color = 'metrics',
                     title='', 
                     show_x_labels=False, show_y_labels=False)

fig2 = chart.box_plot(final_df, x='Model', y='scores', color = 'metrics',
                      width=800, height = 600,
                      title=title_chart)

# fig3 = chart.violin_plot(final_df, x='Model', y='scores', color = 'metrics',
#                       width=800, height = 600,
#                       title=title_chart)

fig.write_image(f'{title_chart}.svg')
# fig2.write_image(f'{title_chart}.png')






