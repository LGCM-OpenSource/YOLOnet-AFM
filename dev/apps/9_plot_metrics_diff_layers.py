from ydata_profiling import ProfileReport
import pandas as pd 
from utils import DataChart
import os 
chart = DataChart()


df_metrics_list = []
# df_metrics_general_validation = []
# layers_list = ['2', '3', '4', '5']
# layers_list = ['validation_metrics_unet_afm_1_channels_weight_expor_4_2.csv',
#                'validation_metrics_unet_afm_1_channels_cosHeightSum.csv',
#                'validation_metrics_unet_afm_2_channels_cosHeightSum_thresh_240_250_and_245_255_sobel_k5_figs_removed.csv',
#                'validation_metrics_unet_afm_2_channels_hist_planned.csv',
#                'validation_metrics_unet_afm_3_channels_hist_planned_maxpos.csv',
#                'validation_metrics_unet_afm_3_channels_blue_hist_weight_expo_4_2.csv',
#                'validation_metrics_unet_afm_3_channels_blue_hist_cosHeightSum.csv',
#                'validation_metrics_UNet_AFM_2_channels_only_optico.csv']


# layers_list = ['AFM_mask_validation_1_channels_cosHeightSum_thresh_240_250_k5.csv',
#                'AFM_mask_validation_1_channels_cosHeightSum_thresh_245_255_k5.csv',
#                'AFM_mask_validation_afm_optico_ok.csv',
#                'AFM_mask_validation_artifacts.csv',
#                'AFM_mask_validation_displaced.csv',
#                'AFM_mask_validation_doubts.csv',
#                'AFM_mask_validation_nucleus_problem.csv'
#                ]



layers_list = [
               'AFM_mask_validation_artifacts.csv',
               'AFM_mask_validation_afm_optico_ok.csv',
               'AFM_mask_validation_corrected_cosHeightSum_thresh_245_255_k5.csv'
               ]


title_chart = 'AFM_masks_threshold_metrics_in_nuleus_corrected_really' # unet_AFM_6_channels_diff_layers
for fold in layers_list:
    df_path = fold
    df = pd.read_csv(df_path, index_col=0)
    # df_metrics_general_validation.append(df)

    df = df.replace({'Model':'unet'},fold[20:-4])
    if fold == 'test_metrics_unet_afm_6_channels_final_model.csv':
        df = df.replace({'Model':f'{fold[13:-4]}'},f'UNet_AFM_6_channels_4_layers_final_model')
        
    df_metrics_list.append(df)
        
# final_df_general_validation = pd.concat(df_metrics_general_validation, axis=0)
final_df = pd.concat(df_metrics_list, axis=0)


# final_df_melt_general = pd.melt(df, id_vars=['Process Date','Model'], value_vars=['Precision', 'Recall', 'F1', 'Dice'], var_name = 'metrics', value_name = 'scores')    
# final_df_melt = pd.melt(final_df, id_vars=['Process Date','Model'], value_vars=['Precision', 'Recall', 'F1', 'Dice'], var_name = 'metrics', value_name = 'scores')
    
fig = chart.box_plot(final_df, x='Model', y='scores', color = 'metrics',
                     width=800, height = 500,
                     title='', 
                     x_label=False, y_label=False)

fig2 = chart.box_plot(final_df, x='Model', y='scores', color = 'metrics',
                      width=800, height = 600,
                      title=title_chart)

fig.write_image(f'{title_chart}.svg')
fig2.write_image(f'{title_chart}.png')



