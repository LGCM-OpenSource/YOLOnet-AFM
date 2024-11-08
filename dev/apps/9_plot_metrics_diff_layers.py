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
#             #    'AFM_mask_validation_artifacts.csv',
#             #    'AFM_mask_validation_afm_optico_ok.csv',
#             #    'AFM_mask_validation_corrected_cosHeightSum_thresh_245_255_k5.csv'
#             #    'test_metrics_unet_afm_2_channels_only_optical.csv',
               
#             #    'test_metrics_unet_afm_1_channels_only_afm_plannedHeight.csv',
#             #    'test_metrics_unet_afm_2_channels_only_afm_plannedHeight_maxpos1500.csv',
#             #    'test_metrics_UNet_AFM_1_channels_cosHeightSum_thresh_erode.csv',
               
#             #    'test_metrics_unet_afm_1_channels_caseC_pure.csv',
#             #    'test_metrics_unet_afm_2_channels_caseC_separe_height_maxpos1500.csv',
#             #    'test_metrics_unet_afm_2_channels_caseC_with_CosHeightSum.csv',
#             #    'test_metrics_unet_afm_2_channels_caseC_with_height.csv',
#             #    'test_metrics_unet_afm_1_channels_cosHeightSum.csv'
#             #    'test_metrics_unet_afm_3_channels_cosHeightSum_optical.csv',
#             #    'test_metrics_unet_afm_2_channels_yolo_simulator_by_afm.csv',
            
#             'test_metrics_unet_afm_1_channels_cosHeightSum_thresh_erode_2_layers.csv',
#             'test_metrics_unet_afm_1_channels_cosHeightSum_thresh_erode_3_layers.csv',
            # 'test_metrics_unet_afm_pp_1_channels_only_afm_cosHeightSum_thresh_erode.csv',
            # 'test_metrics_half_unet_afm_batch_normalization_1_channels_cosHeightSum_thresh_erode.csv',
            # 'test_metrics_UNet_AFM_1_channels_cosHeightSum_thresh_erode.csv',
#             'test_metrics_unet_afm_1_channels_cosHeightSum_thresh_erode_5_layers.csv',
#             # 'AFM_mask_validation_from_test_dataset_only_afm_cosHeightSum_thresh_erode.csv'

                # 'test_metrics_half_unet_afm_2_channels_like_yolo_opt_afm_100_images.csv',
                # 'test_metrics_half_unet_afm_2_channels_like_yolo_opt_afm_200_images.csv',
                
                # 'test_metrics_half_unet_afm_2_channels_only_optical_25_images_2.csv',
                # 'test_metrics_half_unet_afm_2_channels_only_optical_50_images_2.csv',
                # 'test_metrics_half_unet_afm_2_channels_only_optical_100_images_2.csv',
                # 'test_metrics_half_unet_afm_2_channels_only_optical_200_images_2.csv',
                
                # 'test_metrics_half_unet_afm_2_channels_like_yolo_opt_afm_25_images_2.csv',
                # 'test_metrics_half_unet_afm_2_channels_like_yolo_opt_afm_50_images_2.csv',
                # 'test_metrics_half_unet_afm_2_channels_like_yolo_opt_afm_100_images_2.csv',
                # 'test_metrics_half_unet_afm_2_channels_like_yolo_opt_afm_200_images_2.csv'
                
                
                
                # 'test_metrics_unet_afm_1_channels_only_AFM_CosHeightSum_15_samples_stardist_mask_stardist_mask.csv',
                # 'test_metrics_unet_afm_1_channels_only_AFM_CosHeightSum_30_samples_stardist_mask_stardist_mask.csv',
                # 'test_metrics_unet_afm_1_channels_only_AFM_CosHeightSum_60_samples_stardist_mask_stardist_mask.csv',
                # 'test_metrics_unet_afm_1_channels_only_AFM_CosHeightSum_120_samples_stardist_mask_stardist_mask.csv',
                # 'test_metrics_unet_afm_1_channels_only_AFM_CosHeightSum_234_samples_stardist_mask_stardist_mask.csv'
                 
                # 'test_metrics_unet_afm_2_channels_like_yolo_opt_afm_15_samples_stardist_mask_stardist_mask.csv',
                # 'test_metrics_unet_afm_2_channels_like_yolo_opt_afm_30_samples_stardist_mask_stardist_mask.csv',
                # 'test_metrics_unet_afm_2_channels_like_yolo_opt_afm_60_samples_stardist_mask_stardist_mask.csv',
                # 'test_metrics_unet_afm_2_channels_like_yolo_opt_afm_120_samples_stardist_mask_stardist_mask.csv',
                # 'test_metrics_unet_afm_2_channels_like_yolo_opt_afm_234_samples_stardist_mask_stardist_mask.csv'
                
                'test_metrics_unet_afm_2_channels_only_optical_15_samples_stardist_mask_stardist_mask.csv',
                'test_metrics_unet_afm_2_channels_only_optical_30_samples_stardist_mask_stardist_mask.csv',
                'test_metrics_unet_afm_2_channels_only_optical_60_samples_stardist_mask_stardist_mask.csv',
                'test_metrics_unet_afm_2_channels_only_optical_120_samples_stardist_mask_stardist_mask.csv',
                'test_metrics_unet_afm_2_channels_only_optical_234_samples_stardist_mask_stardist_mask.csv'
                
               ]


# layers_list = [f'validation_cv_metrics_fold_{i+1}_unet_afm_crossval_1_channels_only_afm_cosHeightSum_thresh_erode_CORRECTED.csv' for i in range(10)]

# title_chart = 'Unet Optico + AFM: AFM like YOLO (learning curve) balanced (15 - 240)' # unet_AFM_6_channels_diff_layers
# title_chart = 'Unet Only AFM: CHS feature (learning curve) (15 - 234)'
title_chart = 'Unet Only Optico: Blue and Hist Equalize features (learning curve)(15 - 240)'
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
                     width=800, height = 500,
                     title='', 
                     x_label=False, y_label=False)

fig2 = chart.box_plot(final_df, x='Model', y='scores', color = 'metrics',
                      width=800, height = 600,
                      title=title_chart)

# fig3 = chart.violin_plot(final_df, x='Model', y='scores', color = 'metrics',
#                       width=800, height = 600,
#                       title=title_chart)

fig.write_image(f'{title_chart}.svg')
fig2.write_image(f'{title_chart}.png')






