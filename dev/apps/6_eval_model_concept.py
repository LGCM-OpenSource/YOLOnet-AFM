import os
import numpy as np
from utils import UnetProcess, EvalModel
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt 
import pandas as pd 


model_dict = {
    'unet_afm_1_channels_only_AFM_CosHeightSum': {
        'model': 'unet_afm_1_channels_only_AFM_CosHeightSum_NN_samples_stardist_mask.h5',
        'test_path': f'{os.sep}home{os.sep}arthur{os.sep}lgcm{os.sep}projects{os.sep}Segmentation_union_projects{os.sep}data_complete{os.sep}input{os.sep}train{os.sep}train_1_channels_only_AFM_CosHeightSum{os.sep}opt_img_training',
        'mask_path': f'{os.sep}home{os.sep}arthur{os.sep}lgcm{os.sep}projects{os.sep}Segmentation_union_projects{os.sep}data_complete{os.sep}input{os.sep}train{os.sep}train_1_channels_only_AFM_CosHeightSum{os.sep}msk_img_training',
            
    },
    'unet_afm_2_channels_like_yolo_opt_afm':{
            'model':'unet_afm_2_channels_like_yolo_opt_afm_NN_samples_stardist_mask.h5',
            'test_path':f'{os.sep}home{os.sep}arthur{os.sep}lgcm{os.sep}projects{os.sep}Segmentation_union_projects{os.sep}data_complete{os.sep}input{os.sep}train{os.sep}train_2_channels_like_yolo_opt_afm{os.sep}opt_img_training',
            'mask_path':f'{os.sep}home{os.sep}arthur{os.sep}lgcm{os.sep}projects{os.sep}Segmentation_union_projects{os.sep}data_complete{os.sep}input{os.sep}train{os.sep}train_2_channels_like_yolo_opt_afm{os.sep}msk_img_training',
            
    },
    'unet_afm_2_channels_only_optical_data_without_artifacts':{
            'model': 'unet_afm_2_channels_only_optical_NN_samples_stardist_mask.h5',
            'test_path': f'{os.sep}home{os.sep}arthur{os.sep}lgcm{os.sep}projects{os.sep}Segmentation_union_projects{os.sep}data_complete{os.sep}input{os.sep}train{os.sep}train_2_channels_only_optical{os.sep}opt_img_training',
            'mask_path': f'{os.sep}home{os.sep}arthur{os.sep}lgcm{os.sep}projects{os.sep}Segmentation_union_projects{os.sep}data_complete{os.sep}input{os.sep}train{os.sep}train_2_channels_only_optical{os.sep}msk_img_training'
    }
}

opt_image = f'{os.sep}home{os.sep}arthur{os.sep}lgcm{os.sep}projects{os.sep}Segmentation_union_projects{os.sep}data_complete{os.sep}input{os.sep}optical_images_resized'
predict_path = f'{os.sep}home{os.sep}arthur{os.sep}lgcm{os.sep}projects{os.sep}Segmentation_union_projects{os.sep}data_complete{os.sep}input{os.sep}Usefull_data'


test_files_path = f'{os.sep}home{os.sep}arthur{os.sep}lgcm{os.sep}projects{os.sep}Segmentation_union_projects{os.sep}data_complete{os.sep}datasets{os.sep}selected_with_good_optico{os.sep}Test_83_Images.tsv'
df_test_files = pd.read_csv(test_files_path)


dataset_size = [15,30,60,120,234]
for model in model_dict.keys():
        # if model != 'half_unet_afm_2_channels_only_optical_data_without_artifacts': 
        #         continue
        for i in dataset_size:
                
                model_info = model_dict[model]
                model_name = model_info['model'].replace('NN', str(i))
                
                print(f'get {model_name} validation... ')
                
                preprocess_image_path = [os.path.join(model_info['test_path'], file+'_channels_added.npy') for file in df_test_files['Process.Date'].values]
                mask_path = [os.path.join(model_info['mask_path'], file+'_channels_added.npy') for file in df_test_files['Process.Date'].values]
                opt_image_path = [os.path.join(opt_image, file+'_optico_crop_resized.png') for file in df_test_files['Process.Date'].values]
                usefull_path = [os.path.join(predict_path, file+'_UsefullData.tsv') for file in df_test_files['Process.Date'].values]

                df_list = []

                for i in tqdm(range(len(opt_image_path)), colour='#0000FF'):
                        process_data = opt_image_path[i].split(f'{os.sep}')[-1].split('_')[0]
                        unetTrat =   UnetProcess(opt_image_path[i], preprocess_image_path[i], usefull_path[i], mask_path[i], model_path=f'models{os.sep}{model_name}') 
                        
                        # usefull_path_unet = usefull_path[i].replace(f'data{os.sep}input{os.sep}Usefull_data{os.sep}',f'data{os.sep}output{os.sep}{unet_afm_fold}{os.sep}predict_sheets{os.sep}')
                        y,y_pred, _t = unetTrat.unet_predict()
                        y_flatten = y.flatten()
                        y_pred_flatten = y_pred.flatten()

                        
                        eval = EvalModel('unet',process_data, y_flatten, y_pred_flatten) 
                        scores = eval.get_metrics()
                        metric_df = eval.metrics_to_df(scores)
                        df_list.append(metric_df)
                # final_metric_df = pd.concat(df_list, axis=0)
                # final_metrics_melt = pd.melt(final_metric_df, id_vars=['Process Date','Model'], value_vars=['Precision', 'Recall', 'F1', 'Dice'], var_name = 'metrics', value_name = 'scores')
                # final_metrics_melt.to_csv(f'test_metrics_{model_name.split(".")[0]}_stardist_mask.csv')        
