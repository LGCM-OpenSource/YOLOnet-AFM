import os
import numpy as np
from utils import UnetProcess, EvalModel
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt 
import pandas as pd 


model_dict = {
    'half_unet_afm_1_channels_only_AFM_CosHeightSum': {
        'model': 'half_unet_afm_1_channels_only_AFM_CosHeightSum_NN_data_without_artifacts.h5',
        'test_path': f'{os.sep}home{os.sep}arthur{os.sep}lgcm{os.sep}projects{os.sep}Segmentation_union_projects{os.sep}data_complete{os.sep}intermediate{os.sep}pre_processing_afm{os.sep}image',
        'mask_path':f'{os.sep}home{os.sep}arthur{os.sep}lgcm{os.sep}projects{os.sep}Segmentation_union_projects{os.sep}data_complete{os.sep}intermediate{os.sep}pre_processing_afm{os.sep}mask',
            
    },
    'half_unet_afm_2_channels_like_yolo_opt_afm':{
            'model':'half_unet_afm_2_channels_like_yolo_opt_afm_NN_data_without_artifacts.h5',
            'test_path':f'{os.sep}home{os.sep}arthur{os.sep}lgcm{os.sep}projects{os.sep}Segmentation_union_projects{os.sep}data_complete{os.sep}intermediate{os.sep}pre_processing_optico_and_afm{os.sep}image',
            'mask_path':f'{os.sep}home{os.sep}arthur{os.sep}lgcm{os.sep}projects{os.sep}Segmentation_union_projects{os.sep}data_complete{os.sep}intermediate{os.sep}pre_processing_optico_and_afm{os.sep}mask',
            
    },
    'half_unet_afm_2_channels_only_optical_data_without_artifacts':{
            'model': 'half_unet_afm_2_channels_only_optical_NN_data_without_artifacts_zscore.h5',
            'test_path': f'{os.sep}home{os.sep}arthur{os.sep}lgcm{os.sep}projects{os.sep}Segmentation_union_projects{os.sep}data_complete{os.sep}intermediate{os.sep}pre_processing_optico{os.sep}image',
            'mask_path': f'{os.sep}home{os.sep}arthur{os.sep}lgcm{os.sep}projects{os.sep}Segmentation_union_projects{os.sep}data_complete{os.sep}intermediate{os.sep}pre_processing_optico{os.sep}mask'
    }
}

opt_image = f'{os.sep}home{os.sep}arthur{os.sep}lgcm{os.sep}projects{os.sep}Segmentation_union_projects{os.sep}data_complete{os.sep}input{os.sep}optical_images_resized'
predict_path = f'{os.sep}home{os.sep}arthur{os.sep}lgcm{os.sep}projects{os.sep}Segmentation_union_projects{os.sep}data_complete{os.sep}input{os.sep}Usefull_data'


test_files_path = f'{os.sep}home{os.sep}arthur{os.sep}lgcm{os.sep}projects{os.sep}Segmentation_union_projects{os.sep}data_complete{os.sep}datasets{os.sep}Test_86_Images.tsv'
df_test_files = pd.read_csv(test_files_path, sep='\t')


# dataset_size = [15,30,60,120,240]
for model in model_dict.keys():
        if model != 'half_unet_afm_2_channels_only_optical_data_without_artifacts': 
                continue
        # for i in dataset_size:
                
        model_info = model_dict[model]
        model_name = model_info['model'].replace('NN', '240')
        print(f'get {model_name} validation... ')
        
        preprocess_image_path = [os.path.join(model_info['test_path'], file+'_channels_added.npy') for file in df_test_files['Process.Date'].values]
        mask_path = [os.path.join(model_info['mask_path'], file+'_channels_added.npy') for file in df_test_files['Process.Date'].values]
        opt_image_path = [os.path.join(opt_image, file+'_optico_crop_resized.png') for file in df_test_files['Process.Date'].values]
        usefull_path = [os.path.join(predict_path, file+'_UsefullData.tsv') for file in df_test_files['Process.Date'].values]

        df_list = []

        for i in tqdm(range(len(opt_image_path)), colour='#0000FF'):
                process_data = opt_image_path[i].split(f'{os.sep}')[-1]
                unetTrat =   UnetProcess(opt_image_path[i], preprocess_image_path[i], usefull_path[i], mask_path[i], model_path=f'models{os.sep}{model_name}') 
                
                # usefull_path_unet = usefull_path[i].replace(f'data{os.sep}input{os.sep}Usefull_data{os.sep}',f'data{os.sep}output{os.sep}{unet_afm_fold}{os.sep}predict_sheets{os.sep}')
                y,y_pred, _t = unetTrat.unet_predict()
                y_flatten = y.flatten()
                y_pred_flatten = y_pred.flatten()

                
                eval = EvalModel('unet',process_data, y_flatten, y_pred_flatten) 
                scores = eval.get_metrics()
                metric_df = eval.metrics_to_df(scores)
                df_list.append(metric_df)
        final_metric_df = pd.concat(df_list, axis=0)
        final_metrics_melt = pd.melt(final_metric_df, id_vars=['Process Date','Model'], value_vars=['Precision', 'Recall', 'F1', 'Dice'], var_name = 'metrics', value_name = 'scores')
        final_metrics_melt.to_csv(f'test_metrics_{model_name.split(".")[0]}_original_size.csv')        
