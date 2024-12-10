import os
import numpy as np
from utils import UnetProcess, EvalModel, Models, UNET_MODELS_PATH, CROP_PATH
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt 
import pandas as pd 

def create_topography_map(unetTrat, matrix = False):
        if matrix:
                matrix = True
        opt_image_dimension = unetTrat.opt_image
        opt_image_dimension = opt_image_dimension.dimensions(matrix=False)
        
        topography = np.array(unetTrat.df_afm.df['Planned Height']).reshape(opt_image_dimension)
        return topography


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)



opt_image =CROP_PATH['optical_crop_resized']
predict_path = CROP_PATH['usefull_data']


test_files_path = f'data{os.sep}datasets{os.sep}selected_with_good_optico{os.sep}Test_83_Images.tsv'
df_test_files = pd.read_csv(test_files_path)


dataset_size = [234]
for model in UNET_MODELS_PATH.keys():
        if model == 'unet_afm_1_channels_only_AFM_CosHeightSum': 
                continue
        for i in dataset_size:
        
                model_info = UNET_MODELS_PATH[model]
                model_name = model_info['model'].replace('NN', str(i))
                model_path = f'models{os.sep}{model_name}'
                
                print(f'get {model_name} validation... ')
                
                preprocess_image_path = [os.path.join(model_info['test_path'], file+'_channels_added.npy') for file in df_test_files['Process.Date'].values]
                mask_path = [os.path.join(model_info['mask_path'], file+'_channels_added.npy') for file in df_test_files['Process.Date'].values]
                opt_image_path = [os.path.join(opt_image, file+'_optico_crop_resized.png') for file in df_test_files['Process.Date'].values]
                usefull_path = [os.path.join(predict_path, file+'_UsefullData.tsv') for file in df_test_files['Process.Date'].values]

                df_list = []
                model = Models('unet', model_path)
                for i in tqdm(range(len(opt_image_path)), colour='#0000FF'):
                        process_data = opt_image_path[i].split(f'{os.sep}')[-1].split('_')[0]
                        unetTrat =   UnetProcess(opt_image_path[i], preprocess_image_path[i], usefull_path[i], mask_path[i]) 
                        
                        # usefull_path_unet = usefull_path[i].replace(f'data{os.sep}input{os.sep}Usefull_data{os.sep}',f'data{os.sep}output{os.sep}{unet_afm_fold}{os.sep}predict_sheets{os.sep}')
                        y,y_pred, _t = unetTrat.unet_predict(model)
                        y_flatten = y.flatten()
                        y_pred_flatten = y_pred.flatten()

                        
                        eval = EvalModel('unet',process_data, y_flatten, y_pred_flatten) 
                        scores = eval.get_metrics()
                        metric_df = eval.metrics_to_df(scores)
                        if metric_df['Dice'][0] <=0.7:
                                
                                dice_file_name = f'{metric_df["Dice"][0]:.2f}'
                                save_path = f'bad_predictions_by_model2{os.sep}{model_name.split(".")[0]}{os.sep}'
                                create_dir(save_path)
                                save_path = save_path + f'{process_data}_dice_{dice_file_name}.svg'
                                
                                print(metric_df)
                                topography = create_topography_map(unetTrat)
                                unetTrat.show_predict(process_date = process_data, y=y, y_pred = y_pred, topography=topography, save_path=save_path, metrics=metric_df)
