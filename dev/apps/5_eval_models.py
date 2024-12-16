import os
import numpy as np
from utils import UnetProcess, EvalModel, Models, UNET_MODELS_PATH, CROP_PATH, TRAIN_TEST_FILES, create_dir
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


def build_paths(model_info, test_files_list):
        preprocess_image_path = [os.path.join(model_info['test_path'], file+'_channels_added.npy') for file in test_files_list['Process.Date'].values if os.path.isfile(os.path.join(model_info['test_path'], file+'_channels_added.npy'))]
        mask_path = [os.path.join(model_info['mask_path'], file+'_channels_added.npy') for file in test_files_list['Process.Date'].values if os.path.isfile(os.path.join(model_info['mask_path'], file+'_channels_added.npy'))]
        opt_image_path = [os.path.join(CROP_PATH['optical_crop_resized'], file+'_optico_crop_resized.png') for file in test_files_list['Process.Date'].values if os.path.isfile(os.path.join(CROP_PATH['optical_crop_resized'], file+'_optico_crop_resized.png'))]
        usefull_path = [os.path.join(CROP_PATH['usefull_data'], file+'_UsefullData.tsv') for file in test_files_list['Process.Date'].values if os.path.isfile(os.path.join(CROP_PATH['usefull_data'], file+'_UsefullData.tsv'))]

        return preprocess_image_path, mask_path, opt_image_path, usefull_path

def get_model_info(model_name, train_size):
                model_info = UNET_MODELS_PATH[model_name]
                model_name = model_info['model'].replace('NN', train_size)
                model_path = os.path.join('models', model_name) 
                print(f'get {model_name} validation... ')
                
                return model_info, model_name, model_path


def save_specific_metrics():
        dice_file_name = f'{metric_df["Dice"][0]:.2f}'
        create_dir(model_info['save_predict'])
        save_path = os.path.join(model_info['save_predict'], f'{process_data}_dice_{dice_file_name}.svg')  
        
        print(metric_df)
        topography = create_topography_map(unetTrat)
        unetTrat.show_predict(process_date = process_data, y=y, y_pred = y_pred, topography=topography, save_path=save_path, metrics=metric_df)



df_test_files = pd.read_csv(TRAIN_TEST_FILES['test'])


for model in UNET_MODELS_PATH.keys():

        for train_size in TRAIN_TEST_FILES['train'][0].keys():
                model_info, model_name, model_path = get_model_info(model, train_size)
                preprocess_image_path, mask_path, opt_image_path, usefull_path = build_paths(model_info, df_test_files)
                
                df_list = []
                model = Models('unet', model_path)
                for i in tqdm(range(len(opt_image_path)), colour='#0000FF'):
                        process_data = opt_image_path[i].split(f'{os.sep}')[-1].split('_')[0]
                        unetTrat =   UnetProcess(opt_image_path[i], preprocess_image_path[i], usefull_path[i], mask_path[i]) 
                        
                        y,y_pred, _prob = unetTrat.unet_predict(model)
                        y_flatten = y.flatten()
                        y_pred_flatten = y_pred.flatten()

                        eval = EvalModel('unet', y_flatten, y_pred_flatten, process_date=process_data) 
                        scores = eval.get_metrics()
                        metric_df = eval.metrics_to_df(scores)
                        df_list.append(metric_df)
                        
                        save_specific_metrics()
                