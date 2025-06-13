import os
import numpy as np
from utils import (UnetProcess, EvalModel, Models, 
                   UNET_MODELS_PATH, CROP_PATH, create_dir, 
                   DataChart, Charts, TerminalStyles,
                   setup_logger)
from tqdm import tqdm
import matplotlib.pyplot as plt 
import pandas as pd 
import cv2 
import argparse


logger = setup_logger('Evaluating model')

def create_topography_map(unetTrat, matrix = False):
        opt_image_dimension = unetTrat.opt_image
        opt_image_dimension = opt_image_dimension.dimensions(matrix)
        
        topography = np.array(unetTrat.df_afm.df['Planned Height']).reshape(opt_image_dimension)
        return topography


def build_paths(model_info, test_files_list):
        preprocess_image_path = [os.path.join(model_info['preprocess_img'], file+'_channels_added.npy') for file in test_files_list if os.path.isfile(os.path.join(model_info['preprocess_img'], file+'_channels_added.npy'))]
        mask_path = [os.path.join(model_info['preprocess_mask'], file+'_channels_added.npy') for file in test_files_list if os.path.isfile(os.path.join(model_info['preprocess_mask'], file+'_channels_added.npy'))]
        opt_image_path = [os.path.join(CROP_PATH['optical_crop_resized'], file+'_optico_crop_resized.png') for file in test_files_list if os.path.isfile(os.path.join(CROP_PATH['optical_crop_resized'], file+'_optico_crop_resized.png'))]
        usefull_path = [os.path.join(CROP_PATH['usefull_data'], file+'_UsefullData.tsv') for file in test_files_list if os.path.isfile(os.path.join(CROP_PATH['usefull_data'], file+'_UsefullData.tsv'))]
        y_pred_path = [os.path.join(model_info['save_predict'], file+'_unet.png') for file in test_files_list if os.path.isfile(os.path.join(model_info['save_predict'], file+'_unet.png'))]
        return preprocess_image_path, mask_path, opt_image_path, usefull_path, y_pred_path


def save_specific_metrics():
        dice_file_name = f'{metric_df["Dice"][0]:.2f}'
        create_dir(model_info['save_metrics'])
        
        save_path = os.path.join(model_info['save_metrics'], f'{process_data}_dice_{dice_file_name}.png')  
        
        logger.info(f"Metrics DataFrame:\n{metric_df}")
        topography = create_topography_map(unetTrat)
        unetTrat.show_predict(process_date = process_data, y=y, y_pred = y_pred, topography=topography, save_path=save_path, metrics=metric_df)

term = TerminalStyles()
parser = argparse.ArgumentParser()
parser.add_argument('-ms', '--model_selection', type=str, help="select your model to choice preprocess step to make segmentations predictions")
args = parser.parse_args()
# model_selector = args.model_selection

model_selector = 'unet_afm_1_channels_only_AFM_CosHeightSum'

data_chart = DataChart()
chart = Charts(width=800, height = 500)

# df_test_files = pd.read_csv(TRAIN_TEST_FILES['test'])

predict_list = os.listdir(UNET_MODELS_PATH[model_selector]['save_predict'])
predict_list = [i.split('_')[0] for i in predict_list]


model_info = UNET_MODELS_PATH[model_selector]
preprocess_image_path, mask_path, opt_image_path, usefull_path, y_pred_path = build_paths(model_info, predict_list)

df_list = []
model = Models(model_info['model_name'], model_info['model_path'])
for i in tqdm(range(len(opt_image_path)), colour='#0000FF'):
        try:
                process_data = opt_image_path[i].split(f'{os.sep}')[-1].split('_')[0]
                unetTrat =   UnetProcess(opt_image_path[i], preprocess_image_path[i], usefull_path[i], mask_path[i]) 
                
                y = unetTrat.mask.image(matrix=True)
                y_pred = cv2.imread(y_pred_path[i], 0)
                ret2,y_pred_thr_otsu = cv2.threshold(y_pred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
                y_pred_thr_otsu_normalized = y_pred_thr_otsu / 255.0
        
                y_flatten = y.flatten()
                y_pred_flatten = y_pred_thr_otsu_normalized.flatten()

                eval = EvalModel(model.model_name, y_flatten, y_pred_flatten, process_date=process_data) 
                scores = eval.get_metrics()
                metric_df = eval.metrics_to_df(scores)
                df_list.append(metric_df)
                save_specific_metrics()
        except Exception:
                logger.error(f"Error processing image index {i}: {e}", exc_info=True)
        
model_validation_metrics = pd.concat(df_list, axis=0)

model_validation_metrics_melt = data_chart.apply_melt(model_validation_metrics)

fig = chart.box_plot(model_validation_metrics_melt, x='Model', y='scores', color='metrics')
fig.write_image(f'{model_info["save_metrics"]}model_metrics.png')


logger.info(f'''
      {term.BOLD}{model_selector}{term.RESET} metrics saved in {term.SAVE_COLOR}{model_info['save_metrics']}{term.RESET} \n 
      ''')