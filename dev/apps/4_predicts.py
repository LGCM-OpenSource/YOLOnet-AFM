import os
import sys
from utils import UnetProcess, Models, build_file_path, CROP_PATH, UNET_MODELS_PATH, TRAIN_TEST_FILES, TerminalStyles, UserInput 
from tqdm import tqdm
import argparse



def build_paths(files_list, actual_process = '_channels_added.npy'):
        opt_image_path = [build_file_path(CROP_PATH['optical_crop_resized'], img, actual_process = actual_process, new_process='_optico_crop_resized.png') for img in files_list]
        usefull_path = [build_file_path(CROP_PATH['usefull_data'], img, actual_process = actual_process, new_process='_UsefullData.tsv') for img in files_list]
        preprocess_image_path = [os.path.join(model_info['preprocess_img'], img) for img in files_list]
        mask_path = [os.path.join(model_info['preprocess_mask'], img) for img in files_list]
        save_path = [model_info['save_predict']+file.replace('_channels_added.npy', '_unet.png') for file in files_list]

        return opt_image_path, usefull_path, preprocess_image_path, mask_path, save_path


parser = argparse.ArgumentParser()
parser.add_argument('-ms', '--model_selection', type=str, help="select your model to choice preprocess step to make segmentations predictions")
args = parser.parse_args()
model_selector = args.model_selection

# model_selector = 'unet_afm_2_channels_like_yolo_opt_afm'
confirmation = UserInput.get_user_confirmation('DO YOU WANNA VISUALIZE SEGMENTATION?')

model_info = UNET_MODELS_PATH[model_selector]
files_list = os.listdir(model_info['preprocess_img'])
opt_image_path, usefull_path, preprocess_image_path, mask_path, save_path = build_paths(files_list)
model = Models(model_info['model_name'], model_info["model_path"])

for i in tqdm(range(len(opt_image_path)), colour='#0000FF'):
        unetTrat =   UnetProcess(opt_image_path[i], preprocess_image_path[i], usefull_path[i], mask_path[i]) 
        y_pred = unetTrat.unet_predict(model)
        
        y_resized = unetTrat.resize_prediction_to_original_size(y_pred)            
        unetTrat.save_predict(y_resized, save_path[i])
        if confirmation: 
                unetTrat.visualize_prediction(save_path[i])
                
        