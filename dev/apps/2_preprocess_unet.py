from utils import GenerateAFMOptico, UNET_MODELS_PATH, CROP_PATH, build_file_path,TerminalStyles
import os 
from tqdm import tqdm 
import argparse


term = TerminalStyles()
parser = argparse.ArgumentParser()
parser.add_argument('-ms', '--model_selection', type=str, help="select your model to choice preprocess step to make segmentations predictions")
args = parser.parse_args()
model_selector = args.model_selection

# model_selector = 'unet_afm_2_channels_like_yolo_opt_afm'

dire = os.listdir(CROP_PATH['optical_crop_resized'])
last_process = '_optico_crop_resized.png'


model_info = UNET_MODELS_PATH[model_selector]
for img in tqdm(dire):
    
    opt_crop_resized_path = os.path.join(CROP_PATH['optical_crop_resized'], img)
    usefull_path = build_file_path(CROP_PATH['usefull_data'], img, actual_process=last_process, new_process = '_UsefullData.tsv')
    save_img_path  = build_file_path(model_info['preprocess_img'], img, actual_process=last_process, new_process='_channels_added.npy')
    save_mask_path  = build_file_path(model_info['preprocess_mask'], img, actual_process=last_process, new_process='_channels_added.npy')

    
    
    afm_optico_process = GenerateAFMOptico(opt_crop_resized_path, usefull_path)
    new_img, mask = afm_optico_process.run_generate_afm_optico_images(model_info['model_name'])
    
    afm_optico_process.save_matrix(save_img_path, new_img)
    afm_optico_process.save_matrix(save_mask_path, mask)
    
print(f'''{model_selector} pre-processing:\n
      Images saved in {term.SAVE_COLOR}{model_info['preprocess_img']}\033[0m\n
      Masks saved in {term.SAVE_COLOR}{model_info['preprocess_mask']}\033[0m''')
    
