from utils import GenerateAFMOptico, UNET_MODELS_PATH, CROP_PATH, build_file_path
import os 
from tqdm import tqdm 

model = 'unet_afm_2_channels_like_yolo_opt_afm'
pre_process = UNET_MODELS_PATH[model]['pre_process']

dire = os.listdir(CROP_PATH['optical_crop_resized'])
last_process = '_optico_crop_resized.png'

for img in tqdm(dire):
    
    opt_crop_resized_path = os.path.join(CROP_PATH['optical_crop_resized'], img)
    usefull_path = build_file_path(CROP_PATH['usefull_data'], img, actual_process=last_process, new_process = '_UsefullData.tsv')
    save_path  = build_file_path(pre_process, img, actual_process=last_process, new_process='_channels_added.npy')

    afm_optico_process = GenerateAFMOptico(opt_crop_resized_path, usefull_path)
    afm_optico_process.run_generate_afm_optico_images(save_path)
