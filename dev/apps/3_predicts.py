import os
import sys
from utils import UnetProcess, build_file_path, CROP_PATH, UNET_MODELS_PATH
from tqdm import tqdm



model = 'unet_afm_2_channels_like_yolo_opt_afm'
model_info = UNET_MODELS_PATH[model]

model_name = model_info['model']
preprocess_image = model_info['preprocess']


save_path = model_info['save_predict']
dire = os.listdir(preprocess_image)

actual_process = '_channels_added.npy'
opt_image_path = [build_file_path(CROP_PATH['optical_crop_resized'], img, actual_process = actual_process, new_process='_optico_crop_resized.png') for img in dire]
usefull_path = [build_file_path(CROP_PATH['usefull_data'], img, actual_process = actual_process, new_process='_UsefullData.tsv') for img in dire]

preprocess_image_path = [os.path.join(model_info['preprocess_img'], img) for img in dire]
mask_path = [os.path.join(model_info['preprocess_mask'], img) for img in dire]

# Elaborar como armazenar as predições (nos modelos? ou salvamentos?)
save_path = [save_path+file.replace('_channels_added.npy', '_unet.png') for file in dire]

for i in tqdm(range(len(opt_image_path)), colour='#0000FF'):
        unetTrat =   UnetProcess(opt_image_path[i], preprocess_image_path[i], usefull_path[i], mask_path[i], model_path=f'models{os.sep}{model_name}') 
        
        unetTrat.unet_predict(save_path[i],usefull_path =usefull_path[i]) 