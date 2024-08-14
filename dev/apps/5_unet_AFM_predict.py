import os
import sys
from utils import UnetProcess 
from tqdm import tqdm

unet_afm_fold = 'unet_AFM_predictions'
model_name = 'UNet_AFM_2_channels_only_optico.h5'
data_folder = 'data_complete'


preprocess_image = f'{data_folder}{os.sep}intermediate{os.sep}pre_processing_optico_and_afm{os.sep}image{os.sep}'
mask = f'{data_folder}{os.sep}intermediate{os.sep}pre_processing_optico_and_afm{os.sep}mask{os.sep}'
opt_image = f'{data_folder}{os.sep}input{os.sep}optical_images_resized{os.sep}'
save_path = f'{data_folder}{os.sep}output{os.sep}predicts{os.sep}'
predict_path = f'{data_folder}{os.sep}input{os.sep}Usefull_data{os.sep}'
dire = os.listdir(preprocess_image)

opt_image_path = [opt_image + file.replace('_channels_added.npy', '_optico_crop_resized.png') for file in dire]
preprocess_image_path = [preprocess_image+file for file in dire]
usefull_path = [predict_path+file.replace('_channels_added.npy', '_UsefullData.tsv') for file in dire]
mask_path = [mask+file for file in dire]
save_path = [save_path+file.replace('_channels_added.npy', '_unet.png') for file in dire]

for i in tqdm(range(len(opt_image_path)), colour='#0000FF'):
        unetTrat =   UnetProcess(opt_image_path[i], preprocess_image_path[i], usefull_path[i], mask_path[i], model_path=f'models{os.sep}{model_name}') 
        
        # usefull_path_unet = usefull_path[i].replace(f'{data_folder}{os.sep}input{os.sep}Usefull_data{os.sep}',f'{data_folder}{os.sep}output{os.sep}predict_sheets{os.sep}')
        unetTrat.unet_predict(save_path[i],usefull_path =usefull_path[i]) 