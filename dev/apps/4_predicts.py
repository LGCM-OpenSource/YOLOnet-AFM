import os
import sys
from utils import UnetProcess, Models, build_file_path, CROP_PATH, UNET_MODELS_PATH, TRAIN_TEST_FILES
from tqdm import tqdm


def build_paths(files_list, actual_process = '_channels_added.npy'):
        opt_image_path = [build_file_path(CROP_PATH['optical_crop_resized'], img, actual_process = actual_process, new_process='_optico_crop_resized.png') for img in files_list]
        usefull_path = [build_file_path(CROP_PATH['usefull_data'], img, actual_process = actual_process, new_process='_UsefullData.tsv') for img in files_list]
        preprocess_image_path = [os.path.join(model_info['preprocess_img'], img) for img in files_list]
        mask_path = [os.path.join(model_info['preprocess_mask'], img) for img in files_list]
        save_path = [model_info['save_predict']+file.replace('_channels_added.npy', '_unet.png') for file in files_list]

        return opt_image_path, usefull_path, preprocess_image_path, mask_path, save_path


def select_model(user_input):
        models = {
                '1': ('AFM Only','unet_afm_1_channels_only_AFM_CosHeightSum'),
                '2': ('YOLO-AFM' ,'unet_afm_2_channels_like_yolo_opt_afm'),
                '3': ('Optical Only' ,'unet_afm_2_channels_only_optical')
        }
        selected_model = models[user_input][0]
        print(f'\nModel ____{selected_model}____ selected')
        return models[user_input][1]


selector = select_model(input('''Select your model: \n
                 1 - AFM Only\n
                 2 - YOLO-AFM\n
                 3 - Optical Only\n
                 ____________________________________
                 '''))

model_info = UNET_MODELS_PATH[selector]

files_list = os.listdir(model_info['preprocess_img'])
opt_image_path, usefull_path, preprocess_image_path, mask_path, save_path = build_paths(files_list)
model = Models('unet', model_info["model_path"])
for i in tqdm(range(len(opt_image_path)), colour='#0000FF'):
        unetTrat =   UnetProcess(opt_image_path[i], preprocess_image_path[i], usefull_path[i], mask_path[i]) 
        y,y_pred = unetTrat.unet_predict(model)
        unetTrat.save_predict(y, save_path[i])