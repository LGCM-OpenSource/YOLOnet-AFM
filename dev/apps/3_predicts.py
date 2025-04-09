import os
import sys
from utils import UnetProcess, Models, build_file_path, CROP_PATH, UNET_MODELS_PATH, TRAIN_TEST_FILES, TerminalStyles, UserInput 
from tqdm import tqdm
import argparse



def build_paths(files_list, actual_process='_channels_added.npy'):
    valid_data = []

    for img in files_list:
        opt = build_file_path(CROP_PATH['optical_crop_resized'], img, actual_process, '_optico_crop_resized.png')
        use = build_file_path(CROP_PATH['usefull_data'], img, actual_process, '_UsefullData.tsv')
        pre = os.path.join(model_info['preprocess_img'], img)
        mask = os.path.join(model_info['preprocess_mask'], img)
        save = os.path.join(model_info['save_predict'], img.replace('_channels_added.npy', '_unet.png'))

        paths = {
            "optical": opt,
            "usefull": use,
            "preprocess_img": pre,
            "mask": mask
        }

        missing = [name for name, path in paths.items() if not os.path.exists(path)]

        if missing:
            print(f"[AVISO] File '{img}' ignored. Path doesn't exist in: {', '.join(missing)}")
        else:
            valid_data.append((opt, use, pre, mask, save))


    if not valid_data:
        return [], [], [], [], []

    opt, use, pre, mask, save = zip(*valid_data)
    return list(opt), list(use), list(pre), list(mask), list(save)

term = TerminalStyles()
parser = argparse.ArgumentParser()
parser.add_argument('-ms', '--model_selection', type=str, help="select your model to choice preprocess step to make segmentations predictions")
args = parser.parse_args()
model_selector = args.model_selection

# model_selector = 'unet_afm_1_channels_only_AFM_CosHeightSum'
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
                
print(f'''
      {term.BOLD}{model_selector}{term.RESET}\npredicts saved in {term.SAVE_COLOR}{model_info['save_predict']}{term.RESET}
      ''')
                
        