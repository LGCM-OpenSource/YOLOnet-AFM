import os
import sys 
sys.path.append(f'dev{os.sep}scripts')
from models import PixelProcess
from tqdm import tqdm

usefull_data_original_dir = f'data{os.sep}input{os.sep}Usefull_data{os.sep}'
usefull_data_normalized_dir = f'data{os.sep}intermediate{os.sep}pre_processing_only_afm{os.sep}'
opt_image_dir = f'data{os.sep}input{os.sep}optical_images_resized{os.sep}'
save_dir = f'data{os.sep}output{os.sep}only_afm_predictions{os.sep}predicts{os.sep}'

dire = os.listdir(usefull_data_normalized_dir)

usefull_data_original_path = [os.path.join(usefull_data_original_dir, file) for file in dire]
usefull_data_normalized_path = [os.path.join(usefull_data_normalized_dir, file) for file in dire]
opt_image_path = [os.path.join(opt_image_dir, file.replace('_UsefullData.tsv', '_optico_crop_resized.png')) for file in dire]
save_path = [os.path.join(save_dir, file.replace('_UsefullData.tsv', '.png')) for file in dire]

for i in tqdm(range(len(usefull_data_normalized_path))):
     usefull_path_save = usefull_data_original_path[i].replace(f'data{os.sep}input{os.sep}Usefull_data{os.sep}',f'data{os.sep}output{os.sep}only_afm_predictions{os.sep}predict_sheets{os.sep}')  
    
     pixel = PixelProcess(usefull_data_normalized_path[i], opt_image_path[i])
     pixel.pixel_predict(save_path[i], usefull_data_original_path[i], usefull_path_save)
   
    