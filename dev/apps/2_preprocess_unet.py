import os 
import sys
sys.path.append(f'dev{os.sep}scripts')
from image_treatment import GenerateAFMOptico 
from tqdm import tqdm 


opt_image_dir = f'data{os.sep}input{os.sep}optical_images_resized{os.sep}'
usefull_data_dir = f'data{os.sep}input{os.sep}Usefull_data{os.sep}'
save_dir = f'data{os.sep}intermediate{os.sep}pre_processing_optico_and_afm{os.sep}image{os.sep}'

dire = os.listdir(opt_image_dir)

for img in tqdm(dire, colour='#0000FF'):
    usefull_name = img.replace('_optico_crop_resized.png','_UsefullData.tsv')
    processed_name = img.replace('_optico_crop_resized.png','_channels_added.png')

    optical_path = opt_image_dir+img 
    usefull_path = usefull_data_dir+usefull_name
    save_path = save_dir+processed_name
    
    
    afm_optico_process = GenerateAFMOptico(optical_path, usefull_path)
    
    afm_optico_process.run_generate_afm_optico_images(save_path)
