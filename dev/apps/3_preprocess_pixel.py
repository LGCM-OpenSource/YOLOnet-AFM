import os 
import sys
sys.path.append(f'dev{os.sep}scripts')
from dataframe_treatment import PreProcessDataframe 
from tqdm import tqdm 


usefull_data_dir = f'data{os.sep}input{os.sep}Usefull_data{os.sep}'
save_dir = f'data{os.sep}intermediate{os.sep}pre_processing_only_afm{os.sep}'

dire = os.listdir(usefull_data_dir)

for df in tqdm(dire, colour='#0000FF'):
    usefull_path = usefull_data_dir + df
    save_path = save_dir + df
    
    afm_optico_process = PreProcessDataframe(usefull_path)
    
    afm_optico_process.run_preprocess_pixel_segmentation(save_path)