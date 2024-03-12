import os 
import sys
sys.path.append(f'dev{os.sep}scripts')
from dataframe_treatment import PreProcessDataframe 
from models import UnetProcess, PixelProcess
from tqdm import tqdm

preprocess_image = f'data{os.sep}intermediate{os.sep}pre_processing_optico_and_afm{os.sep}image{os.sep}'
mask = f'data{os.sep}intermediate{os.sep}pre_processing_optico_and_afm{os.sep}mask{os.sep}'
opt_image = f'data{os.sep}input{os.sep}optical_images_resized{os.sep}'
save_path = f'data{os.sep}output{os.sep}vunet_AFM_predictions{os.sep}predicts{os.sep}'
usefull_dir = f'data{os.sep}input{os.sep}Usefull_data{os.sep}'

save_preprocess_pixel_path = f'data{os.sep}intermediate{os.sep}pre_processing_only_afm{os.sep}'

dire = os.listdir(preprocess_image)

opt_image_path = [opt_image + file.replace('_channels_added.png', '_optico_crop_resized.png') for file in dire]
preprocess_image_path = [preprocess_image+file for file in dire]
usefull_path = [usefull_dir+file.replace('_channels_added.png', '_UsefullData.tsv') for file in dire]
mask_path = [mask+file for file in dire]
save_path = [save_path+file.replace('_channels_added.png', '_unet.png') for file in dire]


save_preprocess_pixel_path = [save_preprocess_pixel_path+file.replace('_channels_added.png', '_UsefullData.tsv') for file in dire]



for i in tqdm(range(len(opt_image_path))):
    unetTrat =   UnetProcess(opt_image_path[i], preprocess_image_path[i], usefull_path[i], mask_path[i]) 
    
    usefull_path_save = usefull_path[i].replace(f'data{os.sep}input{os.sep}Usefull_data{os.sep}',f'data{os.sep}output{os.sep}vunet_AFM_predictions{os.sep}predict_sheets{os.sep}') 
    _, __, result = unetTrat.unet_predict(save_path[i],usefull_path =usefull_path_save)
    
    if result: 
        save_path[i] = save_path[i].replace('_unet.png', '.png')

        only_afm_process = PreProcessDataframe(usefull_path[i])
        only_afm_process.run_preprocess_pixel_segmentation(save_preprocess_pixel_path[i])
        
        
        pixel = PixelProcess(save_preprocess_pixel_path[i], opt_image_path[i])
        pixel.pixel_predict(save_path[i], usefull_path[i], usefull_path_save)