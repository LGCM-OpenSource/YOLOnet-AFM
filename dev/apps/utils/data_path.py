
import os 


def build_file_path(path, file, actual_process = '_OpticalImg.png',  new_process = ''):
    """
    Processes a file name by replacing a prefix with a suffix and returns 
    the resulting full file path.

    Parameters:
        path (str): The directory path where the file is located.
        file (str): The original file name to be processed.
        preffix (str, optional): The part of the file name to be replaced. 
            Default is '_OpticalImg.png'.
        suffix (str, optional): The text that will replace the prefix. 
            Default is an empty string ('').

    Returns:
        str: The full path of the processed file.

    Example:
        >>> import os
        >>> path = '/data/images'
        >>> file = 'sample_OpticalImg.png'
        >>> treat_file_names(path, file)
        '/data/images/sample'
    """
    filename = file.replace(actual_process, new_process)
    file_path = os.path.join(path, filename)
    return file_path


CROP_PATH = {
    
    'optical_raw' : f'data{os.sep}raw{os.sep}optical_images{os.sep}',
    'optical_bw_raw' : f'data{os.sep}raw{os.sep}bw_images',
    'txt_files': f'data{os.sep}raw{os.sep}txt_files{os.sep}',
    'usefull_data': f'data{os.sep}input{os.sep}Usefull_data{os.sep}',
    'optical_crop_resized' : f'data{os.sep}input{os.sep}optical_images_resized'
}


UNET_MODELS_PATH = {
    
    'unet_afm_1_channels_only_AFM_CosHeightSum': {
            'model': 'unet_afm_1_channels_only_AFM_CosHeightSum_NN_samples_stardist_mask.h5',
            'preprocess_img': f'data{os.sep}intermediate{os.sep}pre_processing_afm{os.sep}image{os.sep}',
            'preprocess_mask': f'data{os.sep}intermediate{os.sep}pre_processing_afm{os.sep}mask{os.sep}',
            'test_path': f'data{os.sep}input{os.sep}train{os.sep}train_1_channels_only_AFM_CosHeightSum{os.sep}opt_img_training',
            'mask_path': f'data{os.sep}input{os.sep}train{os.sep}train_1_channels_only_AFM_CosHeightSum{os.sep}msk_img_training',
            'save_predict': f'data/output/unet_afm_1_channels_only_AFM_CosHeightSum'
                
    },
    'unet_afm_2_channels_like_yolo_opt_afm':{
            'model':'unet_afm_2_channels_like_yolo_opt_afm_NN_samples_stardist_mask.h5',
            'preprocess_img': f'data{os.sep}intermediate{os.sep}pre_processing_optico_and_afm{os.sep}image{os.sep}',
            'preprocess_mask': f'data{os.sep}intermediate{os.sep}pre_processing_optico_and_afm{os.sep}mask{os.sep}',
            'test_path':f'data{os.sep}input{os.sep}train{os.sep}train_2_channels_like_yolo_opt_afm{os.sep}opt_img_training',
            'mask_path':f'data{os.sep}input{os.sep}train{os.sep}train_2_channels_like_yolo_opt_afm{os.sep}msk_img_training',
            'save_predict': f'data/output/unet_afm_2_channels_like_yolo_opt_afm'
            
    },
    'unet_afm_2_channels_only_optical':{
            'model': 'unet_afm_2_channels_only_optical_NN_samples_stardist_mask.h5',
            'preprocess_img':  f'data{os.sep}intermediate{os.sep}pre_processing_optico{os.sep}image{os.sep}',
            'preprocess_mask': f'data{os.sep}intermediate{os.sep}pre_processing_optico{os.sep}mask{os.sep}',
            'test_path': f'data{os.sep}input{os.sep}train{os.sep}train_2_channels_only_optical{os.sep}opt_img_training',
            'mask_path': f'data{os.sep}input{os.sep}train{os.sep}train_2_channels_only_optical{os.sep}msk_img_training',
            'save_predict': f'data/output/unet_afm_2_channels_only_optical'
    }
    
    
}

