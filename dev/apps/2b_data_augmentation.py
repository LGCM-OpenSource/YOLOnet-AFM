import os 
from tqdm import tqdm 
import numpy as np
import cv2 
import pandas as pd 
from utils import UNET_MODELS_PATH, TerminalStyles, load_config, Augmenter
import argparse 

term = TerminalStyles()
parser = argparse.ArgumentParser()
parser.add_argument('-ms', '--model_selection', type=str, help="select your model to choice preprocess step to make segmentations predictions")
args = parser.parse_args()
model_selector = args.model_selection
# model_selector = 'unet_afm_1_channels_only_AFM_CosHeightSum'

config = load_config()
H,W = config['model']['input_shape']

data_train = UNET_MODELS_PATH[model_selector]
    
### Get config pre-process and train path
afm_optico_path = data_train['preprocess_img']
mask_path = data_train['preprocess_mask']
afm_optico_files_dest = data_train['train_path']
mask_files_dest = data_train['mask_path']

augmenter = Augmenter(
        input_img_dir=afm_optico_path,
        input_mask_dir=mask_path,
        output_img_dir=afm_optico_files_dest,
        output_mask_dir=mask_files_dest,
        target_shape=(H, W)
    )

augmenter.apply()