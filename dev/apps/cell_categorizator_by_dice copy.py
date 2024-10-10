import os 
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

import cv2 
from io import BytesIO
from IPython.display import Image, display

from utils import EvalModel


def create_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def create_gs(df):
    """Generate generic segmentation based on conditions."""
    conditions = [
        ((df['Planned Height'] <= 650E-9) & (df['Segment'] != 'Substrate')),
        ((df['Planned Height'] > 650E-9) & (df['Segment'] != 'Substrate') & (df['Free Hand Segmentation'] != 1)), 
        (df['Free Hand Segmentation'] == 1) & (df['Segment'] != 'Substrate')]
    choices = ['Pericellular', 'Cytoplasm', 'Nucleus']
    df['Generic Segmentation'] = np.select(conditions, choices, default='NaN')

    return df

def clean_target(df, target='Generic Segmentation'):
    """Clean and map target column values to numeric values."""
    segment_mapping = {'Cytoplasm': 0, 'Nucleus': 1, 'Pericellular': 0, float('nan'): 0}
    df[target] = df[target].replace(segment_mapping)
    return df

def define_nucleus_region_by_afm(cos_height_sum, kernel_size = (3,3)):

        kernel = np.ones(kernel_size, dtype=np.uint8)

        cos_height_sum_01 = (cos_height_sum - cos_height_sum.min())/ (cos_height_sum.max() - cos_height_sum.min()) * 255
        
        ret, thresh_245_255 = cv2.threshold(cos_height_sum_01, 245,255, cv2.THRESH_BINARY)
    
        cos_heigh_erode_threshold_245_255_k5 = cv2.erode(-thresh_245_255, kernel,iterations = 1)
        cos_heigh_erode_threshold_245_255_k5 = -cos_heigh_erode_threshold_245_255_k5

        return cos_heigh_erode_threshold_245_255_k5

def minMax_Scaler(df, column_name, substrate=True):
    df[column_name] = (df[column_name] - df[column_name].min())/ (df[column_name].max() - df[column_name].min())
    return df



def generate_plot(optical_img, manual_segmentation, afm_segmentation, Dice, save_path):
    fig = plt.figure()
    plt.imshow(manual_segmentation, cmap='gray')
    plt.imshow(afm_segmentation,alpha=0.5)
    plt.imshow(optical_img,alpha=0.5, cmap='gray')
    plt.title(f'{save_path.split("/")[-1][:-4]} --> dice: {Dice:.2f}')
    plt.axis('off')
    plt.show()
    # plt.savefig(save_path, format='png') 
    confirm = input('Good?')
    plt.close()
    return confirm 


dice_score_list = []
process_date_list = []
save_path_list = []
category = []
def process_files(process_date, usefull_predicted_files, opt_images_resized, bw_images, txt_files, opt_images, usefull_files):
    """Process files and evaluate segmentation quality."""
    for i in range(len(usefull_predicted_files)):
 
 
        save_path = 'images_manual_and_afm_segment/'+process_date[i]+'.png'
        save_path_list.append(save_path)
        
        opt_image = cv2.imread(opt_images_resized[i])
        blue = opt_image[:,:,0]
        x, y, _ = opt_image.shape

        usefull_df = pd.read_csv(usefull_predicted_files[i], sep='\t')
        if len(usefull_df.columns)<2:
            usefull_df = pd.read_csv(usefull_predicted_files[i])
            
        
        usefull_df = clean_target(usefull_df)
        usefull_df = minMax_Scaler(usefull_df, 'Planned Height', substrate=True)
        
        manual_segment = np.array(usefull_df['Generic Segmentation']).reshape(x,y)
        manual_segment_flatten = manual_segment.flatten() 
        
        planned = np.array(usefull_df['Planned Height']).reshape(x,y)
        cos_height_sum = (np.cos(planned) + planned)
        
        afm_segment = define_nucleus_region_by_afm(cos_height_sum, kernel_size=(5,5))
        
        #normalize mask to 0 and 1 
        afm_segment_flatten = afm_segment.flatten()/255

        eval_model = EvalModel('unet', process_date[i], manual_segment_flatten, afm_segment_flatten)
        dice_score = eval_model.get_metrics()[-1][-1]
        print(dice_score)
        # process_date_list.append(process_date[i])
        # dice_score_list.append(dice_score)
        # confirm = generate_plot(blue, manual_segment, afm_segment, dice_score, save_path)
        if dice_score>=0.8:
            category.append('good')
            resegment_files(usefull_files[i], opt_images_resized[i], bw_images[i], txt_files[i], opt_images[i])       
        else:
                category.append('bad')
    # cell_info = {'Process Date':process_date_list,
    #                 'Dice': dice_score_list,
    #                 'dice_image': save_path_list,
    #                 'category': category
    #             }
        
    # df_afm_dice = pd.DataFrame(cell_info)
    # df_afm_dice = df_afm_dice.sort_values(by='Dice', ascending=False)
    # df_afm_dice.to_csv('AFM_Mask_Dice_Metrics.csv')
    # print('')
       

def resegment_files(*file_paths):
    """Copy files to resegmentation directories."""
    for file_path in file_paths:
        dest_path = file_path.replace('data_complete', 'good_afm')
        if file_path.split(f'{os.sep}')[2]=='predict_sheets':
            dest_path = file_path.replace('data_complete/output/predict_sheets', 'good_afm/input/Usefull_data')
        
        shutil.move(file_path, dest_path)

# Directory setup
base_dirs = [
    ('input/optical_images_resized', 'data_complete', 'data_resegment'),
    ('raw/bw_images', 'data_complete', 'data_resegment'),
    ('raw/txt_files', 'data_complete', 'data_resegment'),
    ('input/Usefull_data', 'data_complete', 'data_resegment'),
    ('raw/optical_images', 'data_complete', 'data_resegment')
]

for sub_dir, base_src, base_dest in base_dirs:
    create_dir(os.path.join(base_dest, sub_dir))

# File paths setup
def build_file_paths(base_path, file_list, suffixes=None):
    """Build file paths with optional suffixes."""
    if suffixes:
        return [os.path.join(base_path, f + suffix) for f in file_list for suffix in suffixes if os.path.isfile(os.path.join(base_path, f + suffix))]
    return [os.path.join(base_path, f) for f in file_list]

usefull_predicted_files = build_file_paths(f'data_complete{os.sep}input{os.sep}Usefull_data', os.listdir(f'data_complete{os.sep}input{os.sep}Usefull_data'))
process_date = [os.path.basename(file).split('_')[0] for file in usefull_predicted_files]

opt_images_resized = build_file_paths(f'data_complete{os.sep}input{os.sep}optical_images_resized{os.sep}', process_date, ['_optico_crop_resized.png'])
bw_images = build_file_paths(f'data_complete{os.sep}raw{os.sep}bw_images', process_date, ['_OpticalImg-BW.png'])
txt_files = build_file_paths(f'data_complete{os.sep}raw{os.sep}txt_files', process_date, ['_2-reference-force-height-extend.txt', '_3-reference-force-height-extend.txt', '_2-reference-force-height-extend.jpk-qi-image.txt'])
opt_images = build_file_paths(f'data_complete{os.sep}raw{os.sep}optical_images{os.sep}', process_date, ['_OpticalImg.png'])

# Process files
process_files(process_date, usefull_predicted_files, opt_images_resized, bw_images, txt_files, opt_images, usefull_predicted_files)

print("Processing complete.")
