import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
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

def process_files(process_date, usefull_predicted_files, opt_images_resized, bw_images, txt_files, opt_images, usefull_files):
    """Process files and evaluate segmentation quality."""
    for i in range(len(usefull_predicted_files)):
        opt_image = cv2.imread(opt_images_resized[i])
        x, y, _ = opt_image.shape

        usefull_df = pd.read_csv(usefull_predicted_files[i], sep='\t')
        if len(usefull_df.columns)<2:
            usefull_df = pd.read_csv(usefull_predicted_files[i])
            
        # if not 'unet_prediction' in usefull_df.columns:
        #     continue
        
        usefull_df = clean_target(usefull_df)

        manual_segment = usefull_df['Generic Segmentation']
        
        
        plt.subplot(1,2,1)
        plt.title(process_date[i])
        plt.imshow(opt_image)
        
        plt.subplot(1,2,2)
        plt.title('Manual and Unet Segmentation')
        plt.imshow(np.array(manual_segment).reshape(x, y))
        # plt.imshow(np.array(unet_segment).reshape(x, y), alpha=0.5, cmap='gray')
        plt.imshow(opt_image, alpha=0.5)
        plt.show()
        
        confirm = input('Boa segmentação? (s/n): ')
        plt.close()
        
        if confirm.lower() == 'n':
                # usefull_df = usefull_df.drop(['unet_prediction'], axis=1)
                resegment_files(usefull_files[i], opt_images_resized[i], bw_images[i], txt_files[i], opt_images[i])
        elif confirm.lower()=='':
            pass
            # usefull_df['Generic Segmentation'] = manual_segment
            # usefull_df = create_gs(usefull_df)
            # # usefull_df = usefull_df.drop(['unet_prediction'], axis=1)
            # usefull_df.to_csv(usefull_files[i], sep='\t')
        else:
            usefull_df = create_gs(usefull_df)
            # usefull_df = usefull_df.drop(['unet_prediction'], axis=1)
            usefull_df.to_csv(usefull_files[i], sep='\t')
        
        
        
        # unet_segment = usefull_df['unet_prediction']

        # eval_model = EvalModel('unet', process_date[i], manual_segment, unet_segment)
        # dice_score = eval_model.get_metrics()[-1][-1]

        # if dice_score < 0.90:
        #     plt.subplot(1,2,1)
        #     plt.title(process_date[i])
        #     plt.imshow(opt_image)
            
        #     plt.subplot(1,2,2)
        #     plt.title('Manual and Unet Segmentation')
        #     plt.imshow(np.array(manual_segment).reshape(x, y))
        #     plt.imshow(np.array(unet_segment).reshape(x, y), alpha=0.5, cmap='gray')
        #     plt.imshow(opt_image, alpha=0.5)
        #     plt.show()
            
            
        #     confirm = input('Boa segmentação? (s/n): ')
        #     plt.close()
        #     if confirm.lower() == 'n':
        #         # usefull_df = usefull_df.drop(['unet_prediction'], axis=1)
        #         resegment_files(usefull_files[i], opt_images_resized[i], bw_images[i], txt_files[i], opt_images[i])
        #     elif confirm.lower()=='s':
        #         usefull_df['Generic Segmentation'] = manual_segment
        #         usefull_df = create_gs(usefull_df)
        #         # usefull_df = usefull_df.drop(['unet_prediction'], axis=1)
        #         usefull_df.to_csv(usefull_files[i], sep='\t')
        #     else:
        #         usefull_df = create_gs(usefull_df)
        #         # usefull_df = usefull_df.drop(['unet_prediction'], axis=1)
        #         usefull_df.to_csv(usefull_files[i], sep='\t')
        # else:
        #     usefull_df['Generic Segmentation'] = manual_segment
        #     usefull_df = create_gs(usefull_df)
        #     # usefull_df = usefull_df.drop(['unet_prediction'], axis=1)
        #     usefull_df.to_csv(usefull_files[i], sep='\t')

def resegment_files(*file_paths):
    """Copy files to resegmentation directories."""
    for file_path in file_paths:
        dest_path = file_path.replace('data_complete', 'data_resegment')
        if file_path.split(f'{os.sep}')[2]=='predict_sheets':
            dest_path = file_path.replace('data_complete/output/predict_sheets', 'data_resegment/input/Usefull_data')
            
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
