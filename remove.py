import pandas as pd 
import os 
import matplotlib.pyplot as plt
import numpy as np 


def clean_target(df, target='Generic Segmentation'):
    """Clean and map target column values to numeric values."""
    segment_mapping = {'Cytoplasm': 0, 'Nucleus': 1, 'Pericellular': 0, float('nan'): 0}
    df[target] = df[target].replace(segment_mapping)
    return df


data_complete = '/home/arthur/Desktop/data_complete/input/Usefull_data/'
count = 0
for file in os.listdir(data_complete):
        count += 1
        df = pd.read_csv(data_complete+file, sep='\t')
        if 'unet_prediction' in df.columns:
            df = df.drop(['unet_prediction'], axis=1)
            print(f'{count} files removed unet_prediction')
            df.to_csv(data_complete+file, sep='\t')