import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import cv2 
import os 

def zscore(df, column_name, substrate=True):
    
    df_without_substrate = df.loc[df['Segment']!= 'Substrate']
    mean_col = df_without_substrate[column_name].mean()
    std_col = df_without_substrate[column_name].std()
    
    if substrate:
        mean_col = df[column_name].mean()
        std_col = df[column_name].std()
    
    df[column_name] = (df[column_name] - mean_col) / std_col
    return df

def normalize_and_reshape(df, column, dim, limiar=[-3, 3]):
    img_by_feature = np.array(df[column]).reshape((dim[0],dim[1]))
    if feature == 'Planned Height':
        img_by_feature = -img_by_feature
    img_by_feature[img_by_feature > limiar[1]] = limiar[1]
    img_by_feature[img_by_feature < limiar[0]] = limiar[0]
    return img_by_feature

opt_images_path = f'data{os.sep}input{os.sep}optical_images_resized{os.sep}'
usefull_data_path = f'data{os.sep}input{os.sep}Usefull_data{os.sep}'

opt_files = [os.path.join(opt_images_path, files) for files in os.listdir(opt_images_path)]
usefull_files = [os.path.join(usefull_data_path, files) for files in os.listdir(usefull_data_path)]


for opt, tsv in zip(opt_files, usefull_files):
    #open files
    opt_image = cv2.imread(opt)
    opt_image = cv2.cvtColor(opt_image, cv2.COLOR_BGR2RGB)
    blue = opt_image[:,:,2]
    
    df_afm = pd.read_csv(tsv, sep='\t')
    
    
    #get dimensions
    x, y, z = opt_image.shape
    
    fig, axs = plt.subplots(1,6, figsize = (15,8))
    feature_images = [opt_image]
    features_list = ['Optical Image','MaxPosition_F1500pN', 'YM_Fmax1500pN', 'YM_Fmax0300pN', 'Planned Height', 'All_features']
    for i, feature in enumerate(features_list):
        substrate = False
        ignore_list = [0,5]
        if i not in ignore_list:
            if feature == 'Planned Height':
                substrate = True
            
            df_afm = zscore(df_afm, feature, substrate=substrate)
        
            feature_image = normalize_and_reshape(df_afm, feature, (x,y))
            feature_images.append(feature_image)
        
        if i ==5:
            
            feature_image = sum(feature_images[1:-1])
            feature_image += blue 
            feature_images.append(feature_image)
        
        
        axs[i].imshow(feature_images[i], cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(features_list[i])
    plt.show()
     
    
    
