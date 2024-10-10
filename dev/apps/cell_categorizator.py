import pandas as pd 
import numpy as np 
import os 
import cv2 
import shutil 
from tqdm import tqdm

'''
Exemplos positivos:
* 2023.03.26-06.52.31
* 2022.04.07-22.01.27

Exemplos Aceitáveis:
* 2023.06.03-11.34.09
* 2023.03.29-22.09.46

Exemplos negativos:
* 2021.07.28-15.07.56
* 2023.03.23-10.43.28

'''
pd.set_option('future.no_silent_downcasting', True)


def apply_sobel_filter(image):
    
    # Aplicar o filtro de Sobel no eixo x
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    # Aplicar o filtro de Sobel no eixo y
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calcular a magnitude da borda
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Converter a magnitude para escala de 8 bits
    sobel_magnitude = np.uint8(sobel_magnitude / np.max(sobel_magnitude) * 255)
    
    
    return sobel_magnitude



def clean_target(df, target = 'Generic Segmentation'):
    """
    Cleans and maps values in the target column to numeric values.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with cleaned and mapped target values.
    """
    segment = {'Cytoplasm': 0 ,'Nucleus': 1, 'Pericellular': 0, float('nan'): 0}
    df[target] = df[target].replace(segment)
    return df

def zscore(df, column_name, substrate=True):
    
    df_without_substrate = df.loc[df['Segment']!= 'Substrate']
    mean_col = df_without_substrate[column_name].mean()
    std_col = df_without_substrate[column_name].std()
    
    if substrate:
        mean_col = df[column_name].mean()
        std_col = df[column_name].std()
    
    df[column_name] = (df[column_name] - mean_col) / std_col
    return df


def minMax_Scaler(df, column_name, substrate=True):
    df[column_name] = (df[column_name] - df[column_name].min())/ (df[column_name].max() - df[column_name].min())
    return df

def create_channel_by_df(df, column, dimension):
        """
        Creates a NumPy array from a DataFrame column and reshapes it to the specified dimension.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame.
        column : str
            The name of the column to create the channel from.
        dimension : tuple
            The desired dimensions of the resulting NumPy array in (height, width) format.

        Returns
        -------
        numpy.ndarray
            The created NumPy array.
        """
        return np.array(df[column]).reshape(dimension[0], dimension[1])


def add_new_channels(img, channels):
    """
    Adds new channels to the given image.

    Parameters
    ----------
    img : numpy.ndarray
        The input image.
    channels : list
        A list of channel images to be added.

    Returns
    -------
    numpy.ndarray
        The image with added channels.
    """
    num_channels = len(channels)
    new_img = np.zeros((img.shape[0], img.shape[1], num_channels))
    
    for i in range(num_channels):
        new_img[:,:,i] = channels[i]
    
    return new_img.astype(np.float32)

def equalize_img(image):
        """
        Applies histogram equalization to the image.

        Parameters
        ----------
        image : numpy.ndarray
            The input image.

        Returns
        -------
        numpy.ndarray
            The equalized image.
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=50, tileGridSize=(2,2))
        cl1 = clahe.apply(img)
        return cl1



def define_nucleus_region_by_afm(cos_height_sum, kernel_size = (3,3)):

        kernel = np.ones(kernel_size, dtype=np.uint8)

        cos_height_sum_01 = (cos_height_sum - cos_height_sum.min())/ (cos_height_sum.max() - cos_height_sum.min()) * 255
        
        ret, thresh_245_255 = cv2.threshold(cos_height_sum_01, 245,255, cv2.THRESH_BINARY)
    
        cos_heigh_erode_threshold_245_255_k5 = cv2.erode(-thresh_245_255, kernel,iterations = 1)
        cos_heigh_erode_threshold_245_255_k5 = -cos_heigh_erode_threshold_245_255_k5

        return cos_heigh_erode_threshold_245_255_k5

def resegment_files(*file_paths):
    """Copy files to resegmentation directories."""
    for file_path in file_paths:
        dest_path = file_path.replace('data_complete', 'good_afm')
        if file_path.split(f'{os.sep}')[2]=='predict_sheets':
            dest_path = file_path.replace('data_complete/output/predict_sheets', 'good_afm/input/Usefull_data')
        
        shutil.copy(file_path, dest_path)


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


image_list = ['2023.03.26-06.52.31', '2022.04.07-22.01.27', '2023.06.03-11.34.09','2023.03.29-22.09.46', '2021.07.28-15.07.56','2023.03.23-10.43.28']
count = 0 
for i, id in enumerate(process_date):

    try:     
        optical_image_path =f'data_complete/input/optical_images_resized/{id}_optico_crop_resized.png' 
        usefull_data_path = f'data_complete/input/Usefull_data/{id}_UsefullData.tsv'


        optical_image = cv2.imread(optical_image_path )
        x, y, z = optical_image.shape
        blue = optical_image[:,:,0]
        optical_image_equalized = equalize_img(optical_image)

        blue_flatten = blue.flatten()
        optical_image_equalized_flatten = optical_image_equalized.flatten()

        df_afm = pd.read_csv(usefull_data_path, sep='\t', low_memory=False)
        df_afm = clean_target(df_afm)
        df_afm['blue'] = blue_flatten
        df_afm['hist_equalized'] = optical_image_equalized_flatten


        nucleus_size = len(df_afm.loc[df_afm['Generic Segmentation']==1])

        features = ['Optico', 'Planned Height', 'blue', 'hist_equalized', 'Generic Segmentation']

        features_that_no_need_remove_substrate = ['Planned Height']
        for feature in features[1:]:
            substrate = False
            #Remove null or inf values
            mean_without_substrate = df_afm[feature].loc[df_afm['Segment'] != 'Substrate'].mean()
            df_afm[feature] = df_afm[feature].replace([np.inf, - np.inf, np.nan], mean_without_substrate)
            
            if feature in features_that_no_need_remove_substrate:
                #Calc Zscore by mean and std with substrate
                substrate = True
                
            #Calc Zscore by mean and std without substrate
            afm_info = minMax_Scaler(df_afm, feature, substrate=substrate)
            
            
        channels = []
        for feat in features:
            if feat == 'Optico':
                channels.append(optical_image)
                continue
            
            feature_image = create_channel_by_df(afm_info,  feat, [x,y]) 

            if feat == 'Planned Height':
                #create cosHeightSum
                feature_image =  (np.cos(feature_image) + feature_image) 

            if feat not in features_that_no_need_remove_substrate:
                #apply threshold
                feature_image[feature_image > 3] = 3
                feature_image[feature_image < -3] = -3
            channels.append(feature_image)


        mask = channels[-1]*2

        afm_region_nucleus = define_nucleus_region_by_afm(channels[1], kernel_size=(10,10))

        new_optical_img = add_new_channels(optical_image, channels[2:-1])

        nucleus_region = mask*afm_region_nucleus

        nucleu_size_in_afm = len(nucleus_region[nucleus_region>0])/nucleus_size
        # print(f"O Núcleo está {nucleu_size_in_afm*100:.2f}% dentro do AFM")
        
        if nucleu_size_in_afm>0.99:
            count+=1
            print(count)
            resegment_files(opt_images_resized[i], bw_images[i], txt_files[i], opt_images[i])
    except Exception as e:
        print(e)
        
print(f'{count} imagens possuiam acima de 99% do núcleo dentro do AFM')


