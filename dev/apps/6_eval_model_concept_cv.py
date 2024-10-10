import os
import numpy as np
from utils import UnetProcess, EvalModel
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt 
import pandas as pd 
import tensorflow as tf 
from tensorflow.keras.utils import CustomObjectScope
import cv2 

def iou(y_true, y_pred):
    
        '''
        Calculates the Intersection over Union (IoU) metric.
        
        
        Parameters
        ----------
        y_true : (array-like)
            True labels.
        y_pred : (array-like)
            Predicted labels.

        Returns
        -------
        float
            Jaccard index.
        """
        '''

        def f(y_true, y_pred):
            """
            Helper function to calculate IoU.

            Parameters
            ----------
            y_true : (array-like)
                True labels.
            y_pred : (array-like)
                Predicted labels.

            Returns
            -------
            float
                IoU value.
            """

            intersection = (y_true * y_pred).sum()
            union = y_true.sum() + y_pred.sum() - intersection
            x = (intersection + 1e-15) / (union + 1e-15)
            x = x.astype(np.float32)
            return x



        return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15

def dice_coef(y_true, y_pred):
        """
        Calculates the Sørensen–Dice index (Dice coefficient).

        Parameters
        ----------
        y_true : (array-like)
            True labels.
        y_pred : (array-like)
            Predicted labels.

        Returns
        -------
        float
            Sørensen–Dice index.
        """


        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))
        if union == 0:
            return 0
        dice_score = (2. * intersection) / union
        return round(dice_score.numpy(), 7)

def read_image(image):
        '''
        Reads and preprocesses an image.

        Parameters
        ----------
        image: (numpy.ndarray) 
            Input image.

        Returns:
        -----------
           Preprocessed image: (numpy.ndarray).
        '''
        x = np.load(image)
        # x = cv2.resize(x, (256, 256))
        ori_x = x
        # x = x/255.0
        # x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)   ## (1, 256, 256, 3)
        return  x
    
def read_mask(image):

        '''
        Reads and preprocesses a mask image.

        Parameters:
        -----------
        image: (numpy.ndarray)
            Input mask image.

        Returns:
        -----------
           Preprocessed mask image: (numpy.ndarray).
        '''

        x = np.load(image)
        # x = cv2.resize(x, (256, 256))
        ori_x = x
        # x = x/255.0
        x = x > 0.5
        # x = x.astype(np.int32)
        return  x



for fold in range(10):

        model_list = [
                (f'fold_{fold+1}_unet_afm_crossval_1_channels_only_afm_cosHeightSum_thresh_erode_CORRECTED.h5',
                f'models/data_per_fold_cosHeightSum_thresh_erode/test_data_fold_{fold}_unet_afm_crossval_1_channels_only_afm_cosHeightSum_thresh_erode_CORRECTED.csv'),
                ]

        
        for model, test_folder in model_list:
                
                df_test = pd.read_csv(test_folder)
                
                # if model != 'unet_afm_2_channels_only_optical.h5':
                #         continue
                print(f'get {model} validation... ')
                model_name = model
                
                if model == 'unet_afm_6_channels_226_images.h5':
                        model_name = 'UNet_AFM_6_channels_cv_100_epoch2.h5'
                        
                        
                preprocess_image = [read_image(img) for img in df_test['Files'].values]
                mask = [read_mask(mask) for mask in df_test['Masks'].values]

                df_list = []
                # good_afm/input/train_1_channels_only_afm_cosHeightSum_thresh_erode/opt_img_training/2021.07.27-20.24.13_channels_added.npy
                for i in tqdm(range(len(preprocess_image)), colour='#0000FF'):
                        process_data =  df_test['Files'].values[i].split(f'{os.sep}')[-1].split('.npy')[0]
                        # unetTrat =   UnetProcess(opt_image_path[i], preprocess_image_path[i], usefull_path[i], mask_path[i], model_path=f'models{os.sep}{model_name}') 
                        
                        with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}): 
                                model = tf.keras.models.load_model(f'models/{model_name}')
                        # usefull_path_unet = usefull_path[i].replace(f'data{os.sep}input{os.sep}Usefull_data{os.sep}',f'data{os.sep}output{os.sep}{unet_afm_fold}{os.sep}predict_sheets{os.sep}')
                        y_pred = model.predict(preprocess_image[i])
                        y_pred = np.squeeze(y_pred)
                        y_pred = y_pred > 0.5
                        
                        y_flatten = mask[i].flatten()
                        y_pred_flatten = y_pred.flatten()

                        
                        eval = EvalModel(f'unet',process_data, y_flatten, y_pred_flatten) 
                        scores = eval.get_metrics()
                        metric_df = eval.metrics_to_df(scores)
                        df_list.append(metric_df)
                final_metric_df = pd.concat(df_list, axis=0)
                final_metrics_melt = pd.melt(final_metric_df, id_vars=['Process Date','Model'], value_vars=['Precision', 'Recall', 'F1', 'Dice'], var_name = 'metrics', value_name = 'scores')
                final_metrics_melt.to_csv(f'validation_cv_metrics_{model_name.split(".")[0]}.csv')        