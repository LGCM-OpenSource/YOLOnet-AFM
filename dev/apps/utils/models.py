#unet
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope


#Voted System
import xgboost as xgb

import pandas as pd
import os
import matplotlib.pyplot as plt

import pickle
import numpy as np
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
import cv2 

from .image_treatment import ImageTrat
from .dataframe_treatment import DataFrameTrat
import traceback

#centroid
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

#Convex Hull
from skimage.morphology import convex_hull_image


'''
CLASS               LINE

Models              42
EvalModel           339
UnetProcess         364
PixelProcess        566

'''


class Models:
    """
    A class for managing various machine learning models.

    Attributes
    ----------
    unet_path : str
        Path to the UNet model.
    xgb_path : str
        Path to the XGBoost model.
    logreg_path : str
        Path to the Logistic Regression model.
    smooth : float
        A small constant for numerical stability.

    Methods
    -------
    __init__(model_name)
        Initializes the Models object.

    iou(y_true, y_pred)
        Calculates the Intersection over Union (IoU) metric.

    dice_coef(y_true, y_pred)
        Calculates the Sørensen–Dice index (Dice coefficient).

    dice_loss(y_true, y_pred)
        Calculates the Dice Coefficient Loss.

    precision(y_true, y_pred)
        Calculates the precision score.

    recall(y_true, y_pred)
        Calculates the recall score.

    f1_score(y_true, y_pred)
        Calculates the F1 score.

    jaccard(y_true, y_pred)
        Calculates the Jaccard score.

    predict(x)
        Makes predictions using the model.

    predict_proba(x)
        Predicts class probabilities using the model.

    predict_proba_result(y)
        Processes predicted probabilities to obtain class predictions.
    """

    unet_path = f'models{os.sep}unet_afm_optico_more_images.h5'
    xgb_path = f'models{os.sep}xgb_model.pkl'
    logreg_path = f'models{os.sep}logisticRegression_model.pkl'
    
    def __init__(self, model_name, model_path=False):
        """
        Initializes the Models object.

        Parameters
        ----------
        model_name : (str)
            The name of the model.
        """
        self.model_name = model_name.lower()
        if model_name == 'unet':
            with CustomObjectScope({'iou': self.iou, 'dice_coef': self.dice_coef}):
                if os.path.exists(self.unet_path): 
                    self.model = tf.keras.models.load_model(self.unet_path)
            if model_path:
             with CustomObjectScope({'iou': self.iou, 'dice_coef': self.dice_coef}): 
                self.model = tf.keras.models.load_model(model_path)

        elif model_name =='xgb': 
            self.model = pickle.load(open(self.xgb_path, 'rb')) 
            
        elif model_name == 'logreg':
            self.model = pickle.load(open(self.logreg_path, 'rb')) 
            
    def iou(self, y_true, y_pred):
    
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

    def dice_coef(self, y_true, y_pred):
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

    def dice_loss(self, y_true, y_pred):

        """
        Calculates the Dice Coefficient Loss.

        Parameters
        ----------
        y_true : (array-like)
            True labels.
        y_pred : (array-like)
            Predicted labels.

        Returns
        -------
        float
            Dice Coefficient Loss.
        """

        return 1.0 - self.dice_coef(y_true, y_pred)

    def precision(self, y_true, y_pred):
        """
        Calculates the precision score.

        Parameters
        ----------
        y_true : (array-like)
            True labels.
        y_pred : (array-like)
            Predicted labels.

        Returns
        -------
        float
            Precision score.
        """
        return round(precision_score(y_true, y_pred, average="binary",zero_division=0, pos_label=True), 7)
    
    def recall(self, y_true, y_pred):
        """
        Calculates the recall score.

        Parameters
        ----------
        y_true : (array-like)
            True labels.
        y_pred : (array-like)
            Predicted labels.

        Returns
        -------
        float
            Recall score.
        """
        return  round(recall_score(y_true, y_pred, average="binary",zero_division=0, pos_label=True), 7)
    
    def f1(self, y_true, y_pred):
        """
        Calculates the F1 score.

        Parameters
        ----------
        y_true : (array-like)
            True labels.
        y_pred : (array-like)
            Predicted labels.

        Returns
        -------
        float
            F1 score.
        """
        return round(f1_score(y_true, y_pred, average="binary",zero_division=0, pos_label=True),7)
    
    def jaccard(self, y_true, y_pred):
        """
        Calculates the Jaccard score.

        Parameters
        ----------
        y_true : (array-like)
            True labels.
        y_pred : (array-like)
            Predicted labels.

        Returns
        -------
        float
            Jaccard score.
        """
        return round(jaccard_score(y_true, y_pred, average="binary",zero_division=0, pos_label=True), 7) 
    
    def predict(self, x):
        """
        Makes predictions using the model.

        Parameters
        ----------
        x : (array-like)
            Input data for prediction.

        Returns
        -------
        (array-like)
            Predicted labels.
        """
        return self.model.predict(x)
    
    def predict_proba(self, x):
        """
        Predicts class probabilities using the model.

        Parameters
        ----------
        x : (array-like)
            Input data for prediction.

        Returns
        -------
        (array-like)
            Predicted class probabilities.
        """
        return self.model.predict_proba(x)
    
    def predict_proba_result(self, y):
        """
        Processes predicted probabilities to obtain class predictions.

        Parameters
        ----------
        y : (array-like)
            Predicted class probabilities.

        Returns
        -------
        (array-like)
            Predicted class labels after processing.
        """
        # final_df = pd.DataFrame(y,columns = ['Cytoplasm', 'Nucleus', 'Pericellular', 'Substrate'])
        
        final_df = pd.DataFrame(y,columns = ['Cytoplasm', 'Nucleus'])
        
        # df_condition = [(((final_df['Substrate']>final_df['Nucleus']) & (final_df['Substrate']>final_df['Cytoplasm']) & (final_df['Substrate']>final_df['Pericellular']))),
        #                     ((final_df['Nucleus']>0.78) & (final_df['Cytoplasm']<0.22)),
        #                     ((final_df['Cytoplasm']>0.78) & (final_df['Nucleus']<0.22)),
        #                     (((final_df['Pericellular']>final_df['Substrate']) & (final_df['Pericellular']>final_df['Cytoplasm']) & (final_df['Pericellular']>final_df['Nucleus'])))]

        df_condition = [
                        ((final_df['Nucleus']>0.78) & (final_df['Cytoplasm']<0.22)),
                        ((final_df['Cytoplasm']>0.78) & (final_df['Nucleus']<0.22)),
                        ]
        
        
        # df_choices  = [3,1,0,2]
        df_choices  = [0,1]  
        final_df['predict'] = np.select(df_condition,df_choices,default=0)
        return np.array(final_df['predict'])



class EvalModel:
    def __init__(self, model_name, process_date, y_true, y_pred):
        self.model = Models(model_name)
        self.process_date = process_date
        self.y_true = y_true
        self.y_pred = y_pred
        self.SCORE = []
        
    
    def get_metrics(self):
        f1_value = self.model.f1(self.y_true, self.y_pred)
        dice_value = self.model.dice_coef(self.y_true, self.y_pred)
        recall_value = self.model.recall(self.y_true, self.y_pred)
        precision_value = self.model.precision(self.y_true, self.y_pred)
        self.SCORE.append([self.model.model_name, self.process_date, precision_value, recall_value, f1_value, dice_value])
        return self.SCORE
    
    
    def metrics_to_df(self, score):
        df = pd.DataFrame(score, columns=['Model','Process Date', 'Precision', 'Recall', 'F1', 'Dice'])
        return df 
    
    
    

class UnetProcess:
    """
    A class to handle image processing using the UNet model.

    Attributes:
    -
        H (int): Height dimension for image resizing.
        W (int): Width dimension for image resizing.

    Methods:
    -
        __init__(self, optical_img_path, preprocess_path, mask_path=False)
            Initializes the ImageProcessor instance.

            Args:
                optical_img_path (str): Path to the optical image file.
                preprocess_path (str): Path to the preprocessed image file.
                mask_path (str, optional): Path to the mask image file. Defaults to False.

        read_image(self, image)
            Reads an image from the specified path.

            Args:
                image (str): Path to the image file.

        read_mask(self, image)
            Reads a mask image from the specified path.

            Args:
                image (str): Path to the mask image file.

        resize_prediction_to_original_size(self, y_pred)
            Resizes the prediction to the original image size.

            Args:
                y_pred: The predicted image.

        unet_predict(self, save_path, save_unet_path=False)
            Performs UNet model prediction on the input image.

            Args:
                save_path (str): Path to save the predicted image.
                save_unet_path (str, optional): Path to save the UNet image. Defaults to False.
    """
    H = 256
    W = 256
    
    def __init__(self, optical_img_path, preprocess_path, usefull_path, mask_path=False, model_path=False):
        """
        Initializes the ImageProcessor instance.

        Args:
        optical_img_path: (str)
            Path to the optical image file.
        preprocess_path: (str)
            Path to the preprocessed image file.
        mask_path: (str, optional )
            Path to the mask image file. Defaults to False.
        """
        self.model = Models('unet')
        if model_path:
            self.model = Models('unet', model_path)

        self.df_afm = DataFrameTrat(usefull_path)
        self.preprocess_image = ImageTrat(preprocess_path)
        self.opt_image = ImageTrat(optical_img_path)
        self.mask = mask_path
        if self.mask:
            self.mask = ImageTrat(mask_path)
            
    def read_image(self, image):
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
        x = image
        x = self.preprocess_image.resize(x, (self.W, self.H))
        ori_x = x
        # x = x/255.0
        # x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)   ## (1, 256, 256, 3)
        return ori_x, x
    
    def read_mask(self, image):

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

        x = image
        x = self.preprocess_image.resize(x, (self.W, self.H))
        ori_x = x
        # x = x/255.0
        x = x > 0.5
        # x = x.astype(np.int32)
        return ori_x, x


    def resize_prediction_to_original_size(self, y_pred):
        '''
        Resizes a predicted image to its original size.

        Parameters:
        -----------
        y_pred: (numpy.ndarray)
            Predicted image.

        Returns:
        -----------
            Resized predicted image and flattened version: Tuple[numpy.ndarray, numpy.ndarray].
        '''
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        y_pred = np.expand_dims(y_pred, axis=-1) * 255.0

        x_dim,y_dim = self.opt_image.dimensions(matrix=False)
        
        # Removing external outliers
        y_pred = cv2.morphologyEx(y_pred, cv2.MORPH_OPEN, kernel)

        # y_pred = cv2.dilate(y_pred,kernel,iterations = 1)

        # Filling empty spaces inside the nucleus
        y_pred = cv2.morphologyEx(y_pred, cv2.MORPH_CLOSE, kernel)
        y_pred  = cv2.resize(y_pred, (y_dim, x_dim))
        
        y_pred[y_pred>0] = 1
        y_pred_flatten = y_pred.copy()
        y_pred_flatten[y_pred_flatten>0] = 1
        y_pred_flatten = y_pred_flatten.flatten()
        
        return y_pred, y_pred_flatten
    

    def get_count(self, predict_sheet):
        try:
            nucleus_size = predict_sheet['unet_prediction'].value_counts()[1]
            return nucleus_size 
        except: 
            return 0

    def count_objects(self, predict_sheet):
        
        x, y = self.opt_image.dimensions()
        image = np.array(predict_sheet['unet_prediction']).reshape(x,y)
        # Rotula os componentes conectados na imagem binária
        labeled_image = measure.label(image)
        
        # Retorna o número de objetos detectados
        return len(set(labeled_image.flatten())) - 1 # -1 para remover o fundo

    
    
    def show_predict(self, process_date, y, y_pred, topography, save_path, metrics):
        precision = metrics['Precision'][0]
        recall = metrics['Recall'][0]
        dice = metrics['Dice'][0]
        
        
        optical_image = self.opt_image.image()
        optical_image = cv2.cvtColor(optical_image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(15, 6))

        plt.subplot(2, 3, 1)
        plt.imshow(optical_image)
        plt.title(process_date)

        plt.subplot(2, 3, 2)
        plt.imshow(optical_image, alpha=0.7)  
        plt.imshow(y, alpha=0.2, cmap='jet')  
        plt.title("Stardist with Optical Overlay")

        plt.subplot(2, 3, 3)
        plt.imshow(optical_image, alpha=0.7)  
        plt.imshow(y_pred, alpha=0.2, cmap='jet')  
        plt.title("Y_pred with Optical Overlay")

        plt.subplot(2, 3, 4)
        plt.imshow(topography)
        plt.title("Norm Height")

        plt.subplot(2, 3, 5)
        plt.imshow(optical_image, alpha=0.7)  
        plt.imshow(y_pred, alpha=0.2, cmap='jet')  
        plt.imshow(y, alpha=0.2, cmap='gray')  
        plt.title("Y_pred, Y & Optical Overlay")

        plt.subplot(2, 3, 6)
        plt.axis('off')
        plt.text(0.95, 0.05, f'Precision: {precision:.2f}\nRecall: {recall:.2f}\nDice: {dice:.2f}', 
         verticalalignment='bottom', horizontalalignment='right', 
         transform=plt.gca().transAxes, 
         color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))
        
        
        
        plt.savefig(save_path)
        plt.close()
        

    def unet_predict(self, save_path='', usefull_path=False, save_unet_path=False):
        '''
        Performs the UNet prediction and saves the results.

        Parameters:
        -----------
        save_path: (str)
            Path to save the prediction results.
        save_unet_path: (str or False)
            Path to save the UNet prediction visualization.

        Returns:
        -----------
            None
        '''
        try:
            optical_image = self.opt_image.image()
            optical_image_res = cv2.resize(optical_image, (256,256))
            ori_x, x = self.read_image(self.preprocess_image.image(matrix=True))
            ori_y, y = self.read_mask(self.mask.image(matrix=True))
            
            '''prediction'''
            y_proba = self.model.predict(x) 
            y_pred = y_proba[0] > 0.5
            y_proba = np.squeeze(y_proba)
            y_pred = np.squeeze(y_pred)
            y_pred = y_pred > 0.5
            
            #Resize prediction from 256x256 to original image size
            y_pred_resized, y_pred_flatten = self.resize_prediction_to_original_size(y_pred)
            y_resized, y_flatten = self.resize_prediction_to_original_size(y)            
            if usefull_path:
                save_usefulll_path = usefull_path.replace(f'data_complete{os.sep}input{os.sep}Usefull_data{os.sep}',f'data_complete{os.sep}output{os.sep}predict_sheets{os.sep}') 
                df = DataFrameTrat(usefull_path)
                df_afm = df.df
                
                # df_afm['unet_prediction'] = y_pred_flatten
                
                verify_count = self.get_count(df_afm)
                verify_objects = self.count_objects(df_afm)
                
                
            '''transpose prediction to optical image'''
            # y = self.preprocess_image.predicted_nucleus_to_image(optical_image, y_resized)
            # y_pred = self.preprocess_image.predicted_nucleus_to_image(optical_image, y_pred_resized)

            '''Save results'''
            # if save_unet_path:
            #     fig = plt.figure(figsize=(15,7))
            #     fig.patch.set_facecolor('white')
            #     ax1 = fig.add_subplot(1,2,1)
            #     ax1.set_title('Original optico')
            #     plt.imshow(self.preprocess_image.image, cmap='gray')

            #     ax2 = fig.add_subplot(1,2,2)
            #     ax2.set_title('unet predict')
            #     plt.imshow(y_pred, cmap='gray')

             #     fig.savefig(save_path)
                
            # if not verify_count < 177*0.2 or verify_objects > 1:   
                
            # df_afm.to_csv(save_usefulll_path, sep='\t', index=False)
            # self.preprocess_image.save_image(save_path, y_pred)
                
            #     return y, y_proba, False
            
            # to run pixel segmentation  
            # plt.subplot(231); plt.imshow(optical_image);
            # plt.subplot(232); plt.imshow(y_resized);
            # plt.subplot(233); plt.imshow(y_pred_resized);


            
            # plt.show()
            return y_resized, y_pred_resized, True
        except Exception:
            print(traceback.format_exc())


class PixelProcess:
    """
    A class for pixel-based image processing and segmentation.

    Methods
    -------
    __init__(usefull_data_path, optical_image_path)
        Initializes the PixelProcess object.

    vector_to_image(vector, dimensions)
        Converts a vector into an image with specified dimensions.

    centroid_calc(segmentation)
        Calculates the centroid of a segmented image.

    eccentricity_calc(segmentation)
        Calculates the eccentricity of a segmented image.

    convex_hull(predict_nucleus)
        Applies the convex hull filter to a predicted nucleus.

    convex_to_df(convex, model)
        Converts a convex nucleus into a DataFrame.

    check_prediction(predict_matrix)
        Checks if a predicted image can be filtered by convex hull.

    get_difference(nucleus_count, eccentricity)
        Calculates the difference between average and predicted nucleus size.

    distance_from_centroid(predict_df, predict_matrix, centroid)
        Calculates the distance of each pixel from the centroid.

    plot_segmentation(fig, position, title, image)
        Plots a segmentation image on a given figure.

    interquartile_calc(predict_df, model_name)
        Calculates the interquartile filter for predicted DataFrame.

    pixel_predict(save_path, save_pixel_path=False)
        Performs pixel-based prediction and saves the results.

    Attributes
    ----------
    features : list
        List of feature names.

    """
    features = ['Process Date', 
                'Height Position',
                'Planned Height',
                'Norm Height',
                'MaxPosition_F0500pN',
                'YM_Fmax0500pN',
                'Generic Segmentation']
    
    def __init__(self, usefull_data_path, optical_image_path):
        '''
        Initializes the PixelProcess object.

        Parameters:
        -----------
        usefull_data_path : str
            Path to the useful data.
        optical_image_path : str
            Path to the optical image.
        '''
        self.dict_model = {
                            'xgb':Models('xgb'),
                            'xgb_proba':Models('xgb'), 
                            'logisticRegression': Models('logreg'), 
                            'logisticRegression_proba': Models('logreg')
                           }
        self.df_afm_normalized = DataFrameTrat(usefull_data_path)
        self.optical_image = ImageTrat(optical_image_path)

    def vector_to_image(self, vector, dimensions):
        '''
        Converts a vector into an image with specified dimensions.

        Parameters:
        -----------
        vector : numpy.ndarray
            Input vector.
        dimensions : tuple
            Dimensions of the output image.

        Returns
        -------
        numpy.ndarray
            Image representation of the vector.
        '''
        vec_to_np = np.array(vector)
        return np.reshape(vec_to_np, dimensions)

    def centroid_calc(self, segmentation):
        '''
        Calculates the centroid of a segmented image.

        Parameters:
        -----------
        segmentation: numpy.ndarray
            Segmented image.

        Returns:
        ---------
        tuple 
            Centroid coordinates (y, x).
        '''
        thresholded = filters.apply_hysteresis_threshold(segmentation,0.03,0.99).astype(int) 
        labeled = ndi.label(thresholded)[0]

        largest_nonzero_label = np.argmax(np.bincount(labeled[labeled > 0]))

        binary = labeled == largest_nonzero_label
        skeleton = morphology.skeletonize(binary)

        centroid = measure.centroid(labeled > 0)
        return centroid
    
    def eccentricity_calc(self, segmentation):
        '''
        Calculates the eccentricity of a segmented image.

        Parameters:
        ---------
        segmentation: numpy.ndarray
            Segmented image.

        Returns:
        ---------
        Eccentricity: float
            value.
        '''
        thresholded = filters.apply_hysteresis_threshold(segmentation,0.03,0.99).astype(int) 
        labeled = ndi.label(thresholded)[0]

        # largest_nonzero_label = np.argmax(np.bincount(labeled[labeled > 0]))
        props = measure.regionprops(labeled, segmentation)
        return props[0].eccentricity


    def convex_hull(self, predict_nucleus: np.ndarray):
        '''
        Applies the convex hull filter to a predicted nucleus.

        Parameters:
        ---------
            predict_nucleus: numpy.ndarray
                Predicted nucleus.

        Returns:
        ---------
        Nucleus: numpy.ndarray
            Nucleus after applying the convex hull filter.
        '''
        try:
            figure = np.array(predict_nucleus)
            figure_hull = convex_hull_image(figure).astype(float)
            return figure_hull
        except:
            return predict_nucleus


    def convex_to_df(convex: np.ndarray, model):
        '''
        Converts a convex nucleus into a DataFrame.

        Parameters:
        ---------
        convex: numpy.ndarray
            Convex nucleus.
        model: str
            Model name.

        Returns:
        ---------
        DataFrame: pandas.DataFrame
            DataFrame with the filtered segmentation.
        '''
        nucleus_list=[]
        for line in convex:
            for column in line:
                nucleus_list.append(column)
        df = pd.DataFrame(nucleus_list, columns=[f'{model}_predict'])
        return df


    #check if image is can be filtred by convexHull
    def check_prediction(self, predict_matrix: np.ndarray):
        '''
        Checks if a predicted image can be filtered by convex hull.

        Parameters:
        ---------
        predict_matrix: numpy.ndarray 
            Predicted image.    

        Returns:
        ---------
        Processed predicted image: numpy.ndarray & count: int.
        '''
        count= np.count_nonzero(predict_matrix==1)
        
        if count <=2:
            predict_matrix = np.ones((len(predict_matrix),len(predict_matrix[0])))
        return predict_matrix, count

    def get_difference(self, nucleus_count, eccentricity):
        '''
        Calculates the difference between average and predicted nucleus size.

        Parameters:
        ---------
        nucleus_count: int 
            Predicted nucleus size.
        eccentricity: float
            Eccentricity value.

        Returns:
        ---------
        Pound difference: float.
        '''
        # Calculate nucleus difference
        if 131 <= nucleus_count <= 223:
            difference = 177 - nucleus_count
        elif nucleus_count > 223:
            difference = 223 - nucleus_count
        elif nucleus_count < 131:
            difference = 131 - nucleus_count
        difference = abs(difference)
        
        # Calculate eccentricity difference
        if 0.6 <= eccentricity <= 0.85:
            eccentricity_dif = 0.73 - eccentricity
        elif eccentricity > 0.85:
            eccentricity_dif = 0.85 - eccentricity
        elif eccentricity < 0.6:
            eccentricity_dif = 0.6 - eccentricity
        eccentricity_dif = abs(eccentricity_dif)
        
        # Define weights
        weight_eccentricity = 103
        weight_count = 27
        
        # Calculate pound difference
        pound_difference = (weight_eccentricity * eccentricity_dif + weight_count * difference) / (weight_eccentricity + weight_count)
        return pound_difference
    
    def distance_from_centroid(self, predict_df, predict_matrix, centroid):
        '''
        Calculates the distance of each pixel from the centroid.

        Parameters:
        ---------
        predict_df: pandas.DataFrame
            Predicted DataFrame.
        predict_matrix: numpy.ndarray
            Predicted image.
        centroid: tuple
            Centroid coordinates (y, x).

        Returns:
        ---------
        Updated DataFrame with distance values: pandas.DataFrame.
        '''
        index=0
        for x in range(len(predict_matrix)):
            for y in range(len(predict_matrix[0])):
                if predict_matrix[x][y]==1:
                    distance = np.linalg.norm((x,y)-centroid)
    #                 distance = dist((x,y),centroid)
                    predict_df.at[index, 'distance_from_centroid']=distance
                else:
                    predict_df.at[index, 'distance_from_centroid']=0
                index += 1
        return predict_df
    
    def plot_segmentation(self, fig, position, title, image):
        '''
        Plots a segmentation image on a given figure.

        Parameters:
        -------
        fig: matplotlib.figure.Figure
            Figure object.
        position: list
            Position of the subplot.
        title: str 
            Subplot title.
        image: numpy.ndarray
            Image data.

        Returns:
        -------
            None
        '''
        ax1 = fig.add_subplot(position[0], position[1], position[2])
        ax1.set_title(title)
        plt.imshow(image, cmap='gray')
        
    def interquartile_calc(self, predict_df, model_name):
        '''
        Calculates the interquartile filter for predicted DataFrame.

        Parameters:
        --------
        predict_df: pandas.DataFrame 
            Predicted DataFrame.
        model_name: str
            Model name.

        Returns:
        --------
        DataFrame after applying the filter: pandas.DataFrame.
        '''
        distance_from_centroid = 'distance_from_centroid'

        percentile25 = predict_df[distance_from_centroid].loc[predict_df[distance_from_centroid]>0].quantile(0.25)
        percentile75 = predict_df[distance_from_centroid].loc[predict_df[distance_from_centroid]>0].quantile(0.75)

        #Calc interval interquartile
        iqr = percentile75-percentile25
        upper_limit = percentile75 + 1.5 * iqr

        condition = [((predict_df[model_name]==1)&(predict_df[distance_from_centroid]>=upper_limit)),((predict_df[model_name]==1)&(predict_df[distance_from_centroid]<upper_limit))]
        choice = (0,1)
        predict_df[model_name]=np.select(condition,choice,default=0)
        predict_df = predict_df.drop([distance_from_centroid], axis=1)
        return predict_df      
    
    def pixel_predict(self, save_path, usefull_path, save_pixel_path=False):
        '''
        Performs pixel-based prediction and saves the results.

        Parameters:
        ---------
        save_path: str
            Path to save the prediction results.
        save_pixel_path: str or False
            Path to save the pixel prediction visualization.

        Returns:
        ---------
            None
        '''
        
        original_df = pd.read_csv(usefull_path, index_col=0, sep='\t')
        
        df_afm = self.df_afm_normalized.df
        df_afm = df_afm[self.features]
                
        opt_image = self.optical_image.image
        dimensions = self.optical_image.dimensions()
        
        dict_predict = {}
        dict_predict['votation']=[]
        
        for model_name in self.dict_model.keys():
            tmp_df = original_df.copy()
            
            check_proba = model_name.split('_')
            
            x = df_afm.drop(['Process Date'], axis=1)
            if self.df_afm_normalized.target:
                y = df_afm[self.df_afm_normalized.target]
                x = x.drop([self.df_afm_normalized.target], axis=1)
            
            #drop null, inf, -inf values
            x = x.apply(lambda col: col.replace((np.inf, -np.inf, np.nan), col.mean()).reset_index(drop=True))
            
            if check_proba[-1] =='proba':
                predicts = self.dict_model[model_name].predict_proba(x)
                predicts = self.dict_model[model_name].predict_proba_result(predicts)
            else: 
                predicts = self.dict_model[model_name].predict(x)
                
            # model_name = model_name+'_prediction'
            tmp_df[model_name] = predicts
            
            nucleus = tmp_df[model_name]==1
            nucleus_predict = self.vector_to_image(nucleus, dimensions)
            
            #check if prediction had more than 2 pixel as nucleus and return nucleus size
            nucleus_predict, predict_count = self.check_prediction(nucleus_predict)
            
            predict_centroid = self.centroid_calc(nucleus_predict)
            predict_eccentricity = self.eccentricity_calc(nucleus_predict)
            
            #transpose nucleus to image
            nucleus_img = self.optical_image.predicted_nucleus_to_image(opt_image, nucleus_predict)
            
            
            #remove outliers
            tmp_df = self.distance_from_centroid(tmp_df, nucleus_predict, predict_centroid)
            tmp_df = self.interquartile_calc(tmp_df, model_name)
            
            #NPredicts without outliers
            nucleus = tmp_df[model_name]==1
            nucleus_predict = self.vector_to_image(nucleus, dimensions)
            #check if prediction had more than 2 pixel as nucleus and return nucleus size
            nucleus_predict, predict_count = self.check_prediction(nucleus_predict)
            
            #apply ConvexHull
            convex_nucleus = self.convex_hull(nucleus_predict)
            convex_eccentricity = self.eccentricity_calc(nucleus_predict)
            convex_nucleus_img = self.optical_image.predicted_nucleus_to_image(opt_image, convex_nucleus)
            # convex_predict_df = self.convex_to_df(convex_nucleus, model_name)

            nucleus_difference = self.get_difference(predict_count, convex_eccentricity)
            dict_predict['votation'].append((nucleus_difference, model_name, convex_nucleus_img, predict_centroid, tmp_df))
            
        best_model = min(dict_predict['votation'])
        best_model_name = best_model[1]
        best_img = best_model[2]
        best_segment = best_model[4]
        # best_centroid = best_model[3]

        # Save results
        save_path = save_path.replace('.png', f'_{best_model_name}.png')
        save_usefull_path = usefull_path.replace(f'data{os.sep}input{os.sep}Usefull_data{os.sep}', f'data{os.sep}output{os.sep}predict_sheets{os.sep}')
        
        self.optical_image.save_image(save_path, best_img)
        best_segment.to_csv(save_usefull_path, index=False, sep='\t')

        if save_pixel_path:
            save_pixel_path = save_pixel_path.replace(".png", "")
            fig = plt.figure(figsize=(15,7))
            fig.patch.set_facecolor('white')

            best_img = cv2.cvtColor(best_img, cv2.COLOR_BGR2RGB)
            opt_image = cv2.cvtColor(opt_image, cv2.COLOR_BGR2RGB)

            self.plot_segmentation(fig, [1,2,1], 'Original optico',opt_image)
            self.plot_segmentation(fig, [1,2,2], best_model_name, best_img)

            fig.savefig(f'{save_pixel_path}_{best_model_name}_voted.png')
            plt.close(fig)
 