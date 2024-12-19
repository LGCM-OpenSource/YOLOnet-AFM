#unet
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope


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
'''


class Models:
    """
    A class for managing various machine learning models.

    Attributes
    ----------
    unet_path : str
        Path to the UNet model.
   
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

    """

    unet_path = f'models{os.sep}unet_afm_optico_more_images.h5'

    
    def __init__(self, model_name, model_path=False):
        """
        Initializes the Models object.

        Parameters
        ----------
        model_name : (str)
            The name of the model.
        """
        self.model_name = model_name.lower()
        with CustomObjectScope({'iou': self.iou, 'dice_coef': self.dice_coef}):
            if os.path.exists(self.unet_path): 
                    self.model = tf.keras.models.load_model(self.unet_path)
        if model_path:
             with CustomObjectScope({'iou': self.iou, 'dice_coef': self.dice_coef}): 
                self.model = tf.keras.models.load_model(model_path)

            
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
    



class EvalModel:
    def __init__(self, model_name, y_true, y_pred, process_date=''):
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
        x = np.expand_dims(x, axis=0)   ## (1, 256, 256, 2)
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


    def resize_prediction_to_original_size(self, y):
        '''
        Resizes a predicted image to its original size.

        Parameters:
        -----------
        y: (numpy.ndarray)
            Predicted image.

        Returns:
        -----------
            Resized predicted image and flattened version: Tuple[numpy.ndarray, numpy.ndarray].
        '''
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

        x_dim,y_dim = self.opt_image.dimensions(matrix=False)
        
        y_opening_filter = cv2.morphologyEx(y.astype(np.uint8), cv2.MORPH_OPEN, kernel)

        y_closing_filter = cv2.morphologyEx(y_opening_filter, cv2.MORPH_CLOSE, kernel)
        y_resized  = cv2.resize(y_closing_filter, (y_dim, x_dim))
        
        return y_resized
    

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
        # Extração de métricas
        precision = metrics['Precision'][0]
        recall = metrics['Recall'][0]
        dice = metrics['Dice'][0]

        # Obtenção e processamento da imagem óptica
        optical_image = self.opt_image.image()
        optical_image = cv2.cvtColor(optical_image, cv2.COLOR_BGR2RGB)

        # Configuração da figura
        plt.figure(figsize=(10, 6))

        # Subplot 1: Imagem óptica original
        plt.subplot(2, 3, 1)
        plt.imshow(optical_image)
        plt.title(process_date)
        plt.axis('off')

        # Subplot 2: Stardist com sobreposição óptica
        plt.subplot(2, 3, 2)
        plt.imshow(optical_image, alpha=0.7)
        plt.imshow(y, alpha=0.5)
        plt.title("Stardist Golden Standard")
        plt.axis('off')

        # Subplot 3: Y_pred com sobreposição óptica
        plt.subplot(2, 3, 3)
        plt.imshow(optical_image, alpha=0.7)
        plt.imshow(y_pred, alpha=0.5, cmap='jet')
        plt.title("Y_pred")
        plt.axis('off')

        # Subplot 4: Topografia normalizada
        plt.subplot(2, 3, 4)
        plt.imshow(topography)
        plt.title("Norm Height")
        plt.axis('off')

        # Subplot 5: Combinação de Y_pred, Y e imagem óptica
        plt.subplot(2, 3, 5)
        plt.imshow(optical_image, alpha=0.7)
        plt.imshow(y_pred, alpha=0.5, cmap='jet')
        plt.imshow(y, alpha=0.5)
        plt.title("Y_pred and Y Overlay")
        plt.axis('off')

        # Subplot 6: Métricas
        plt.subplot(2, 3, 6)
        plt.axis('off')
        metrics_text = (f'Precision: {precision:.2f}\n'
                        f'Recall: {recall:.2f}\n'
                        f'Dice: {dice:.2f}')
        plt.text(
            0.95, 0.05, metrics_text,
            verticalalignment='bottom', horizontalalignment='right',
            transform=plt.gca().transAxes,
            color='white', fontsize=10,
            bbox=dict(facecolor='black', alpha=0.5)
        )

        # Salvar a figura
        plt.savefig(save_path)
        plt.close()

    def visualize_prediction(self, prediction_path):
        original_image = self.opt_image.image()
        prediction = cv2.imread(prediction_path)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Optical Image')
        plt.imshow(original_image[:,:,::-1], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title('Predict')
        plt.imshow(prediction[:,:,::-1], cmap='gray')
        plt.show()
        
    def save_predict(self, y_pred, save_path):
        fig, ax = plt.subplots(figsize=(self.opt_image.image().shape[1] / 100, 
                                        self.opt_image.image().shape[0] / 100), dpi=100)
        
        ax.imshow(self.opt_image.image()[:,:,::-1], alpha=0.7)
        ax.imshow(y_pred, cmap='gray', alpha=0.5)
        ax.axis('off') 
        
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig) 
    
    def model_predict(self, model, x):
            y_proba_predicts = model.predict(x) 
            y_pred = y_proba_predicts[0] > 0.5
            y_proba_predicts = np.squeeze(y_proba_predicts)
            y_pred = np.squeeze(y_pred)
            
            return y_pred, y_proba_predicts
        
    def unet_predict(self, model, prc_curve=False):
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
            ori_x, x = self.read_image(self.preprocess_image.image(matrix=True))
            ori_y, y = self.read_mask(self.mask.image(matrix=True))
            
            y_pred, y_proba = self.model_predict(model, x)
                        
            if prc_curve:
                return y_proba
                
            return  y_pred
        except Exception:
            print(traceback.format_exc()) 