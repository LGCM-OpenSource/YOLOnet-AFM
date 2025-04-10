import numpy as np
import cv2
import traceback
import os
from .dataframe_treatment import DataFrameTrat
import matplotlib.pyplot as plt
import logging # <-- Import logging
from .logger import get_logger, setup_logger # <-- Import logger functions
from .data_path import create_dir
'''
CLASS               LINE

ImageTrat             25
CropImages           230
GenerateAFMOptico    357

'''


#Esta classe inclui todos os métodos para o recorte da imagem optica a partir da imagem BW
class ImageTrat:
    """
        A class used for image processing and manipulation.

    ...

    Attributes
    ----------
        self : ImageTrat
            An instance of the ImageTrat class.
        img_path : str
            The path to the image file.

        Methods
        -------
        image : numpy.ndarray
            Returns the loaded image using OpenCV.

        dimensions(txt_path=False)
            Retrieves the dimensions of the image.
        image_channels(image)
            Separates the image channels into blue, green, and red.
        flip(image)
            Flips the given image vertically.
        resize(image, dimension)
            Resizes the given image to the specified dimensions.
        equalize_img(image)
            Applies histogram equalization to the image.
        save_image(save_path, new_image)
            Saves the given image to the specified path.
    """
    def __init__(self, img_path):
        """
        Constructs an instance of the ImageTrat class.

        Parameters
        ----------
            img_path : str
                The path to the image file.
        """
        self.img_path = img_path

    def image(self, matrix=False):
        """
        Returns the loaded image using OpenCV.

        Returns
        -------
        numpy.ndarray
            The loaded image.
        """
        if matrix:
            return np.load(self.img_path).astype(np.float32)
        return cv2.imread(self.img_path)
    
    @property
    def mask(self):
        """
        Returns the loaded mask using OpenCV.

        Returns
        -------
        numpy.ndarray
            The loaded image.
        """
        return cv2.imread(self.img_path, 0)
    
    
    def dimensions(self, txt_path=False, matrix=False):
        """
        Retrieves the dimensions of the image.

        Parameters
        ----------
        txt_path : str, optional
            The path to the text file containing dimensions, by default False.

        Returns
        -------
        tuple
            The dimensions of the image in (height, width) format.
        """
        if txt_path:
            open_spreadsheets = open(txt_path)
            read_spreadsheets = open_spreadsheets.readlines()
            x_dim = None
            y_dim = None
            for lines in read_spreadsheets:
                line = lines.split(' ')
                if line[0]!= '#':
                    break
                elif line[1]=='jLength:':
                    x_dim = lines.split(' ')
                    x_dim = int(x_dim[2])
                elif line[1] == 'iLength:':
                    y_dim = lines.split(' ')
                    y_dim = int(y_dim[2])
            return (y_dim, x_dim)
        else:
            if len(self.image(matrix=matrix).shape)>2:
                x_dim, y_dim, _ = self.image(matrix=matrix).shape
            else:
                x_dim, y_dim = self.image(matrix=matrix).shape
                
            return (x_dim, y_dim)
    
    def image_channels(self, image):
        """
        Separates the image channels into blue, green, and red.

        Parameters
        ----------
        image : numpy.ndarray
            The input image.

        Returns
        -------
        tuple
            A tuple containing blue, green, and red channel images.
        """
        blue = image[:,:,0]
        green = image[:,:,1]
        red = image[:,:,2]
        return blue, green, red
    
    def flip(self, image):
        """
        Flips the given image vertically.

        Parameters
        ----------
        image : numpy.ndarray
            The input image.

        Returns
        -------
        numpy.ndarray
            The vertically flipped image.
        """
        return cv2.flip(image, 0)
    
    def resize(self, image, dimension):
        """
        Resizes the given image to the specified dimensions.

        Parameters
        ----------
        image : numpy.ndarray
            The input image.
        dimension : tuple
            The new dimensions in (width, height) format.

        Returns
        -------
        numpy.ndarray
            The resized image.
        """
        return cv2.resize(image, dimension, interpolation=cv2.INTER_LINEAR)  
    
    
    def equalize_img(self, image):
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
    
    def predicted_nucleus_to_image(self, image, y_pred: np.ndarray):
        '''
        Parameters
        ----------
        opt_image: optical image
        y_pred: y of prediction
        
        Return
        -------

        Returns the optical image with the filtered predictions superimposed on it
        :return np.ndarray new_img

        '''
        try:
            new_img = image.copy()
            x_indices, y_indices = np.where(y_pred == 1)
            new_img[x_indices, y_indices] = (0, 0, 255)

            return new_img
        except Exception:
            print(traceback.format_exc())

    def save_image(self,save_path, new_images):
        """
        Saves the given image to the specified path.

        Parameters
        ----------
        save_path : str
            The path where the image should be saved.
        new_image : numpy.ndarray
            The image to be saved.

        Returns
        -------
        bool
            True if the image was successfully saved, False otherwise.
        """
        # flipped = cv2.flip(new_images, 0)
        return  cv2.imwrite(save_path, new_images)
    
    
                 

class CropImages:
    """
    A class used for cropping and processing images.

    ...

    Attributes
    ----------
    self : CropImages
        An instance of the CropImages class.
    opt_image : ImageTrat
        An instance of the ImageTrat class for the optical image.
    bw_image : ImageTrat
        An instance of the ImageTrat class for the black and white image.

    Methods
    -------
    pixel_definition()
        Retrieves the coordinates of the area of interest to be cropped.
    getMinMaxPositionByPixelList(list_pixel)
        Determines the minimum and maximum positions from the obtained pixel coordinates.
    run_crop_image(save_path, txt_path)
        Applies cropping and processing to the optical image and saves the result.
    """
    def __init__(self, optical_image_path, bw_image_path):
        """
        Constructs an instance of the CropImages class.

        Parameters
        ----------
        optical_image_path : str
            The path to the optical image file.
        bw_image_path : str
            The path to the black and white image file.
        """
        
        if os.path.exists(optical_image_path):
            self.opt_image = ImageTrat(optical_image_path)
        else:
            raise FileNotFoundError(f"Optical image file not found: {optical_image_path}")

        if os.path.exists(bw_image_path):
            self.bw_image = ImageTrat(bw_image_path)
        else:
            raise FileNotFoundError(f"Black and white image file not found: {bw_image_path}")

    
    #pegando as coordenadas da área de interesse a ser recortada
    def pixel_definition(self):
        """
        Retrieves the coordinates of the area of interest to be cropped.

        Returns
        -------
        list
            A list of pixel coordinates representing the area of interest.
        """
        list_pixel=[]
        kernel = np.ones((5,5), dtype=np.uint8)
        
        image_bw = np.array(self.bw_image.image())
        image_opt = np.array(self.opt_image.image())
        
        comparison_result = np.any(image_bw != image_opt, axis=-1).astype(np.uint8)
        comparison_result_opening = cv2.morphologyEx(comparison_result, cv2.MORPH_OPEN, kernel)
        

        # Get the indices of different pixels
        rows, cols = np.where(comparison_result_opening)
        list_pixel = list(zip(rows, cols))

        return list_pixel

    #definindo a posição minima e máxima das coordenadas optidas para definir a área de corte
    def getMinMaxPositionByPixelList(self,list_pixel):
        """
        Determines the minimum and maximum positions from the obtained pixel coordinates.

        Parameters
        ----------
        list_pixel : list
            A list of pixel coordinates.

        Returns
        -------
        dict
            A dictionary containing the minimum and maximum positions along rows and columns.
        """
        list_pixel = np.array(list_pixel)

        iMin = np.min(list_pixel[:, 0])
        iMax = np.max(list_pixel[:, 0])
        jMin = np.min(list_pixel[:, 1])
        jMax = np.max(list_pixel[:, 1])

        return {'iMin': iMin, 'iMax': iMax, 'jMin':jMin, 'jMax': jMax}

    #gerando nova imagem com o recorte aplicado
    def run_crop_image(self, save_path, txt_path):
        """
        Applies cropping and processing to the optical image and saves the result.

        Parameters
        ----------
        save_path : str
            The path where the cropped and processed image will be saved.
        txt_path : str
            The path to the text file containing dimensions.

        Returns
        -------
        None
            The function doesn't return a value but saves the image.
        """
        try:
            '''take coordinates from crop region'''
            crop_position = self.getMinMaxPositionByPixelList(self.pixel_definition())
            get_iMin = crop_position['iMin']
            get_iMax = crop_position['iMax']
            get_jMin =crop_position['jMin']
            get_jMax = crop_position['jMax']
            
            '''cropp image'''
            crop_opt_image = self.opt_image.image()[get_iMin:get_iMax, get_jMin:get_jMax]
            
            
            '''resize image'''
            dim = self.opt_image.dimensions(txt_path)
            crop_opt_image = self.opt_image.resize(crop_opt_image, dim) 
            crop_opt_image  = self.opt_image.flip(crop_opt_image )

            '''save image'''
            self.opt_image.save_image(save_path, crop_opt_image)
        except Exception:
             print(traceback.format_exc()) 


class GenerateAFMOptico:
    """
    A class used for generating a new image from AFM information and optical images union .

    ...

    Attributes
    ----------
    self : GenerateAFMOptico
        An instance of the GenerateAFMOptico class.
    opt_image : ImageTrat
        An instance of the ImageTrat class for the optical image.
    df_afm : DataFrameTrat
        An instance of the DataFrameTrat class for AFM data.
    target : str
        The name of the target column in the DataFrame.
    process_date : str
        The name of the process date column in the DataFrame.
    flatten_height : str
        The name of the planned height column in the DataFrame.

    Methods
    -------
    add_new_channels(img, channels)
        Adds new channels to the given image.
    read_transparent_png(image_4channel)
        Converts a 4-channel image to a readable RGB image.
    run_generate_afm_optico_images(save_path)
        Generates AFM optical images and saves the results.
    """
    logger = setup_logger('image_treatment', level=logging.INFO)
    
    def __init__(self, img_path, df_path, target_col='stardist', process_date_col='Process Date', height_col='Planned Height'):
        """
        Constructs an instance of the GenerateAFMOptico class.

        Parameters
        ----------
        img_path : str
            The path to the optical image file.
        df_path : str
            The path to the CSV file containing AFM data.
        """
        self.logger.info(f"Initializing GenerateAFMOptico:")
        self.logger.info(f"  Image Path: {img_path}")
        self.logger.info(f"  DataFrame Path: {df_path}")
        self.logger.info(f"  Target Column: {target_col}")
        self.logger.info(f"  Process Date Column: {process_date_col}")
        self.logger.info(f"  Height Column: {height_col}")
        try:
            if not os.path.exists(img_path):
                 self.logger.error(f"GenerateAFMOptico init failed: Image path does not exist: {img_path}")
                 raise FileNotFoundError(f"Image path does not exist: {img_path}")
            if not os.path.exists(df_path):
                 self.logger.error(f"GenerateAFMOptico init failed: DataFrame path does not exist: {df_path}")
                 raise FileNotFoundError(f"DataFrame path does not exist: {df_path}")

            self.opt_image = ImageTrat(img_path)
            self.df_afm = DataFrameTrat(df_path)
            self.target = target_col
            self.process_date = process_date_col
            self.flatten_height = height_col
            self.logger.info("GenerateAFMOptico initialized successfully.")
        
        except FileNotFoundError as fnf_error:
             self.logger.error(f"Initialization failed due to missing file: {fnf_error}")
             raise
        except Exception as e:
            self.logger.error(f"Error during GenerateAFMOptico initialization: {e}")
            self.logger.error(traceback.format_exc())
            raise # Re-raise the exception
    
    
    def add_new_channels(self, img, channels):
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
        new_img = np.zeros((img.shape[0], img.shape[1], num_channels), dtype=np.float32)
        
        for i in range(num_channels):
            new_img[:,:,i] = channels[i]
        
        return new_img
    
    def apply_sobel_filter(self,image):
    
        # Aplicar o filtro de Sobel no eixo x
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        # Aplicar o filtro de Sobel no eixo y
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calcular a magnitude da borda
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Converter a magnitude para escala de 8 bits
        sobel_magnitude = np.uint8(sobel_magnitude / np.max(sobel_magnitude) * 255)
        
        
        return sobel_magnitude
    
    
    
    def read_transparent_png(self, image_4channel):
        """
        Converts a 4-channel image to a readable RGB image.

        Parameters
        ----------
        image_4channel : numpy.ndarray
            The 4-channel input image.

        Returns
        -------
        numpy.ndarray
            The converted RGB image.
        """
        alpha_channel = image_4channel[:,:,3]
        rgb_channels = image_4channel[:,:,:3]
        white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255
        alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
        alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)
        base = rgb_channels.astype(np.float32) * alpha_factor
        white = white_background_image.astype(np.float32) * (1 - alpha_factor)
        final_image = base + white
        return final_image.astype(np.uint8)
    
    
    def cos_height_sum_feature(self, channels, kernel_size=(10,10), only_afm=False):
            self.logger.debug(f"Applying cos_height_sum_feature. Kernel: {kernel_size}, Only AFM: {only_afm}")
            if not channels:
                self.logger.error("Cannot apply cos_height_sum_feature: input channel list is empty.")
                raise ValueError("Input channel list is empty.")
            try:
                
                height_channel = channels[0]
                cos_height_sum = (np.cos(height_channel) + height_channel)
                kernel = np.ones(kernel_size, np.uint8)
                
                cos_height_sum_01 = 255 * (cos_height_sum - cos_height_sum.min())/ (cos_height_sum.max() - cos_height_sum.min())
                
                ret, binary_threshold = cv2.threshold(cos_height_sum_01, 245,255, cv2.THRESH_BINARY)
                cos_heigh_erode = cv2.erode(-binary_threshold, kernel,iterations = 1)
                self.logger.debug(f"Generated cos_height_erode mask. Shape: {cos_heigh_erode.shape}")
            
                if only_afm:
                    self.logger.debug("Returning only the eroded cos_height feature.")
                    return cos_heigh_erode
                else:
                    channels.pop(0)
                    channels = [channels[i] * cos_heigh_erode for i in range(len(channels))]
                    self.logger.debug(f"Applied eroded mask to {len(channels)} remaining channels.")

                return channels
            except Exception as e:
                    self.logger.error(f"Error in cos_height_sum_feature: {e}")
                    self.logger.error(traceback.format_exc())
                    raise
        
    def select_feature_type(self, channels, pre_process):
        self.logger.info(f"Selecting feature type based on pre_process: {pre_process}")
        original_channel_count = len(channels)
        if pre_process == 'YOLO-AFM':
            self.logger.debug("Applying YOLO-AFM feature selection (cos_height_sum, kernel 15x15, not only AFM).")
            channels = self.cos_height_sum_feature(channels, kernel_size=(15, 15), only_afm=False)
        elif pre_process == 'AFM-Only':
            self.logger.debug("Applying AFM-Only feature selection (cos_height_sum, kernel 5x5, only AFM).")
            channels = self.cos_height_sum_feature(channels, kernel_size=(5, 5), only_afm=True)
        else: 
            channels.pop(0)
        return channels
         
    def create_image_based_on_feature(self, optical_image, channels):
        self.logger.debug(f"Creating image based on {len(channels)} selected features.")
        new_img = self.add_new_channels(optical_image, channels)
        
        if len(channels)>3:
            new_img = channels
        self.logger.debug(f"Created feature-based image with shape: {new_img.shape}")
        return new_img   

    def select_normalization(self, afm_info, feat, pre_process,  substrate=False):
        self.logger.debug(f"Selecting normalization for feature '{feat}'. Pre-process: {pre_process}, Substrate: {substrate}")
        if pre_process == 'yolo-afm':
            if feat == 'Planned Height':
                self.logger.debug(f"Applying Min-Max scaling to '{feat}'.")
                afm_info = self.df_afm.min_max_scale(afm_info, feat, substrate=substrate)
            else: 
                self.logger.debug(f"Applying Z-score scaling to '{feat}'.")
                afm_info = self.df_afm.zscore(afm_info, feat, substrate=substrate)
        
        elif pre_process == 'AFM-Only':
                self.logger.debug(f"Applying Min-Max scaling to '{feat}'.")
                afm_info = self.df_afm.min_max_scale(afm_info, feat, substrate=substrate)
        else: 
                self.logger.debug(f"Applying Z-score scaling to '{feat}'.")
                afm_info = self.df_afm.zscore(afm_info, feat, substrate=substrate)
    
        return afm_info
    
    
    def data_normalizer(self,afm_info, features, pre_process, features_that_no_need_remove_substrate = ['Planned Height', 'blue', 'hist_equalized']):
        self.logger.debug(f"Starting data normalization for features: {features[1:-2]}")
        for feat in features[1:-2]:
            self.logger.debug(f"Normalizing feature: {feat}")
            #Remove null or inf values
            try:
                mean_without_substrate = afm_info[feat].loc[afm_info['Segment'] != 'Substrate'].mean()
                self.logger.debug(f"  Mean without substrate for {feat}: {mean_without_substrate}")
                
                replacement_null_values_dict = {np.inf:mean_without_substrate,
                                                - np.inf:mean_without_substrate,
                                                np.nan:mean_without_substrate}
                
                afm_info[feat] = afm_info[feat].replace(replacement_null_values_dict)
                substrate = feat in features_that_no_need_remove_substrate
                afm_info = self.select_normalization(afm_info, feat, pre_process, substrate)
            except KeyError as ke:
                    self.logger.error(f"KeyError during normalization of '{feat}': {ke}. Check if 'Segment' column exists.")
                    raise
            except Exception as e:
                    self.logger.error(f"Error normalizing feature '{feat}': {e}")
                    self.logger.error(traceback.format_exc())
                    raise
        self.logger.debug("Data normalization completed.")
        return afm_info
                    
                    
    def save_matrix(self, save_path, img):
        """Saves the numpy matrix, logging the action."""
        self.logger.debug(f"Attempting to save matrix of shape {img.shape} to: {save_path}")
        try:
            # Ensure directory exists before saving
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                self.logger.warning(f"Save directory does not exist, creating: {save_dir}")
                os.makedirs(save_dir) # Use makedirs to create parent dirs if needed

            np.save(save_path, img)
            self.logger.info(f"Matrix saved successfully to: {save_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save matrix to {save_path}: {e}")
            self.logger.error(traceback.format_exc())
            return False          

    def run_generate_afm_optico_images(self, pre_process = ''):
        """
        Generates AFM optical images and saves the results.

        Parameters
        ----------
        save_path : str
            The path where the generated image will be saved.

        Returns
        -------
        None
            The function doesn't return a value but saves the image.
        """
        self.logger.info(f"Running generation for pre-process type: '{pre_process}'")
        try:

            self.logger.debug("Loading optical image...")
            optical_image = self.opt_image.image()
            dimensions = self.opt_image.dimensions()
            self.logger.debug(f"Optical image loaded. Dimensions: {dimensions}")
            self.logger.debug("Extracting blue channel and calculating equalized image...")
            blue, _,__ = self.opt_image.image_channels(optical_image)
            equalized_image = self.opt_image.equalize_img(optical_image)
            
            #Flatten blue and equalized_image
            blue_flatten = blue.flatten()
            equalized_image_flatten = equalized_image.flatten()
            self.logger.debug("Blue channel and equalized image processed.")
            
            self.logger.debug("Loading AFM DataFrame...")
            afm_info = self.df_afm.df.copy()
            self.logger.debug(f"AFM DataFrame loaded. Shape: {afm_info.shape}")
            afm_info['blue'] = blue_flatten 
            afm_info['hist_equalized'] = equalized_image_flatten
            self.logger.debug("Added 'blue' and 'hist_equalized' columns to DataFrame.")
            
            # Define features - Ensure 'Segment' and target exist
            required_cols = [self.process_date, self.flatten_height, 'Segment', self.target]
            missing_cols = [col for col in required_cols if col not in afm_info.columns]
            if missing_cols:
                 self.logger.error(f"Missing required columns in DataFrame: {missing_cols}")
                 raise ValueError(f"DataFrame missing required columns: {missing_cols}")
            features = [self.process_date, self.flatten_height,'blue','hist_equalized', 'Segment', self.target]
            self.logger.debug(f"Using features: {features}")
            afm_info = afm_info[features]
            # afm_info = self.df_afm.clean_target(afm_info)
            
            features_that_no_need_remove_substrate = ['Planned Height', 'blue', 'hist_equalized']
            afm_info = self.data_normalizer(afm_info, features, pre_process, features_that_no_need_remove_substrate)
            self.logger.debug("Creating channels from DataFrame features...")
            channels = []
            for feat in features[1:-2]: # Iterate through features used for channels
                try:
                    channel = self.df_afm.create_channel_by_df(afm_info, feat, dimensions)
                    channels.append(channel)
                    self.logger.debug(f"  Created channel for '{feat}'. Shape: {channel.shape}")
                except Exception as e:
                    self.logger.error(f"Error creating channel for feature '{feat}': {e}")
                    raise
            self.logger.debug(f"Total channels created before selection: {len(channels)}")

            # Apply threshold/clipping
            self.logger.debug("Applying clipping to relevant channels...")
            
            # Apply threshold to certain channels
            for i, feat in enumerate(features[1:-2]):
                if feat not in features_that_no_need_remove_substrate:
                    channels[i] = np.clip(channels[i], -1, 1)
            
            selected_channels = self.select_feature_type(channels, pre_process)
            self.logger.debug("Creating final image based on selected features...")
            new_img = self.create_image_based_on_feature(optical_image, selected_channels)
            mask = self.df_afm.create_channel_by_df(afm_info,  self.target, dimensions).astype(np.uint8)
                        # Ensure mask is binary 0 or 1
            mask[mask > 0] = 1
            self.logger.info("Image and mask generation completed successfully.")
            self.logger.debug(f"  Final image shape: {new_img.shape}, dtype: {new_img.dtype}")
            self.logger.debug(f"  Final mask shape: {mask.shape}, dtype: {mask.dtype}, unique values: {np.unique(mask)}")

            return new_img, mask
        except FileNotFoundError as fnf_error:
            self.logger.error(f"File not found during image/mask generation: {fnf_error}")
            raise # Re-raise specific error
        except ValueError as val_error:
            self.logger.error(f"Value error during image/mask generation: {val_error}")
            raise # Re-raise specific error
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during image/mask generation: {e}")
            self.logger.error(traceback.format_exc())
            raise # Re-raise general error