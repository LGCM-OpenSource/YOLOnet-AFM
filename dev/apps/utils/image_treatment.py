import numpy as np
import cv2
import traceback
import os
from .dataframe_treatment import DataFrameTrat
import matplotlib.pyplot as plt
'''
CLASS               LINE

ImageTrat             18
CropImages           223
GenerateAFMOptico    350

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
            x_dim, y_dim, chan = self.image(matrix=matrix).shape
            return x_dim, y_dim
    
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
        
        image_bw = np.array(self.bw_image.image)
        image_opt = np.array(self.opt_image.image)

        # Compare all color channels using NumPy
        comparison_result = np.any(image_bw != image_opt, axis=-1)

        # Get the indices of different pixels
        rows, cols = np.where(comparison_result)
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
            crop_opt_image = self.opt_image.image[get_iMin:get_iMax, get_jMin:get_jMax]
            
            
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
    
    def __init__(self, img_path, df_path):
        """
        Constructs an instance of the GenerateAFMOptico class.

        Parameters
        ----------
        img_path : str
            The path to the optical image file.
        df_path : str
            The path to the CSV file containing AFM data.
        """
        self.opt_image = ImageTrat(img_path)
        self.df_afm = DataFrameTrat(df_path)
        self.target = 'Generic Segmentation'
        self.process_date = 'Process Date'
        self.flatten_height = 'Planned Height'  
    
    
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
        new_img = np.zeros((img.shape[0], img.shape[1], num_channels))
        
        for i in range(num_channels):
            new_img[:,:,i] = channels[i]
        
        return new_img.astype(np.float32)
    
    
    def treat_planned_height_by_viridis_map(self, planned_image, dimensions):
            plt.imshow(planned_image)
            plt.axis('off')  # Desativar os eixos para uma visualização mais limpa
            plt.savefig('planned_virids.png', bbox_inches='tight', pad_inches=0)
            planned_viridis = cv2.imread('planned_virids.png')
            planned_viridis = cv2.resize(planned_viridis, (dimensions[1], dimensions[0]), cv2.INTER_AREA)
            # test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
            
            colors_list = []
            for chan in self.opt_image.image_channels(planned_viridis):
                color_normalized = (chan - np.mean(chan)) / np.std(chan)
                colors_list.append(color_normalized)
                
            # Return list in sequence, blue, green, red
            return colors_list

    def show_rgb_virids(self, blue, green, red, new_img, optical_image):

            plt.subplot(1,5,1)
            plt.imshow(blue, cmap='gray')
            plt.title("Planned BLUE")
            plt.axis('off')
            
            plt.subplot(1,5,2)
            plt.imshow(green, cmap='gray')
            plt.title("Planned GREEN")
            plt.axis('off')
            
            plt.subplot(1,5,3)
            plt.imshow(red, cmap='gray')
            plt.title("Planned RED")
            plt.axis('off')
            
            plt.subplot(1,5,4)
            plt.imshow(new_img[:,:,0], cmap='gray')
            plt.title("Planned Height")
            plt.axis('off')
            
            plt.subplot(1,5,5)
            plt.imshow(optical_image)
            plt.title("Optical Image")
            plt.axis('off')
            plt.show()
    
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
    
    def run_generate_afm_optico_images(self, save_path):
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
        try:
            optical_image = self.opt_image.image()
            dimensions = self.opt_image.dimensions()
            blue, _,__ = self.opt_image.image_channels(optical_image)
            equalized_image = self.opt_image.equalize_img(optical_image)
            
            #Flatten blue and equalized_image
            blue_flatten = blue.flatten()
            equalized_image_flatten = equalized_image.flatten()
            
            afm_info = self.df_afm.df
            afm_info['blue'] = blue_flatten 
            afm_info['hist_equalized'] = equalized_image_flatten
            
            
            '''2 channels (only optico)'''
            features = [self.process_date, 'blue','hist_equalized','Segment', self.target]
            
            '''4 channels (only AFM)'''
            # features = [self.process_date, self.flatten_height,'YM_Fmax1500pN', 'YM_Fmax0300pN', 'MaxPosition_F1500pN','Segment', self.target]
            
            '''6_channels (optico + AFM)'''
            # features = [self.process_date, self.flatten_height, 'blue', 'hist_equalized', 'YM_Fmax1500pN', 'YM_Fmax0300pN', 'MaxPosition_F1500pN','Segment', self.target]
            
            afm_info = afm_info[features]
            afm_info = self.df_afm.clean_target(afm_info)
            features_that_no_need_remove_substrate = ['Planned Height', 'blue', 'hist_equalized']
            for feat in features[1:-2]:
                substrate = False
                #Remove null or inf values
                mean_without_substrate = afm_info[feat].loc[afm_info['Segment'] != 'Substrate'].mean()
                afm_info[feat].replace([np.inf, - np.inf, np.nan], mean_without_substrate, inplace=True)
                
                if feat in features_that_no_need_remove_substrate:
                    #Calc Zscore by mean and std with substrate
                    substrate = True
                    
                #Calc Zscore by mean and std without substrate
                afm_info = self.df_afm.zscore(afm_info, feat, substrate=substrate)
                    
            
            channels = []
            for feat in features[1:-2]:
                feature_image = self.df_afm.create_channel_by_df(afm_info,  feat, dimensions) 
                if feat == 'Planned Height':
                    feature_image = - feature_image
                if feat not in features_that_no_need_remove_substrate:
                    #apply threshold
                    feature_image[feature_image > 3] = 3
                    feature_image[feature_image < -3] = -3
                channels.append(feature_image)
            
            new_img = self.add_new_channels(optical_image, channels)
            new_img = new_img.astype(np.float32)
            

            # new_img = self.read_transparent_png(new_img)

            #Create mask
            mask = self.df_afm.create_channel_by_df(afm_info,  self.target, dimensions)
            mask = mask.astype(np.uint8)
            
            
            np.save(save_path, new_img)
            mask_save_path = save_path.replace(f'pre_processing_only_optico{os.sep}image', f'pre_processing_only_optico{os.sep}mask')
            np.save(mask_save_path, mask)
        except Exception:
            print(traceback.format_exc())