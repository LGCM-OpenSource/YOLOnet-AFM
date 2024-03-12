# Unet_AFM
 In this repository, the project for segmenting cellular structures through atomic force microscopy (AFM) and its respective models will be stored.
## 🚀 Start


These instructions will allow you to obtain a copy of the project up and running on your local machine for development and testing purposes.

`git clone <GIT_URL>”`

### 📋 Prerequisites

# 💻 Requirements
Before you begin, make sure you've met the following requirements:
<!---Estes são apenas requisitos de exemplo. Adicionar, duplicar ou remover conforme necessário--->
* You have installed version  `python >= 3.9.12 `


### 🔧 Install

```
pip install -r requirements.txt

```
if fail try: 
```
pip install --force-reinstall -r requirements.txt
```

> If any dependency is not installed, the list below has the commands to install the uninstalled libraries manually
                 <!DOCTYPE html>
                        <html>
                        <head>
                        </head>
                        <body>
                          <table>
                            <thead>
                              <tr>
                                <th>Import</th>
                                <th>Version</th>
                                <th>Installation command</th>
                              </tr>
                            </thead>
                            <tbody>
                              </tr>
                                 <tr>
                                <td>matplotlib</td>
                                <td>3.7.2</td>
                                <td>pip install matplotlib==3.5.2</td>
                              </tr>
                                 <tr>
                                <td>numpy</td>
                                <td>1.24.3</td>
                                <td>pip install numpy==1.24.3</td>
                              </tr>
                                 <tr>
                                <td>opencv_python</td>
                                <td>4.9.0.80</td>
                                <td>pip install opencv_python==4.9.0.80</td>
                              </tr>
                              <tr>
                                <td>pandas</td>
                                <td>2.0.3</td>
                                <td>pip install pandas==2.0.3</td>
                              </tr>
                              <tr>
                                <td>plotly</td>
                                <td>5.9.0</td>
                                <td>pip install plotly==5.9.0</td>
                              </tr>
                               <tr>
                                <td>scikit_image</td>
                                <td></td>
                                <td>pip install scikit-image</td>
                              </tr>
                               <tr>
                                <td>scikit-learn</td>
                                <td>1.4.0</td>
                                <td>pip install scikit-learn==1.4.0</td>
                              </tr>
                               <tr>
                                <td>scipy</td>
                                <td>1.7.3</td>
                                <td>pip install scipy==1.7.3</td>
                              </tr>
                              <tr>
                                <td>Tensorflow</td>
                                <td>2.15.0</td>
                                <td>pip install tensorflow==2.15.0</td>
                              </tr>
                              <tr>
                                <td>Torch</td>
                                <td>2.2.1</td>
                                <td>pip install torch==2.2.1</td>
                              </tr>
                              <tr>
                                <td>tqdm</td>
                                <td>4.65.0</td>
                                <td>pip install tqdm==4.65.0</td>
                              </tr>
                              <tr>
                                <td>xgboost</td>
                                <td>2.0.3</td>
                                <td>pip install xgboost==1.6.2</td>
                              </tr>
                            </tbody>
                          </table>
                        </body>
                        </html>


### :file_folder: DATA

The data is available at the link ```https://drive.google.com/drive/folders/15N1tuNQ12LPO_nU7bUBwtT78i2IcD40k?usp=drive_link```

* Download the file `data.zip` and the folder `models`;
* Copy the folders to the project's root directory.
* Now you are ready to run the project.
  
### 💻 Run project
> The scripts have been named numerically to follow the order in which they should be executed.

:robot: 0_run_scripts.py
> This script is responsible for running all processes, allowing the user to select the model they wish to test and obtain their respective metrics.

:scissors: 1_cropping_opt_images.py

> This script is responsible for matching the pixels of the optical image with the AFM spreadsheet, applying the cropping of the region of interest and resizing the optical images.

**INPUT:**
> * Optical Images `data/raw/optical_images`
> * BW Images `data/raw/bw_images`
> * AFM Matrix `data/raw/txt_files`

**OUTPUT:**
> * Optical Images Crop `data/input/optical_images_resized`

> ![Slide1](https://github.com/ArtRocha/Unet_AFM/assets/61946276/11099a1d-a287-41ca-bbae-23626afe96bd)

:framed_picture: 2_preprocess_unet.py
> This script is responsible for processing the data that will be used in the UNet_AFM, meaning the integration of the optical image with the AFM information.

**INPUT:**
> * Optical Images Crop `data/input/optical_images_resized`
> * AFM data file `data/input/Usefull_data`

**OUTPUT:**
> * AFM optical image `data/intermediate/pre_processing_optico_and_afm/image`
> * Mask `data/intermediate/pre_processing_optico_and_afm/mask`

>  ![Slide1](https://github.com/ArtRocha/Unet_AFM/assets/61946276/4d4579d3-6c4a-4008-af93-a22f9077d976)

:open_file_folder: 3_preprocess_pixel.py 
> This script is responsible for creating the features 'Norm Height' and 'Height Position', which are derived from the feature 'Flatten Height'. Additionally, it normalizes the features 'MaxPosition_F0500pN' and 'YM_Fmax0500pN' using the StandardScaler method.

**INPUT:**
> * AFM data file `data/input/Usefull_data`

**OUTPUT**
> * Normalized AFM data file `data/intermediate/pre_processing_only_afm`

> ![Ap038_Segmentação enxugado - Copia pptx](https://github.com/ArtRocha/Unet_AFM/assets/61946276/e55d929e-4ffc-4adb-8131-4427e19a814c)

:dart: 4_vUnet_AFM_predict.py
> This script performs segmentations using the vUnet_AFM model, where pixel-wise segmentation is triggered in cases where the Unet_AFM does not provide adequate segmentation and sends the results to the folder `data/output/vunet_AFM_predictions/`.

**INPUT:**
> * Optical Images Crop `data/input/optical_images_resized`
> * AFM data file `data/input/Usefull_data`
> * AFM optical image `data/intermediate/pre_processing_optico_and_afm/image`
> * Mask `data/intermediate/pre_processing_optico_and_afm/mask`

**OUTPUT**
> * Segmented image `data/output/vunet_AFM_predictions/predicts`
> * AFM data file segmented `data/output/vunet_AFM_predictions/predict_sheets`

:dart: 5_Unet_AFM_predict.py
> This script performs segmentations using the Unet_AFM model and sends the results to the folder `data/output/unet_AFM_predictions/`

**INPUT:**
> * Optical Images Crop `data/input/optical_images_resized`
> * AFM data file `data/input/Usefull_data`
> * AFM optical image `data/intermediate/pre_processing_optico_and_afm/image`
> * Mask `data/intermediate/pre_processing_optico_and_afm/mask`

**OUTPUT**
> * Segmented image `data/output/unet_AFM_predictions/predicts`
> * AFM data file segmented `data/output/unet_AFM_predictions/predict_sheets`

:dart: 6_pixel_predict.py
> This script performs segmentations using the pixel-wise segmentation model and sends the results to the folder `data/output/only_AFM_predictions/`.

**INPUT:**
> * Normalized AFM data file `data/intermediate/pre_processing_only_afm`
> * AFM data file `data/input/Usefull_data`
> * Optical Images Crop `data/input/optical_images_resized`

**OUTPUT**
> * Segmented image `data/output/only_afm_predictions/predicts`
> * AFM data file segmented `data/output/only_afm_predictions/predict_sheets`


:bar_chart: 7_eval_model.py
> This script retrieves information specified from user:

**INPUT:**
> * AFM data file segmented`data/ouput/<SELECTED_MODEL>/predict_sheets`

**OUTPUT:**
> * Model metrics `data/ouput/<SELECTED_MODEL>/`

:bar_chart: 8_eval_model_per_cell.py
> In this script, the user selects the path based on the model to be evaluated, and the script returns the segmented cell and its respective metrics.

**INPUT:**
> * Segmented image `data/output/<SELECTED_MODEL>/predicts`
> * AFM data file segmented`data/ouput/<SELECTED_MODEL>/predict_sheets`

**OUTPUT:**
> * Metrics per cell `data/output/<SELECTED_MODEL>/metrics_per_cell`


