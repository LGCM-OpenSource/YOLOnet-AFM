# UNet_AFM
 In this repository, the project for segmenting cellular structures through atomic force microscopy (AFM) and its respective models will be stored.

## 📋 Hardware Prerequisites
Before you begin, make sure you've met the following prerequisites:
* RAM: At least 8 GB recommended to ensure adequate performance during the model segmentation and evaluation process.
* PROCESSOR: A 64-bit multicore processor capable of executing SSE2 instructions or higher is recommended. An Intel® Core™ i5-10400 processor or equivalent is sufficient for most tasks.
* GRAPHICS: A dedicated graphics card with CUDA support is recommended if using deep learning techniques that make use of GPU acceleration. However, the code provided in this repository should run smoothly on an integrated graphics card such as Intel® UHD Graphics 630.

## 💻 Requirements
* `python >= 3.9.19 `
* `Docker`

<details>
<summary>
 Click here for the Docker installation guide
</summary>

 ### For installing Docker on Windows, please follow these steps and documentation:

1. **Install WSL (Windows Subsystem for Linux)**
   - [WSL Installation Guide](https://docs.microsoft.com/windows/wsl/install)
   - WSL will ensure an environment where Docker will function properly.

2. **Install Docker Desktop for Windows**
   - [Docker Desktop Download](https://www.docker.com/products/docker-desktop)
   - After installation, create a free account on Docker Hub and make sure you are logged into Docker Desktop with it.
   - In some cases, the Docker Desktop installation process will include registration and login.

### For installing Docker on Linux:
1. **Docker Ubuntu**
   - [Docker Ubuntu Download](https://docs.docker.com/engine/install/ubuntu/)
   - If you are using a different Linux distribution, you can consult the menu on the left side of the page for other possible distributions for Docker installation.

</details>

> [!TIP]
> After installation, remember to run the command `sudo usermod -aG docker $USER` and restart your session to use Docker without needing to use `sudo`.

### :books: Libraries

<details>

<summary> Click Here to see project libraries </summary>
> Here is the table with the libraries used and their respective versions.
                 <!DOCTYPE html>
                        <html>
                        <head>
                        </head>
                        <body>
                          <table>
                            <thead>
                              <tr>
                                <th>Lib</th>
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
                                <td>kaleido</td>
                                <td>0.2.1</td>
                                <td>pip install ==0.2.1</td>
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
                                <td>1.6.2</td>
                                <td>pip install xgboost==1.6.2</td>
                              </tr>
                            </tbody>
                          </table>
                        </body>
                        </html>


</details>

> [!NOTE]
> All projects libraries will be installed on docker build comand.


## 🚀 Start
These instructions will allow you to obtain a copy of the project up and running on your local machine for development and testing purposes.

1. Clone project
   ```
   git clone <GIT_URL>
   ```
3. Access the [Drive link](https://drive.google.com/drive/folders/15N1tuNQ12LPO_nU7bUBwtT78i2IcD40k?usp=drive_link)
   * Download the file `data.zip` and the folder `models`;
   * Extract `data.zip` folder to root directory;
   * Set the `models` folders to the project's root directory.
4. To build docker container and install all dependences run:
   ```
   sudo docker compose up -d --build
   ```
5. Run project:
   ```
   sudo docker exec -it unet_afm_container python /app/dev/apps/main.py
   ```

### :file_folder: FOLDER ARCHITECTURE
<details>

 <summary>See folder architecture</summary>
 
```
├── data
│   │
├───input
│    ├───optical_images_resized
│    ├───original_images
│    ├───Usefull_data
│    
├───intermediate
│   ├─── pre_processing_optico_and_afm
│       ├───image
│       └───mask
├───output
│   ├───only_afm_predictions
│   │   ├───metrics_per_cell
│   │   ├───predicts        
│   │   └───predict_sheets  
│   ├───unet_AFM_predictions
│   │   ├───metrics_per_cell
│   │   ├───predicts
│   │   └───predict_sheets
│   └───vunet_AFM_predictions
│       ├───metrics_per_cell
│       ├───predicts
│       └───predict_sheets
└───raw
    ├───bw_images
    ├───optical_images
    └───txt_files
├── dev
│   │
└───apps
    │   1_cropping_opt_images.py
    │   2_preprocess_unet.py
    │   3_preprocess_pixel.py
    │   4_vUnet_AFM_predictions.py
    │   5_unet_AFM_predict.py
    │   6_pixel_predict.py
    │   7_eval_model.py
    │   8_eval_model_per_cell.py
    │   main.py
    │   metrics.py
    │   __init__.py
    │
    ├───utils
    │   │   colors_to_pcr.py
    │   │   dataframe_treatment.py
    │   │   image_treatment.py
    │   │   models.py
    │   │   unet_model.py
    │   │   __init__.py
├── models
│   
├── README.md
├── requirements_linux.txt
├── requirements_win.txt
└── setup.py
```

 
</details>


## :arrow_forward: Project scripts


:robot: main.py
> This script runs the entire project according to the option selected by the user:
> * Option 0: Runs all models and returns their respective general and specific metrics.
> * Option 1: Runs the vUnet_AFM model and returns its general and specific metrics.
> * Option 2: Runs the Unet_AFM model and returns its general and specific metrics.
> * Option 3: Runs the AFM_only model (per pixel) and returns its general and specific metrics.

### 💻 Scripts Details

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


:heavy_check_mark: Solve NUMA (Non-Uniformed Memory Access)
> (https://gist.github.com/zrruziev/b93e1292bf2ee39284f834ec7397ee9f)

## 🛠️ Build with

> * [Visual Studio Code](https://code.visualstudio.com/) - Code Editor
> * [Notion](https://notion.so/) - Task Manager 
> * [Git Lab](http://172.22.133.244:8081/) - Code repo 
> * [Python](https://www.python.org/) -  Python superset
> * [Pandas](https://www.python.org/) -  Python lib
> * [Scikit-Learn](https://scikit-learn.org/stable/#) -  Python lib
> * [Scikit-Image](https://scikit-image.org/) -  Python lib
> * [Opencv](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) -  Python lib

## ✒️ Authors


<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/ArtRocha">
        <img src="https://github.com/ArtRocha/Unet_AFM/assets/61946276/486735c6-716a-445f-a7f9-8a2d849e3d47" width="150px;" alt="Foto Arthur Rocha" /><br>
        <sub>
          <b style="font-size:20px"> Arthur Rocha </b>
        </sub>
      </a>
    </td>
    <td align="center" >
         <a href="http://lattes.cnpq.br/8207473893996045">
           <img src="https://github.com/ArtRocha/Unet_AFM/assets/61946276/b114b01f-8df3-4acf-bc1a-8981331ff67c" width="150px;" alt="Foto de Ayumi Aurea"/><br>
           <sub  >
             <b style="font-size:20px"> Ayumi Aurea Miyakawa </b><br>
           </sub>
         </a>
       </td>
        <td align="center" >
      <a href="http://lattes.cnpq.br/0399495551887391">
        <img src="http://servicosweb.cnpq.br/wspessoa/servletrecuperafoto?tipo=1&id=K8153430T2" width="150px;" alt="Foto de Cleyton Biffe"/><br>
        <sub  >
          <b style="font-size:20px"> Cleyton Biffe </b><br>
        </sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/EdCarlos-dev">
        <img src="https://github.com/ArtRocha/Unet_AFM/assets/61946276/558ae40e-626f-4305-b550-cde25f2f8172" width="150px;" alt="Foto de Ed Santos e Silva"/><br>
        <sub>
          <b style="font-size:20px"> Ed Santos e Silva </b><br>
        </sub>
      </a>
    </td>
     <td align="center" >
      <a href="http://lattes.cnpq.br/9674023945962136">
        <img src="https://lh3.googleusercontent.com/a-/ALV-UjUC_CbCkHR3n6mjft683RBUYRlmXo9xZdX01RdaTErJyQ=s272-p-k-rw-no" width="150px;" alt="Foto de Jose Patane"/><br>
        <sub  >
          <b style="font-size:20px"> Jose Patane </b><br>
        </sub>
      </a>
    </td>
  </tr>
</table>


<div align="center"> 
 <img  src="https://genetica.incor.usp.br/wp-content/uploads/2021/05/cropped-temp.png" width="400px;"/>  

© Todos os direitos reservados, 2024 
</div>
