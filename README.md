# UNet_AFM
 In this repository, the project for segmenting cellular structures through atomic force microscopy (AFM) and its respective models will be stored.
 
![Slide1](https://github.com/ArtRocha/Unet_AFM/assets/61946276/4d4579d3-6c4a-4008-af93-a22f9077d976)

> The project is a U-Net aimed at segmenting cell nuclei using features derived from Atomic Force Microscopy (AFM).
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
                                <td>h5py</td>
                                <td>3.11.0</td>
                                <td>pip install h5py==3.11.0</td>
                              </tr>
                               <tr>
                                <td>kaleido</td>
                                <td>0.2.1</td>
                                <td>pip install kaleido==0.2.1</td>
                              </tr>
                               <tr>
                                <td>keras</td>
                                <td>2.14.0</td>
                                <td>pip install keras==2.14.0</td>
                              </tr>
                               <tr>
                                <td>Keras-Preprocessing</td>
                                <td>1.1.2</td>
                                <td>pip install Keras-Preprocessing==1.1.2</td>
                              </tr>
                                 <tr>
                                <td>matplotlib</td>
                                <td>3.7.0</td>
                                <td>pip install matplotlib==3.7.0</td>
                              </tr>
                                 <tr>
                                <td>numpy</td>
                                <td>1.26.4</td>
                                <td>pip install numpy==1.26.4</td>
                              </tr>
                                 <tr>
                                <td>opencv_python</td>
                                <td>4.5.5.64</td>
                                <td>pip install opencv_python==4.5.5.64</td>
                              </tr>
                                 <tr>
                                <td>opencv-python-headless</td>
                                <td>4.5.4.60</td>
                                <td>pip install opencv-python-headless==4.5.4.60</td>
                              </tr>
                              <tr>
                                <td>opt-einsum</td>
                                <td>3.3.0</td>
                                <td>pip install opt-einsum==3.3.0</td>
                              </tr>
                              <tr>
                                <td>pandas</td>
                                <td>1.5.3</td>
                                <td>pip install pandas==1.5.3</td>
                              </tr>
                              <tr>
                                <td>plotly</td>
                                <td>10.1.0</td>
                                <td>pip install plotly==10.1.0</td>
                              </tr>
                           <tr>
                                <td>plotly</td>
                                <td>5.9.0</td>
                                <td>pip install plotly==5.9.0</td>
                              </tr>
                               <tr>
                                <td>scikit_image</td>
                                <td>0.19.3</td>
                                <td>pip install scikit-image==0.19.3</td>
                              </tr>
                               <tr>
                                <td>scikit-learn</td>
                                <td>1.2.1</td>
                                <td>pip install scikit-learn==1.2.1</td>
                              </tr>
                               <tr>
                                <td>scipy</td>
                                <td>1.10.0</td>
                                <td>pip install scipy==1.10.0</td>
                              </tr>
                               <tr>
                                <td>setuptools</td>
                                <td>69.5.1</td>
                                <td>pip install setuptools==69.5.1</td>
                              </tr>
                               <tr>
                                <td>six</td>
                                <td>1.16.0</td>
                                <td>pip install six==1.16.0</td>
                              </tr>
                              <tr>
                                <td>Tensorflow</td>
                                <td>2.14.0</td>
                                <td>pip install tensorflow==2.14.0</td>
                              </tr>
                              <tr>
                                <td>Torch</td>
                                <td>2.3.0</td>
                                <td>pip install torch==2.3.0</td>
                              </tr>
                              <tr>
                                <td>tqdm</td>
                                <td>4.64.1</td>
                                <td>pip install tqdm==4.64.1</td>
                              </tr>
                              <tr>
                                <td>xgboost</td>
                                <td>1.7.3</td>
                                <td>pip install xgboost==1.7.3</td>
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
   sudo docker exec -it YOLO-AFM python /app/dev/apps/main.py
   ```

### :file_folder: DATA FOLDER ARCHITECTURE
<details>

 <summary>See data folder architecture</summary>
 
```
├── datasets
├── input
│   ├── optical_images_resized
│   ├── train
│   │   ├── train_1_channels_only_AFM_CosHeightSum
│   │   │   ├── msk_img_training
│   │   │   └── opt_img_training
│   │   ├── train_2_channels_like_yolo_opt_afm
│   │   │   ├── msk_img_training
│   │   │   └── opt_img_training
│   │   └── train_2_channels_only_optical
│   │       ├── msk_img_training
│   │       └── opt_img_training
│   └── Usefull_data
├── intermediate
│   ├── pre_processing_afm
│   │   ├── image
│   │   └── mask
│   ├── pre_processing_optico
│   │   ├── image
│   │   └── mask
│   └── pre_processing_optico_and_afm
│       ├── image
│       └── mask
├── output
│   ├── unet_afm_1_channels_only_AFM_CosHeightSum
│   │   └── predicts
│   ├── unet_afm_2_channels_like_yolo_opt_afm
│   │   └── predicts
│   └── unet_afm_2_channels_only_optical
│       └── predicts
└── raw
    ├── bw_images
    ├── optical_images
    └── txt_files
```

 
</details>


## :arrow_forward: Project scripts


:robot: main.py
> This script runs the entire project according to the option selected by the user:
> * Option 1: Runs the AFM-Only model process and returns its segmentations, general and specific metrics.
> * Option 2: Runs the YOLO-AFM model process and returns its segmentations, general and specific metrics.
> * Option 3: Runs the Optical-Only model process and returns its segmentations, general and specific metrics.

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

:open_file_folder: 3_prdicts.py 
> This script is responsible to take pre-process images selected by user and make yours respective predictions.

**INPUT:**
> * Optical Images Crop `data/input/optical_images_resized`
> * AFM data file `data/input/Usefull_data`
> * selected preprocess image `data/intermediate/pre_processing_<model-selected>`
> * selected preprocess mask `data/intermediate/pre_processing_<model-selected>` --- optional

**OUTPUT**
> * Segmented image `data/output/<model-selected>/predicts`

> ![Ap038_Segmentação enxugado - Copia pptx](https://github.com/ArtRocha/Unet_AFM/assets/61946276/e55d929e-4ffc-4adb-8131-4427e19a814c)

:dart: 4_ecal_models.py
> This script search for test files and compare with model segmentation to show yours performance in `data/output/<model-selected>/model_metrics.png`

**INPUT:**
> * selected preprocess mask `data/intermediate/pre_processing_<model-selected>`
> * Segmented image `data/output/<model-selected>/predicts`

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
