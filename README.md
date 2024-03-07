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
                                <td>pip install matplotlib==3.7.2</td>
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
                                <td>1.3.0</td>
                                <td>pip install scikit-learn==1.3.0</td>
                              </tr>
                               <tr>
                                <td>scipy</td>
                                <td>1.11.1</td>
                                <td>pip install scipy==1.11.1</td>
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
                                <td>pip install xgboost==2.0.3</td>
                              </tr>
                            </tbody>
                          </table>
                        </body>
                        </html>


### :file_folder: DATA

The data is available at the link ```<link>```

* Download the file `data.zip` and the folder `models`;
* Copy the folders to the project's root directory.
* Now you are ready to run the project.
  
### 💻 Run project
> The scripts have been named numerically to follow the order in which they should be executed.

:scissors: 1_cropping_opt_images.py

> This script performs cropping and resizing of optical images.
