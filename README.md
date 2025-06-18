# UNet_AFM: Cellular Structure Segmentation using U-Net and Atomic Force Microscopy

This repository contains the code and models for a project focused on segmenting cellular structures, specifically nuclei, using data derived from Atomic Force Microscopy (AFM), optionally combined with optical microscopy images. The core segmentation model is based on the U-Net architecture.

![Project Workflow Diagram](papers_fig/mainText/fig2_models_preprocessing.jpg)

*Figure: Overview of the data preprocessing and model application workflow.*

## ✨ Features

*   **AFM Data Processing:** Scripts to process raw AFM data (`.csv` files).
*   **Optical Image Integration:** Tools to crop, resize, and align optical images with AFM data.
*   **Multiple Data Modes:** Supports segmentation using:
    *   AFM data only (1 channel: CosHeightSum)
    *   Optical data only (2 channels)
    *   Combined AFM and Optical data (2 channels)
*   **U-Net Model Implementation:** Utilizes TensorFlow/Keras for the U-Net model.
*   **Prediction Pipeline:** Generates segmentation masks for input images using trained models.
*   **Performance Evaluation:** Calculates and visualizes segmentation metrics (e.g., Precision, Recall and Dice coefficient).
*   **Dockerized Environment:** Ensures reproducibility and simplifies dependency management.

## 📋 Table of Contents

*   [Hardware Prerequisites](#️-hardware-prerequisites)
*   [Software Requirements](#-software-requirements)
*   [Installation & Setup](#-installation--setup)
*   [Usage](#️-usage)
*   [Project Structure](#-data-folder-structure)
*   [Workflow Pipeline](#-integrated-workflow-orchestrated-by-mainpy)
*   [Script Details](#-script-details)
*   [Libraries Used](#-libraries-used)
*   [Built With](#️-built-with)
*   [Contributing](#-contributing)
*   [Authors](#️-authors)
*   [Acknowledgments](#-acknowledgments)

## ⚙️ Hardware Prerequisites

Ensure your system meets the following minimum recommendations for optimal performance:

*   **RAM:** 8 GB+ (More recommended for training with larger datasets/models)
*   **Processor:** 64-bit multicore CPU (e.g., Intel Core i5-10400 or equivalent/better)
*   **Graphics:**
    *   *Recommended:* NVIDIA GPU with CUDA support for accelerated training/inference.
    *   *Minimum:* Integrated graphics (e.g., Intel UHD Graphics 630) should suffice for running predictions, though potentially slower.

## 💻 Software Requirements

*   **Python:** Version 3.9.19 or higher.
*   **Docker:** Required for containerized setup.

<details>
<summary>Docker Installation Guide</summary>

### 🪟 Windows

1. Install WSL:

   * [Install WSL Guide](https://docs.microsoft.com/windows/wsl/install)
2. Install Docker Desktop:

   * [Download Docker Desktop](https://www.docker.com/products/docker-desktop)
   * Create a Docker Hub account and log in.

### 🐧 Linux

* Install Docker for your distribution:

  * [Install Docker on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)

> \[!TIP] To use Docker without `sudo`, run: `sudo usermod -aG docker $USER` and restart your session.

</details>

## 🚀 Installation & Setup

Follow these steps to get the project running locally using Docker:

1. **Clone the Repository:**

   ```bash
   git clone <YOUR_GIT_REPOSITORY_URL>
   ```

2. **Download Data and Models:**

   * Download `data.zip` and the `models` folder from the provided source.
   * Extract `data.zip` into the project root. This will create the `data/` folder.
   * Place the `models` folder in the project root.

3. **Build and Start the Docker Container:**

   ```bash
   make build
   make up
   ```
## 📚 Python Libraries

<details>
<summary>Click Here to see project libraries</summary>

> The following Python libraries are used in this project. They are automatically installed when building the Docker container via `requirements.txt`.

| Library                  | Version   | Purpose                                     |
| :----------------------- | :-------- | :------------------------------------------ |
| h5py                     | 3.11.0    | Interacting with HDF5 files (often for models) |
| kaleido                  | 0.2.1     | Saving Plotly figures as static images      |
| keras                    | 2.14.0    | High-level API for TensorFlow               |
| Keras-Preprocessing      | 1.1.2     | Data preprocessing utilities for Keras      |
| matplotlib               | 3.7.0     | Plotting and visualization                  |
| numpy                    | 1.26.4    | Fundamental package for numerical computing |
| opencv_python            | 4.5.5.64  | Computer Vision library (image processing)  |
| opencv-python-headless | 4.5.4.60  | Headless OpenCV (for servers/containers)    |
| opt-einsum               | 3.3.0     | Optimized tensor contractions (dependency)  |
| pandas                   | 1.5.3     | Data manipulation and analysis (CSV, etc.)  |
| Pillow                   | 10.1.0    | Python Imaging Library (fork)               |
| plotly                   | 5.9.0     | Interactive plotting library                |
| scikit_image             | 0.19.3    | Image processing algorithms                 |
| scikit-learn             | 1.2.1     | Machine learning tools (metrics, etc.)      |
| scipy                    | 1.10.0    | Scientific and technical computing          |
| setuptools               | 69.5.1    | Package building utilities                  |
| six                      | 1.16.0    | Python 2/3 compatibility library          |
| Tensorflow               | 2.14.0    | Deep Learning framework                     |
| Torch                    | 2.3.0     | Deep Learning framework (potentially used)  |
| tqdm                     | 4.64.1    | Progress bars for loops                     |

</details>

> [!NOTE]
> All project libraries are installed automatically within the Docker container when running `make build`.


## ▶️ Usage

* **Run the Main Script:**

  ```bash
  make run
  ```

  This will execute `main.py` inside the Docker container `YOLOnet-AFM`.

* **Monitor Logs:**

  ```bash
  make logs
  ```

* **Access a Shell in the Container:**

  ```bash
  make shell
  ```

## 🛠️ Makefile Commands

The following commands are defined in the `Makefile` to simplify your workflow:

| Comando        | Descrição                                                               |
| -------------- | ----------------------------------------------------------------------- |
| `make build`   | Constrói a imagem Docker definida em `docker/docker-compose.yml`        |
| `make up`      | Inicializa os containers em modo detached (`-d`)                        |
| `make down`    | Encerra e remove os containers                                          |
| `make restart` | Reinicia o ambiente (equivalente a `make down && make up`)              |
| `make logs`    | Mostra os logs do serviço `yolonet_afm`                                 |
| `make shell`   | Abre um terminal interativo dentro do container                         |
| `make run`     | Executa o script `main.py` dentro do container                          |
| `make clean`   | Remove imagens e volumes Docker não utilizados (libera espaço em disco) |

## 🔄 Workflow Options

* **User Options:**
  The script will prompt you to select an option:

  * **Option 1:** Run the **AFM-Only** model pipeline
  * **Option 2:** Run the **Combined Optical+AFM** model pipeline
  * **Option 3:** Run the **Optical-Only** model pipeline

Each pipeline loads a pre-trained model from the `models/` folder, applies preprocessing, generates segmentation masks, and saves results in `data/output/<model-selected>/`.


## 📁 Folder Structure

<details>
<summary>Click to show </summary>

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


## 📜 Script Details

<details>
<summary>Click to expand script descriptions</summary>

---

### :scissors: `1_cropping_opt_images.py`

*   **Purpose:** Aligns optical images with AFM data by cropping the region of interest and resizing.
*   **Input:**
    *   `data/raw/optical_images/`
    *   `data/raw/bw_images/` (Likely ground truth masks used for alignment/reference)
    *   `data/raw/txt_files/` (AFM data files, potentially for coordinates)
*   **Output:**
    *   `data/input/optical_images_resized/`

---

### :open_file_folder: `2_preprocess_unet.py`

*   **Purpose:** Prepares the final input tensors for the U-Net model by combining/formatting optical and/or AFM data channels as required for each specific model type. Also prepares corresponding masks.
*   **Input:**
    *   `data/input/optical_images_resized/`
    *   `data/input/Usefull_data/` (Supporting AFM info)
    *   Raw data sources as needed (`data/raw/`)
*   **Output (Example for Combined):**
    *   `data/intermediate/pre_processing_optico_and_afm/image/`
    *   `data/intermediate/pre_processing_optico_and_afm/mask/`
    *   *(Similar outputs created in `pre_processing_afm` and `pre_processing_optico` directories depending on execution)*

---

### :dart: `3_predicts.py`

*   **Purpose:** Loads a pre-trained model and generates segmentation predictions on a specified preprocessed dataset.
*   **Input:**
    *   Preprocessed images: `data/intermediate/pre_processing_<model-selected>/image/`
    *   Trained model file: `models/<model_file_name>`
*   **Output:**
    *   Predicted masks: `data/output/<model-selected>/predicts/`

---

### :bar_chart: `4_eval_models.py`

*   **Purpose:** Compares predicted segmentation masks against ground truth masks to evaluate model performance.
*   **Input:**
    *   Predicted masks: `data/output/<model-selected>/predicts/`
    *   Ground truth masks: `data/intermediate/pre_processing_<model-selected>/mask/`
*   **Output:**
    *   Metrics visualization: `data/output/<model-selected>/model_metrics.png`
    *   (Potentially) Text files or logs with detailed metrics.

---
</details>


## 🛠️ Built With

*   **Languages:** Python
*   **Frameworks:** TensorFlow, Keras, PyTorch
*   **Libraries:** OpenCV, Scikit-Image, Scikit-Learn, Pandas, NumPy, Matplotlib, Plotly
*   **Tools:** Docker, Git
*   **Editors/Management:** Visual Studio Code, Notion (Task Management)

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeatureName`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/YourFeatureName`).
6.  Open a Pull Request.

Please ensure your code adheres to existing style conventions and includes relevant tests if applicable.

<!-- ## 📄 License

This project is licensed under the [NAME OF LICENSE - e.g., MIT License] - see the `LICENSE` file (if available) for details.
*(If no LICENSE file exists, consider adding one, e.g., MIT, Apache 2.0)* -->

## ✒️ Authors

<table align="center" style="border: none; background: none;">
  <tr style="border: none; background: none;">
    <td align="center" style="border: none; background: none;">
      <a href="https://github.com/ArtRocha">
        <img src="https://media.licdn.com/dms/image/v2/D4D03AQEDHHDwcJ71-Q/profile-displayphoto-shrink_200_200/profile-displayphoto-shrink_200_200/0/1676640793420?e=1741824000&v=beta&t=PyCIUjPJ4R1YoHwnBFh7H-MELeTjA-BwU-dO-zz10Rk" width="150px;" alt="Foto Arthur Rocha" /><br>
        <sub>
          <b style="font-size:16px"> Arthur Rocha </b>
        </sub>
      </a>
    </td>
    <td align="center" style="border: none; background: none;">
         <a href="http://lattes.cnpq.br/8207473893996045">
           <img src="https://media.licdn.com/dms/image/v2/C4D03AQHao6xS_1V4MQ/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1517361324904?e=1741824000&v=beta&t=utQZ1VrmSCGQzZePZkn12QNi_30F05B61aplPDLTYkM" width="150px;" alt="Foto de Ayumi Aurea"/><br>
           <sub>
             <b style="font-size:16px"> Ayumi Aurea Miyakawa </b><br>
           </sub>
         </a>
       </td>
        <td align="center" style="border: none; background: none;">
      <a href="http://lattes.cnpq.br/0399495551887391">
        <img src="http://servicosweb.cnpq.br/wspessoa/servletrecuperafoto?tipo=1&id=K8153430T2" width="150px;" alt="Foto de Cleyton Biffe"/><br>
        <sub>
          <b style="font-size:16px"> Cleyton Biffe </b><br>
        </sub>
      </a>
    </td>
    <td align="center" style="border: none; background: none;">
      <a href="https://github.com/EdCarlos-dev">
        <img src="https://media.licdn.com/dms/image/v2/D4D03AQHPPE38HWKxgQ/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1682383748193?e=1741824000&v=beta&t=WJhEASVpf3eCXmboJok2n1dxDRvvxkApnQao0h4KdGo" width="150px;" alt="Foto de Ed Santos e Silva"/><br>
        <sub>
          <b style="font-size:16px"> Ed Santos e Silva </b><br>
        </sub>
      </a>
    </td>
     <td align="center" style="border: none; background: none;">
      <a href="http://lattes.cnpq.br/9674023945962136">
        <img src="https://lh3.googleusercontent.com/a-/ALV-UjUC_CbCkHR3n6mjft683RBUYRlmXo9xZdX01RdaTErJyQ=s272-p-k-rw-no" width="150px;" alt="Foto de Jose Patane"/><br>
        <sub>
          <b style="font-size:16px"> Jose Patane </b><br>
        </sub>
      </a>
    </td>
  </tr>
</table>

## 🙏 Acknowledgments

<div align="center">
 <img src="https://genetica.incor.usp.br/wp-content/uploads/2021/05/cropped-temp.png" width="300px;" alt="InCor Logo"/>

 © Todos os direitos reservados, 2025 - Instituto do Coração (InCor) HCFMUSP
</div>

