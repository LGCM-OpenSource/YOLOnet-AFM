import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2

from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import Recall, Precision

from utils import build_unet, build_unet_2_layers, build_unet_3_layers, build_unet_5_layers, iou, dice_coef
from half_unet_batch_normalization_model import build_half_unet_model_batch_normalization
from unet_plus_plus_model import build_unet_plusplus_model

import argparse
import sys 
import pandas as pd


# Verificar GPUs disponíveis
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Função para criar diretórios se não existirem
def create_dir(path):
    '''
    Parameters
    ----------
    path(string): recieve a directory name 

    Return
    -------
    Create a path directory if no existis
    '''
    if not os.path.exists(path):
        os.makedirs(path)

# Função para carregar e pré-processar os dados
def load_data(image_paths, mask_paths):
    images = [np.load(img_path) for img_path in image_paths]
    masks = [np.load(mask_path) for mask_path in mask_paths]
    return images, masks

# Função para criar dataset TensorFlow a partir de dados pré-processados
def create_tf_dataset(images, masks, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

if __name__ == "__main__":
    #Seeding
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Parâmetros
    H, W = 256, 256
    batch_size = 16
    num_epochs = 100
    learning_rate = 1e-4
    model_folder = 'models'
    
    monitor = 'val_loss'

    create_dir(model_folder)

    model_select = [
                #   'train_1_channels_only_AFM_CosHeightSum',
                  'train_2_channels_only_optical',
                #   'train_2_channels_like_yolo_opt_afm',
                  ]
    
    train_sizes = [234]
    # layers = [(2, build_unet_2_layers((H,W,2))),(3, build_unet_3_layers((H,W,2))),(5, build_unet_5_layers((H,W,2)))] 
    layers = [(4, build_half_unet_model_batch_normalization((H,W,2)))] 
    
    for train_path in model_select:

        channels = 2
        if train_path == 'train_1_channels_only_AFM_CosHeightSum':
            channels = 1

        for i in train_sizes:
            # indentar depois
            tf.keras.backend.clear_session()
            tmp_df_files = pd.read_csv(f'data_complete/datasets/df_train_{i}_selected.csv')
            
            
            for lay, mod in layers:
                name = f'half_unet_optical{train_path[5:]}_{i}_samples_stardist_mask'
                # model = build_unet((H,W,channels))
                model = mod
                print(f'Training {name} model...')

                dataset_path = f'data_complete/input/train/{train_path}/'

                suffixes = [
                    '_channels_added_90.npy',
                    '_channels_added_180.npy',
                    '_channels_added_270.npy',
                    '_channels_added_flip_90.npy',
                    '_channels_added_flip_180.npy',
                    '_channels_added_flip_270.npy',
                    '_channels_added_flip.npy',
                    '_channels_added.npy'
                ]

                # Criar image_paths e mask_paths em uma linha
                image_paths = [os.path.join(f'{dataset_path}opt_img_training', f"{file}{suffix}") for file in tmp_df_files['Process.Date'].values for suffix in suffixes]
                mask_paths  = [os.path.join(f'{dataset_path}msk_img_training', f"{file}{suffix}") for file in tmp_df_files['Process.Date'].values for suffix in suffixes]

                train_ratio = 1
                num_samples = len(image_paths)
                num_train = int(num_samples * train_ratio)
                num_validation = int(num_samples - num_train)

                train_image_paths = image_paths[:num_train]
                train_mask_paths = mask_paths[:num_train]
                # validation_image_paths = image_paths[num_train:num_train + num_validation]
                # validation_mask_paths = mask_paths[num_train:num_train + num_validation]

                print(f'Train SIZE: {num_train}')
                print(f'Validation SIZE: {num_validation}')


                train_images, train_masks = load_data(train_image_paths, train_mask_paths)
                # valid_images, valid_masks = load_data(validation_image_paths, validation_mask_paths)

                train_dataset = create_tf_dataset(train_images, train_masks, batch_size)
                # valid_dataset = create_tf_dataset(valid_images, valid_masks, batch_size)

                model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate), metrics=[dice_coef, iou, tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

                # Callbacks
                model_checkpoint = ModelCheckpoint(os.path.join(model_folder, f"{name}.h5"), monitor=monitor, save_best_only=True, verbose=1)
                reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=5, min_lr=1e-7, verbose=1)
                csv_logger = CSVLogger(os.path.join(model_folder, f"training_{name}.log"))
                early_stopping = EarlyStopping(monitor=monitor, patience=20, restore_best_weights=False)

                history = model.fit(
                    train_dataset,
                    epochs=num_epochs,
                    # validation_data=valid_dataset,
                    callbacks=[model_checkpoint, reduce_lr, csv_logger, early_stopping]
                )