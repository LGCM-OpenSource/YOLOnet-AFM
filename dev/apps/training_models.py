import os
import argparse
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from utils import build_unet, iou, dice_coef
from utils import build_half_unet_model_batch_normalization, load_config, UNET_MODELS_PATH
import pandas as pd
from tensorflow.python.client import device_lib
from glob import glob

def check_gpu_availability():
    """Checks for available GPUs and configures memory growth."""
    print(device_lib.list_local_devices())
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

def create_dir(path):
    """Creates a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def load_data(image_paths, mask_paths):
    """Loads image and mask data from .npy files."""
    images = [np.load(img_path) for img_path in image_paths]
    masks = [np.load(mask_path) for mask_path in mask_paths]
    return images, masks

def create_tf_dataset(images, masks, batch_size):
    """Creates a TensorFlow dataset from preprocessed data."""
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def select_unet_architecture(model_type, input_shape=(256, 256, 1)):
    """Selects the U-Net architecture based on the model_type."""
    if model_type == "unet":
        print("Using U-Net model")
        model_builder = build_unet(input_shape)
    else:
        print("Using Half-UNet model")
        model_builder = build_half_unet_model_batch_normalization(input_shape)
    return model_builder


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Segmentation models training script")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to the config.yaml file")
    parser.add_argument("-mt", "--model_type", type=str, choices=["unet", "half-unet"], help="Select unet arch: 'unet' or 'half-unet'")
    parser.add_argument('-ms', '--model_selection', type=str, help="select your model to choice preprocess step to make segmentations predictions")
    parser.add_argument("-bs", "--batch_size", type=int, help="Batch size for training.")
    parser.add_argument("-ne", "--num_epochs", type=int, help="Number of training epochs.")
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate for the optimizer.")
    return parser.parse_args()

def train_model(model, train_dataset, valid_dataset, model_name, config):
    """Trains the specified model."""
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(config["training"]["learning_rate"]),
        metrics=[dice_coef, iou, tf.keras.metrics.Recall(), tf.keras.metrics.Precision()],
    )

    # Callbacks
    model_checkpoint = ModelCheckpoint(
        os.path.join(config["general"]["model_folder"], f"{model_name}.h5"),
        monitor=config["general"]["monitor"],
        save_best_only=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor=config["general"]["monitor"], factor=0.1, patience=5, min_lr=1e-7, verbose=1
    )
    csv_logger = CSVLogger(os.path.join(config["general"]["model_folder"], f"training_{model_name}.log"))
    early_stopping = EarlyStopping(
        monitor=config["general"]["monitor"], patience=20, restore_best_weights=False
    )

    history = model.fit(
        train_dataset,
        epochs=config["training"]["num_epochs"],
        validation_data=valid_dataset,
        callbacks=[model_checkpoint, reduce_lr, csv_logger, early_stopping],
    )
    return history

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Load the configuration from the YAML file
    config = load_config()

    # Override configuration values with command-line arguments if provided
    if args.model_type:
        config["model"]["model_type"] = args.model_type  
    if args.model_selection:
        config["model"]["model_selection"] = args.model_selection
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.num_epochs:
        config["training"]["num_epochs"] = args.num_epochs
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate

    # Reproducibility
    np.random.seed(config["general"]["seed"])
    tf.random.set_seed(config["general"]["seed"])

    # Check GPU availability
    check_gpu_availability()

    # Create model folder
    create_dir(config["general"]["model_folder"])

    # Training parameters
    H, W = config["model"]["input_shape"]
    
    model_type = config['model']['model_type']
    model_select = config["model"]["model_selection"]
    
    if model_select == 'unet_afm_1_channels_only_AFM_CosHeightSum':
        config["model"]["channels"] = 1
    
    channels = config["model"]["channels"]
    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["num_epochs"]
    learning_rate = config["training"]["learning_rate"]

    # Model selection
    model = select_unet_architecture(model_type, (H, W, channels))

    name = f"{model_type}_{model_select}"
    print(f"Training {name} model...")


    image_paths = sorted(glob(os.path.join(UNET_MODELS_PATH[model_select]['train_path'], '*.npy')))
    mask_paths = sorted(glob(os.path.join(UNET_MODELS_PATH[model_select]['mask_path'], '*.npy')))


    train_ratio = config["data"]["train_ratio"]
    num_samples = len(image_paths)
    num_train = int(num_samples * train_ratio)
    num_validation = int(num_samples - num_train)

    train_image_paths = image_paths[:num_train]
    train_mask_paths = mask_paths[:num_train]
    validation_image_paths = image_paths[num_train:]
    validation_mask_paths = mask_paths[num_train:]

    print(f"Train SIZE: {num_train}")
    print(f"Validation SIZE: {num_validation}")

    train_images, train_masks = load_data(train_image_paths, train_mask_paths)
    
    valid_images, valid_masks = load_data(
        validation_image_paths, validation_mask_paths
    )

    train_dataset = create_tf_dataset(train_images, train_masks, batch_size)
    valid_dataset = create_tf_dataset(valid_images, valid_masks, batch_size)

    train_model(model, train_dataset, valid_dataset, name, config)
    tf.keras.backend.clear_session()
    
            