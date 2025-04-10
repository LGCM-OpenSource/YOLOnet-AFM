import os
import argparse
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from utils import (build_half_unet_model_batch_normalization,
                   build_unet, iou, dice_coef, create_dir,
                   load_config, UNET_MODELS_PATH,
                   setup_logger, get_logger, YAML_FILE) 
from tensorflow.python.client import device_lib
from glob import glob
import traceback 
import logging 

# --- Logger Setup ---
# Setup a logger for this script
logger = setup_logger('training_process', level=logging.INFO) # Use INFO level, or DEBUG for more detail

def check_gpu_availability():
    """Checks for available GPUs and configures memory growth. Logs results."""
    logger.info("Checking for available GPUs...")
    try:
        devices = device_lib.list_local_devices()
        logger.debug(f"All local devices: {devices}")
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            logger.info(f"Found {len(gpus)} physical GPU(s): {gpus}")
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.debug(f"Enabled memory growth for GPU: {gpu.name}")
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                logger.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs configured.")
            except RuntimeError as e:
                logger.error(f"RuntimeError configuring GPU memory growth: {e}")
        else:
            logger.warning("No GPUs found. Training will use CPU.")
    except Exception as e:
        logger.error(f"Error checking GPU availability: {e}")
        logger.error(traceback.format_exc())


def load_data(image_paths, mask_paths):
    """Loads image and mask data from .npy files. Logs counts and errors."""
    logger.info(f"Loading data from {len(image_paths)} image paths and {len(mask_paths)} mask paths.")
    if len(image_paths) != len(mask_paths):
        logger.error("Mismatch between number of image paths and mask paths!")
        raise ValueError("Number of image paths and mask paths must be equal.")
    if not image_paths:
        logger.warning("No image paths provided to load_data.")
        return [], []

    images = []
    masks = []
    load_errors = 0
    for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        try:
            logger.debug(f"Loading image: {img_path}")
            img = np.load(img_path)
            logger.debug(f"Loading mask: {mask_path}")
            mask = np.load(mask_path)
            images.append(img)
            masks.append(mask)
        except FileNotFoundError as fnf_error:
            logger.error(f"File not found during data loading: {fnf_error}")
            load_errors += 1
        except Exception as e:
            logger.error(f"Error loading data pair (Image: {img_path}, Mask: {mask_path}): {e}")
            logger.error(traceback.format_exc())
            load_errors += 1

    if load_errors > 0:
        logger.error(f"Encountered {load_errors} errors during data loading.")
        # Decide if you want to raise an error or continue with partial data
        # raise RuntimeError(f"Failed to load {load_errors} data pairs.")

    logger.info(f"Successfully loaded {len(images)} images and {len(masks)} masks.")
    return images, masks

def create_tf_dataset(images, masks, batch_size):
    """Creates a TensorFlow dataset from preprocessed data. Logs details."""
    logger.info(f"Creating TensorFlow dataset with batch size: {batch_size}")
    if not images or not masks:
        logger.error("Cannot create dataset: images or masks list is empty.")
        raise ValueError("Images and masks lists cannot be empty for dataset creation.")
    try:
        dataset = tf.data.Dataset.from_tensor_slices((np.array(images), np.array(masks))) # Convert lists to numpy arrays first
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        logger.info("TensorFlow dataset created successfully.")
        return dataset
    except Exception as e:
        logger.error(f"Error creating TensorFlow dataset: {e}")
        logger.error(traceback.format_exc())
        raise

def select_unet_architecture(model_type, input_shape=(256, 256, 1)):
    """Selects the U-Net architecture based on the model_type. Logs selection."""
    logger.info(f"Selecting U-Net architecture. Type: '{model_type}', Input Shape: {input_shape}")
    try:
        if model_type == "unet":
            logger.info("Building standard U-Net model.")
            model_builder = build_unet(input_shape)
        else: # Assuming 'half-unet' or default to it
            logger.info("Building Half-UNet model.")
            model_builder = build_half_unet_model_batch_normalization(input_shape)
        logger.info(f"Model architecture '{model_type}' selected.")
        return model_builder
    except Exception as e:
        logger.error(f"Error building model architecture '{model_type}': {e}")
        logger.error(traceback.format_exc())
        raise


def parse_arguments():
    """Parses command-line arguments. Logs the arguments."""
    logger.debug("Parsing command-line arguments.")
    parser = argparse.ArgumentParser(description="Segmentation models training script")
    parser.add_argument("-c", "--config", type=str, default=f"{YAML_FILE}", help="Path to the config.yaml file")
    parser.add_argument("-mt", "--model_type", type=str, choices=["unet", "half-unet"], help="Select unet arch: 'unet' or 'half-unet'")
    parser.add_argument('-ms', '--model_selection', type=str, help="select your model to choice preprocess step to make segmentations predictions")
    parser.add_argument("-bs", "--batch_size", type=int, help="Batch size for training.")
    parser.add_argument("-ne", "--num_epochs", type=int, help="Number of training epochs.")
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate for the optimizer.")
    args = parser.parse_args()
    logger.info(f"Command-line arguments parsed: {args}")
    return args

def train_model(model, train_dataset, valid_dataset, model_name, config):
    """Trains the specified model. Logs configuration and progress."""
    logger.info(f"Starting training for model: {model_name}")
    try:
        # Log compilation parameters
        optimizer_config = Adam(config["training"]["learning_rate"]).get_config()
        logger.info(f"Compiling model with loss: 'binary_crossentropy', optimizer: Adam({optimizer_config}), metrics: [dice_coef, iou, Recall, Precision]")
        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(config["training"]["learning_rate"]),
            metrics=[dice_coef, iou, tf.keras.metrics.Recall(), tf.keras.metrics.Precision()],
        )
        model.summary(print_fn=logger.info) # Log model summary

        # Callbacks setup and logging
        model_save_path = os.path.join(config["general"]["model_folder"], f"{model_name}.h5")
        log_save_path = os.path.join(config["general"]["model_folder"], f"training_{model_name}.log")
        monitor_metric = config["general"]["monitor"]

        logger.info("Setting up callbacks:")
        model_checkpoint = ModelCheckpoint(
            model_save_path, monitor=monitor_metric, save_best_only=True, verbose=1
        )
        logger.info(f" - ModelCheckpoint: Saving best model to {model_save_path}, monitoring '{monitor_metric}'")

        reduce_lr = ReduceLROnPlateau(
            monitor=monitor_metric, factor=0.1, patience=5, min_lr=1e-7, verbose=1
        )
        logger.info(f" - ReduceLROnPlateau: Monitoring '{monitor_metric}', factor=0.1, patience=5")

        csv_logger = CSVLogger(log_save_path)
        logger.info(f" - CSVLogger: Saving training logs to {log_save_path}")

        early_stopping = EarlyStopping(
            monitor=monitor_metric, patience=20, restore_best_weights=False, verbose=1
        )
        logger.info(f" - EarlyStopping: Monitoring '{monitor_metric}', patience=20")

        callbacks_list = [model_checkpoint, reduce_lr, csv_logger, early_stopping]

        # Start training
        logger.info(f"Starting model.fit with {config['training']['num_epochs']} epochs.")
        history = model.fit(
            train_dataset,
            epochs=config["training"]["num_epochs"],
            validation_data=valid_dataset,
            callbacks=callbacks_list,
            verbose=1 # Keep verbose=1 to see epoch progress in console/logs
        )
        logger.info(f"Training finished for model: {model_name}")
        return history

    except Exception as e:
        logger.error(f"Error during model training for {model_name}: {e}")
        logger.error(traceback.format_exc())
        raise
def main():
    logger.info("=============================================")
    logger.info("Starting Training Script...")

    try:
        # Parse command-line arguments
        args = parse_arguments()

        # Load the configuration from the YAML file
        logger.info(f"Loading configuration from: {args.config}")
        try:
            config = load_config(args.config) # Pass the path from args
            logger.info("Configuration loaded successfully.")
            logger.debug(f"Config details: {config}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found at: {args.config}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            raise

        # Override configuration values with command-line arguments if provided
        logger.info("Overriding config with command-line arguments if provided...")
        if args.model_type:
            logger.info(f"Overriding model_type: {config['model']['model_type']} -> {args.model_type}")
            config["model"]["model_type"] = args.model_type
        if args.model_selection:
            logger.info(f"Overriding model_selection: {config['model'].get('model_selection', 'N/A')} -> {args.model_selection}")
            config["model"]["model_selection"] = args.model_selection
        if args.batch_size:
            logger.info(f"Overriding batch_size: {config['training']['batch_size']} -> {args.batch_size}")
            config["training"]["batch_size"] = args.batch_size
        if args.num_epochs:
            logger.info(f"Overriding num_epochs: {config['training']['num_epochs']} -> {args.num_epochs}")
            config["training"]["num_epochs"] = args.num_epochs
        if args.learning_rate:
            logger.info(f"Overriding learning_rate: {config['training']['learning_rate']} -> {args.learning_rate}")
            config["training"]["learning_rate"] = args.learning_rate

        # Reproducibility
        seed = config["general"]["seed"]
        logger.info(f"Setting random seeds for reproducibility: {seed}")
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Check GPU availability
        check_gpu_availability()

        # Create model folder
        model_folder = config["general"]["model_folder"]
        logger.info(f"Ensuring model save directory exists: {model_folder}")
        create_dir(model_folder)

        # Training parameters
        H, W = config["model"]["input_shape"]
        
        model_type = config['model']['model_type']
        model_select = config["model"]["model_selection"]
        
        # --- Determine Channels based on model_selection ---
        # Default to 2 channels unless specified otherwise
        default_channels = 2
        if model_select == 'unet_afm_1_channels_only_AFM_CosHeightSum':
            channels = 1
            logger.info(f"Model selection '{model_select}' implies 1 channel.")
        else:
            channels = config["model"].get("channels", default_channels) # Use config value or default
            logger.info(f"Using {channels} channels based on config or default.")
        # Update config dict if necessary (optional, but good practice)
        config["model"]["channels"] = channels
        batch_size = config["training"]["batch_size"]
        num_epochs = config["training"]["num_epochs"]
        learning_rate = config["training"]["learning_rate"]

        logger.info(f"Training Parameters:")
        logger.info(f"  Input Shape (H, W, C): ({H}, {W}, {channels})")
        logger.info(f"  Model Type: {model_type}")
        logger.info(f"  Model Selection Key: {model_select}")
        logger.info(f"  Batch Size: {batch_size}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Learning Rate: {learning_rate}")
        # Model selection
        model = select_unet_architecture(model_type, (H, W, channels))

        model_train_name = f"{model_type}_{model_select}" # Consistent naming
        logger.info(f"Model '{model_train_name}' built.")

        # --- Data Loading and Splitting ---
        logger.info("Loading data paths...")
        try:
            train_img_dir = UNET_MODELS_PATH[model_select]['train_path']
            train_mask_dir = UNET_MODELS_PATH[model_select]['mask_path']
            logger.info(f"  Train Image Dir: {train_img_dir}")
            logger.info(f"  Train Mask Dir: {train_mask_dir}")

            image_paths = sorted(glob(os.path.join(train_img_dir, '*.npy')))
            mask_paths = sorted(glob(os.path.join(train_mask_dir, '*.npy')))
            if not image_paths:
                logger.error(f"No .npy files found in image directory: {train_img_dir}")
                raise FileNotFoundError(f"No training images found in {train_img_dir}")
            if not mask_paths:
                logger.error(f"No .npy files found in mask directory: {train_mask_dir}")
                raise FileNotFoundError(f"No training masks found in {train_mask_dir}")
            if len(image_paths) != len(mask_paths):
                logger.error(f"Mismatch in number of images ({len(image_paths)}) and masks ({len(mask_paths)}).")
                # Decide how to handle: raise error or try to match? Raising is safer.
                raise ValueError("Number of images and masks found do not match.")

            logger.info(f"Found {len(image_paths)} total image/mask pairs.")

        except KeyError:
            logger.error(f"Invalid model_selection key '{model_select}' not found in UNET_MODELS_PATH.")
            raise
        except Exception as e:
            logger.error(f"Error accessing data paths for model_selection '{model_select}': {e}")
            raise
        train_ratio = config["data"]["train_ratio"]
        num_samples = len(image_paths)
        num_train = int(num_samples * train_ratio)
        # Ensure validation set gets at least 1 sample if possible
        num_validation = max(1, num_samples - num_train) if num_samples > num_train else 0
        # Adjust num_train if validation took the last sample
        num_train = num_samples - num_validation

        logger.info(f"Splitting data: {train_ratio*100:.1f}% train, {(1-train_ratio)*100:.1f}% validation.")

        # Shuffle indices before splitting for better randomness
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        train_indices = indices[:num_train]
        validation_indices = indices[num_train:]

        train_image_paths = [image_paths[i] for i in train_indices]
        train_mask_paths = [mask_paths[i] for i in train_indices]
        validation_image_paths = [image_paths[i] for i in validation_indices]
        validation_mask_paths = [mask_paths[i] for i in validation_indices]

        logger.info(f"Training set size: {len(train_image_paths)}")
        logger.info(f"Validation set size: {len(validation_image_paths)}")

        # Load data into memory
        logger.info("Loading training data into memory...")
        train_images, train_masks = load_data(train_image_paths, train_mask_paths)
        logger.info("Loading validation data into memory...")
        valid_images, valid_masks = load_data(validation_image_paths, validation_mask_paths)

        train_dataset = create_tf_dataset(train_images, train_masks, batch_size)
        valid_dataset = create_tf_dataset(valid_images, valid_masks, batch_size)

        # Train the model
        train_model(model, train_dataset, valid_dataset, model_train_name, config)

        logger.info("Training script finished successfully.")

    except FileNotFoundError as fnf_error:
        logger.critical(f"Critical file not found error: {fnf_error}")
    except KeyError as key_error:
        logger.critical(f"Configuration key error: {key_error}. Check config.yaml and UNET_MODELS_PATH.")
    except ValueError as val_error:
        logger.critical(f"Value error during setup or processing: {val_error}")
    except Exception as main_exc:
        logger.critical("An unexpected critical error occurred during training.")
        logger.critical(traceback.format_exc())
    finally:
        tf.keras.backend.clear_session() # Clear session regardless of success/failure
        logger.info("TensorFlow backend session cleared.")
        logger.info("=============================================\n")
if __name__ == "__main__":
    main()
