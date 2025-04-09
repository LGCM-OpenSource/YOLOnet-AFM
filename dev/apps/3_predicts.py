import os
import sys
from utils import UnetProcess, Models, build_file_path, CROP_PATH, UNET_MODELS_PATH, TRAIN_TEST_FILES, TerminalStyles, UserInput
from tqdm import tqdm
import argparse
import traceback # Import traceback for better error logging if needed

def build_paths(files_list, actual_process='_channels_added.npy'):
    """Builds and validates paths for processing."""
    valid_data = []
    print(f"Building paths for {len(files_list)} potential files...") # Add logging

    for img_file in files_list:
        # Ensure we are working with the base filename if needed
        base_name = img_file.replace(actual_process, '')

        opt = build_file_path(CROP_PATH['optical_crop_resized'], base_name, actual_process='', new_process='_optico_crop_resized.png')
        use = build_file_path(CROP_PATH['usefull_data'], base_name, actual_process='', new_process='_UsefullData.tsv')
        pre = os.path.join(model_info['preprocess_img'], img_file) # Use the original file name from listdir
        mask = os.path.join(model_info['preprocess_mask'], img_file) # Use the original file name from listdir
        save = os.path.join(model_info['save_predict'], base_name + '_unet.png') # Construct save path from base name

        paths = {
            "optical": opt,
            "usefull": use,
            "preprocess_img": pre,
            "mask": mask
        }

        missing = [name for name, path in paths.items() if not os.path.exists(path)]

        if missing:
            # Use print for now, consider using the logger if passed down
            print(f"[WARN] File '{img_file}' related paths ignored. Missing: {', '.join(missing)}")
            print(f"  - Optical expected: {opt}")
            print(f"  - Usefull expected: {use}")
            print(f"  - Preprocess expected: {pre}")
            print(f"  - Mask expected: {mask}")
        else:
            valid_data.append((opt, use, pre, mask, save))

    if not valid_data:
        print("[ERROR] No valid data found after checking paths.") # Add logging
        return [], [], [], [], []

    print(f"Found {len(valid_data)} valid sets of files.") # Add logging
    opt_paths, use_paths, pre_paths, mask_paths, save_paths = zip(*valid_data)
    return list(opt_paths), list(use_paths), list(pre_paths), list(mask_paths), list(save_paths)

# --- Argument Parsing ---
term = TerminalStyles()
parser = argparse.ArgumentParser(description="Run U-Net predictions.")
parser.add_argument('-ms', '--model_selection', type=str, required=True, help="Select model configuration key (e.g., 'unet_afm_1_channels_only_AFM_CosHeightSum').")
parser.add_argument('-vis','--visualize',type=bool, default=False, help="Enable visualization of segmentation results after prediction.") # Added argument

args = parser.parse_args()
model_selector = args.model_selection
visualize_segmentation = args.visualize # Use the argument value

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting prediction process for model: {model_selector}")
    print(f"Visualization enabled: {visualize_segmentation}")

    try:
        model_info = UNET_MODELS_PATH[model_selector]

        # Ensure output directory exists
        os.makedirs(model_info['save_predict'], exist_ok=True)

        # List files in the preprocess directory (assuming these are the inputs)
        if not os.path.isdir(model_info['preprocess_img']):
             print(f"[ERROR] Preprocess image directory not found: {model_info['preprocess_img']}")
             sys.exit(1)

        files_list = os.listdir(model_info['preprocess_img'])
        if not files_list:
            print(f"[ERROR] No files found in preprocess image directory: {model_info['preprocess_img']}")
            sys.exit(1)

        # Build and validate paths
        opt_image_path, usefull_path, preprocess_image_path, mask_path, save_path_list = build_paths(files_list)

        # Check if any valid paths were found
        if not opt_image_path:
            print("[ERROR] No valid file sets found to process. Exiting.")
            sys.exit(1)

        # Load the model
        print(f"Loading model: {model_info['model_name']} from {model_info['model_path']}")
        if not os.path.exists(model_info['model_path']):
            print(f"[ERROR] Model file not found: {model_info['model_path']}")
            sys.exit(1)
            
        model = Models(model_info['model_name'], model_info["model_path"])
        print("Model loaded successfully.")

        # Process each valid file set
        print(f"Processing {len(opt_image_path)} images...")
        for i in tqdm(range(len(opt_image_path)), desc="Predicting", colour='#0000FF'):
            try:
                unetTrat = UnetProcess(opt_image_path[i], preprocess_image_path[i], usefull_path[i], mask_path[i])
                y_pred = unetTrat.unet_predict(model)

                if y_pred is None:
                    print(f"[WARN] Prediction failed for {os.path.basename(opt_image_path[i])}. Skipping.")
                    continue

                y_resized = unetTrat.resize_prediction_to_original_size(y_pred)
                unetTrat.save_predict(y_resized, save_path_list[i])

                if visualize_segmentation:
                    unetTrat.visualize_prediction(save_path_list[i])

            except FileNotFoundError as fnf_error:
                print(f"\n[ERROR] File not found during processing loop for index {i}: {fnf_error}")
            except Exception as loop_error:
                print(f"\n[ERROR] Unexpected error during processing loop for index {i} ({os.path.basename(opt_image_path[i])}):")
                traceback.print_exc() # Print detailed traceback for loop errors

        print(f'''
            {term.BOLD}{model_selector}{term.RESET}
            Predictions saved in {term.SAVE_COLOR}{model_info['save_predict']}{term.RESET}
            ''')

    except KeyError:
        print(f"[ERROR] Invalid model selection key: '{model_selector}'. Check UNET_MODELS_PATH in data_path.py.")
        sys.exit(1)
    except Exception as main_error:
        print("\n[CRITICAL ERROR] An unexpected error occurred in the main prediction script:")
        traceback.print_exc()
        sys.exit(1)