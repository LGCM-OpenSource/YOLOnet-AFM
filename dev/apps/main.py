# /home/arthur/lgcm/projects/Unet_AFM/dev/apps/main.py
from utils import UserInput, setup_logger, get_logger # <-- Import logger functions
import sys
import platform
import subprocess
import os
from tqdm import tqdm
import traceback # <-- Import traceback for detailed error logging

# --- CONFIGURAÇÕES DO SCRIPT ---

# Diretório atual do script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Scripts para treino de novo modelo
training_flow_scripts = {
    'CROP_IMG': ('1_cropping_opt_images.py', 'Cropping images...'),
    'PREP_UNET': ('2_preprocess_unet.py', 'Generating optical AFM image...'),
    'DATA_AUG': ('2b_data_augmentation.py', 'Applying data augmentation...'),
    'TRAINING': ('training_models.py', 'Training new model...')
}

# Scripts para predição normal (caso queira manter)
process_flow_scripts_dict = {
    'CROP_IMG': ('1_cropping_opt_images.py', 'Cropping images...'),
    'PREP_UNET': ('2_preprocess_unet.py', 'Generating optical AFM image...'),
    'PRED_UNET': ('3_predicts.py', 'Segmenting cells...'),
    'EVAL_MODEL': ('4_eval_models.py', 'Getting general metrics...')
}

# --- FUNÇÕES AUXILIARES ---

def run_by_operating_system(abs_path_file, logger):
    """Runs a command based on the operating system and logs the outcome."""
    try:
        logger.info(f"Executing command: {abs_path_file}")
        # Use check=True to raise CalledProcessError on failure (non-zero exit code)
        # Capture stdout and stderr
        result = subprocess.run(
            abs_path_file.split(), # Split command properly
            shell=False, 
            check=True,
            text=True,
            cwd=current_directory,
            stdout=sys.stdout,
            stderr=sys.stderr
            # Ensure scripts run from the correct directory
        )
        logger.info(f"Command stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Command stderr:\n{result.stderr}")
        return True # Indicate success
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}.")
        logger.error(f"Stderr:\n{e.stderr}")
        logger.error(f"Stdout:\n{e.stdout}")
        return False # Indicate failure
    except FileNotFoundError:
        logger.error(f"Error: Python executable or script not found: {abs_path_file}")
        return False # Indicate failure
    except Exception as e:
        logger.error(f"An unexpected error occurred while running command: {abs_path_file}")
        logger.error(traceback.format_exc())
        return False # Indicate failure


def run_script(file, msg, logger, args=None):
    """Constructs the command and runs a script using run_by_operating_system."""
    script_path = os.path.join(current_directory, file)
    command_parts = [sys.executable, script_path] # Start with python executable and script path

    if args:
        for k, v in args.items():
             command_parts.extend([str(k), str(v)]) # Add arguments correctly

    command_str = " ".join(command_parts) # Create string representation for logging

    logger.info(f"--- Starting: {file} ({msg}) ---")
    # Use the modified run_by_operating_system which now returns True/False
    success = run_by_operating_system(command_str, logger) # Pass the command string

    if success:
        logger.info(f"--- Finished: {file} ({msg}) ---")
    else:
        logger.error(f"--- Failed: {file} ({msg}) ---")
    return success # Return the success status


# --- EXECUÇÃO PRINCIPAL ---

if __name__ == "__main__":
    # Setup logger for this run
    main_logger = setup_logger('main_process')
    main_logger.info("=============================================")
    main_logger.info("Main process started.")
    main_logger.info(f"Running in directory: {current_directory}")
    main_logger.info(f"Python executable: {sys.executable}")
    main_logger.info(f"Platform: {platform.system()}")

    all_steps_succeeded = True # Flag to track overall success

    try:
        # Seleção do modelo
        model_selector = UserInput.select_model()
        if not model_selector:
            main_logger.error("No model selected. Exiting.")
            sys.exit(1) # Exit if no model is chosen
        main_logger.info(f"Model selected: {model_selector}")

        # Pergunta se o usuário quer treinar um novo modelo
        opcao = UserInput.select_operation()
        if not opcao:
            main_logger.error("No operation selected. Exiting.")
            sys.exit(1) # Exit if no operation is chosen
        main_logger.info(f"Operation selected: {opcao}")

        select_arch = None # Initialize select_arch
        visualize_confirm = False 
        if opcao.strip().lower() == "train":
            select_arch = UserInput.select_training_type()
            if not select_arch:
                main_logger.error("No training architecture selected. Exiting.")
                sys.exit(1) # Exit if no architecture is chosen
            main_logger.info(f"Training architecture selected: {select_arch}")
            main_logger.info("Initializing training process flow...")
            flow = training_flow_scripts
        else:
            visualize_confirm = UserInput.get_user_confirmation('DO YOU WANNA VISUALIZE SEGMENTATION RESULTS?')
            main_logger.info(f"Visualize segmentation results: {visualize_confirm}")
            main_logger.info("Initializing segmentation process flow...")
            flow = process_flow_scripts_dict

        # Executa os scripts na ordem definida
        # Use tqdm for visual progress, but logging handles the detailed tracking
        for key in tqdm(flow, desc="Processing Steps", colour='green'):
            script_file, description = flow[key]
            script_args = {} # Common argument

            if script_file != '1_cropping_opt_images.py':
                 script_args["-ms"] = model_selector 
                 
            if script_file == 'training_models.py' and opcao.strip().lower() == "train":
                 script_args["-mt"] = select_arch # Add training type argument only if training
                    # --- Conditionally add --visualize for 3_predicts.py ---
            if script_file == '3_predicts.py' and visualize_confirm:
                 script_args["-vis"] = True # Add the flag if confirmed
             
            if not script_args:
                script_args = None
            
            # Pass the logger instance to run_script
            success = run_script(script_file, description, main_logger, args=script_args)

            if not success:
                all_steps_succeeded = False
                main_logger.error(f"Process stopped due to failure in script: {script_file}")
                break # Stop the loop if a script fails

        if all_steps_succeeded:
            main_logger.info("All process steps finished successfully!")
        else:
             main_logger.error("Process finished with errors.")

    except Exception as e:
        main_logger.critical("An unexpected critical error occurred in the main execution block.")
        main_logger.critical(traceback.format_exc())
        all_steps_succeeded = False

    finally:
         main_logger.info("Main process finished.")
         main_logger.info("=============================================\n")
         # Optional: Close handlers if necessary, though Python usually handles this
         # for handler in main_logger.handlers[:]:
         #     handler.close()
         #     main_logger.removeHandler(handler)

