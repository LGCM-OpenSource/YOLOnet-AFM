import os
import sys
import platform
import subprocess
from tqdm import tqdm

# Verify and adjust the execution command according to the operating system
def run_by_operating_system(abs_path_file):
    identify = platform.system()
    if identify == 'Linux':
            return subprocess.run([f'{sys.executable} {abs_path_file}'], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
    else:
            return subprocess.run([sys.executable,  f'{abs_path_file}'], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

# Function to execute a script with or without arguments
def run_script(file, msg, arg=None):
    script_path = os.path.join(current_directory, file)

    if arg is not None and file.split('_')[1] == 'eval':
        arg_string = f' -op {arg}'
        script_path = script_path + arg_string
        
    print(f'START {file}... {msg}\nPlease wait')
    run_by_operating_system(script_path)
    print(f'FINISH {file}... {msg}\n')


# Current directory of the script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Dictionary with the scripts to be executed and their corresponding messages
scripts_dict = {
    'CROP_IMG': ('1_cropping_opt_images.py', 'cropping images...'),
    'PREP_UNET': ('2_preprocess_unet.py', 'Generating optical AFM image...'),
    'PREP_PIXEL': ('3_preprocess_pixel.py', 'Normalizing AFM data file...'),
    'PRED_VUNET': ('4_vUnet_AFM_predictions.py', 'Segmenting cells...'),
    'PRED_UNET': ('5_unet_AFM_predict.py', 'Segmenting cells...'),
    'PRED_PIXEL': ('6_pixel_predict.py', 'Segmenting cells...'),
    'EVAL_MODEL': ('7_eval_model.py', 'Getting general metrics...'),
    'EVAL_CELL': ('8_eval_model_per_cell.py', 'Getting specific metrics...')
}

# Options list
options = {
    '0': ['CROP_IMG', 'PREP_UNET', 'PREP_PIXEL', 'PRED_VUNET', 'PRED_UNET', 'PRED_PIXEL', 'EVAL_MODEL', 'EVAL_CELL'],
    '1': ['CROP_IMG', 'PREP_UNET', 'PRED_VUNET', 'EVAL_MODEL', 'EVAL_CELL'],
    '2': ['CROP_IMG', 'PREP_UNET', 'PRED_UNET', 'EVAL_MODEL', 'EVAL_CELL'],
    '3': ['CROP_IMG', 'PREP_PIXEL', 'PRED_PIXEL', 'EVAL_MODEL', 'EVAL_CELL']
}



# Solicits the user to select an option
while True:
    print(
        '''
        Select a number according to which model you want to evaluate:

        0 - All Models
        1 - vUnet_AFM
        2 - Unet_AFM
        3 - Pixel_AFM
        '''
    )
    option = input('Enter the number of the desired option:\n')

    if option.isdigit() and option in options.keys():
        option = option
        break
    else:
        print("Enter a valid option.\n")

# Execution of the scripts
print('START PROJECT')
for script_key in tqdm(options[option], colour='green'):
    run_script(scripts_dict[script_key][0], scripts_dict[script_key][1], option)
print('FINISH PROJECT')
