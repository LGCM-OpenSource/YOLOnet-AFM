from utils import UserInput
import sys
import platform
import subprocess
import os 
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

    if arg is not None:
        arg_string = f' -ms {arg}'
        script_path = script_path + arg_string
        
    print(f'START {file}... {msg}\nPlease wait')
    run_by_operating_system(script_path)
    print(f'FINISH {file}... {msg}\n')


def have_arguments(process_script):
    if len(process_script) < 3:
        return False
    return True

# Current directory of the script
current_directory = os.path.dirname(os.path.abspath(__file__))

process_flow_scripts_dict = {
    'CROP_IMG': ('1_cropping_opt_images.py', 'cropping images...'),
    'PREP_UNET': ('2_preprocess_unet.py',  '-ms' , 'Generating optical AFM image...'),
    'PRED_UNET': ('4_predicts.py', '-ms', 'Segmenting cells...'),
    'EVAL_MODEL': ('5_eval_models.py', '-ms',  'Getting general metrics...'),
}



model_selector = UserInput.select_model()

# Execution of the scripts
for script_key in tqdm(process_flow_scripts_dict, colour='green'):
    arg = None
    if have_arguments(process_flow_scripts_dict[script_key]):
        arg =  model_selector
        run_script(process_flow_scripts_dict[script_key][0], process_flow_scripts_dict[script_key][2], arg)
print('FINISH PROJECT')