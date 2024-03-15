import os
import sys
import platform
import subprocess
from tqdm import tqdm

# Verifica e ajusta o comando de execução conforme o sistema operacional
def run_by_operating_system(abs_path_file):
    identify = platform.system()

    if identify == 'Linux':
        run_file =  subprocess.run([f'{sys.executable} {abs_path_file}'], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
    else:
        run_file = subprocess.run([sys.executable,  f'{abs_path_file}'], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

    return run_file

# Diretório atual do script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Dicionário com os scripts a serem executados e suas mensagens correspondentes
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

# Lista de opções de execução
options = {
    0: ['CROP_IMG', 'PREP_UNET', 'PREP_PIXEL', 'PRED_VUNET', 'PRED_UNET', 'PRED_PIXEL', 'EVAL_MODEL', 'EVAL_CELL'],
    1: ['CROP_IMG', 'PREP_UNET', 'PRED_VUNET', 'EVAL_MODEL', 'EVAL_CELL'],
    2: ['CROP_IMG', 'PREP_UNET', 'PRED_UNET', 'EVAL_MODEL', 'EVAL_CELL'],
    3: ['CROP_IMG', 'PREP_PIXEL', 'PRED_PIXEL', 'EVAL_MODEL', 'EVAL_CELL']
}

# Função para executar um script com ou sem argumento
def run_script(file, msg, arg=None):
    script_path = os.path.join(current_directory, file)

    if arg is not None and file.split('_')[1] == 'eval':
        arg_string = f' -op {arg}'
        script_path = script_path + arg_string
    else:
        script_path = script_path

    print(f'START {file}... {msg}\nPlease wait')
    run_by_operating_system(script_path)
    print(f'FINISH {file}... {msg}')

# Solicita ao usuário que selecione uma opção
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

    if option.isdigit() and int(option) in options.keys():
        option = int(option)
        break
    else:
        print("Enter a valid option.\n")

# Execução dos scripts
print('START PROJECT')
for script_key in tqdm(options[option], colour='green'):
    run_script(scripts_dict[script_key][0], scripts_dict[script_key][1], option)
print('FINISH PROJECT')
