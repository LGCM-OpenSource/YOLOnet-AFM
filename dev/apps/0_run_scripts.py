import os
import time
import sys
import platform
import subprocess

def run_by_operacional_system_ajust(abs_path_file):
    identify = platform.system()

    if identify == 'Linux':
        run_file = subprocess.run([f'{sys.executable} {abs_path_file}'], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
        #,  stdin=None, stdout=None, stderr=None, close_fds=True -> retornei essas variáveis para rodar novamentde no shell
        # ,capture_output=True, text=True, -> para salvar a saída
    else:
        run_file = subprocess.run([sys.executable, f'{abs_path_file}'], capture_output=True, text=True, shell=True)
        # , stdin=None, stdout=None, stderr=None, close_fds=True

    return run_file  

self_absolute_path = os.path.abspath(__file__)
self = os.path.dirname(self_absolute_path)

scripts_list = [('1_cropping_opt_images.py', 'cropping images...'),
                ('2_preprocess_unet.py', 'Generating optical AFM image...'),
                ('3_preprocess_pixel.py', 'Normalizing AFM data file...'),
                ('4_vUnet_AFM_predictions.py', 'Segmenting cells...'),
                ('7_eval_model.py', 'Getting general metrics...'),
                ('8_eval_model_per_cell.py', 'Getting specific metrics...')
                ]

print('START PROJECT')
for script, msg in scripts_list:
    time.sleep(2)
    print(f'Start {script}... {msg}\nPlease wait')
    time.sleep(2)
    run_by_operacional_system_ajust(f'{self}{os.sep}{script}')
print('FINISH PROJECT')