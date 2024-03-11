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

scripts_list = ['1_cropping_opt_images.py',
                '2_preprocess_unet.py',
                '3_preprocess_pixel.py',
                '4_predictions.py',
                '4_unet_predict.py',
                '5_pixel_predict.py',
                '6_eval_model.py'
                ]

print('START PROJECT')
for script in scripts_list:
    time.sleep(2)
    print(f'Start {script} Please wait')
    time.sleep(2)
    run_by_operacional_system_ajust(f'{self}{os.sep}{script}')
print('FINISH PROJECT')