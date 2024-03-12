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

scripts_list_dict = {
                    'CROP_IMG':('1_cropping_opt_images.py', 'cropping images...'),
                    'PREP_UNET':('2_preprocess_unet.py', 'Generating optical AFM image...'),
                    'PREP_PIXEL':('3_preprocess_pixel.py', 'Normalizing AFM data file...'),
                    'PRED_VUNET':('4_vUnet_AFM_predictions.py', 'Segmenting cells...'),
                    'PRED_UNET':('5_unet_AFM_predict.py', 'Segmenting cells...'),
                    'PRED_PIXEL':('6_pixel_predict.py', 'Segmenting cells...'),
                    'EVAL_MODEL':('7_eval_model.py', 'Getting general metrics...'),
                    'EVAL_CELL':('8_eval_model_per_cell.py', 'Getting specific metrics...')
                }


while True:
        print(
        '''
        Select a number according to which model you want to evaluate:\n
        0 - All Models
        1 - vUnet_AFM
        2 - Unet_AFM
        3 - Pixel_AFM
        '''
        )
        option  = input('Enter the number of the desired option:\n')
        
        if option.isdigit() and int(option) in [0, 1, 2, 3]:
            option = int(option)
            break
        else: 
            print("Enter a valid option:\n")
            
# all models
if option == 0:
    run_list = [
        scripts_list_dict['CROP_IMG'],
        scripts_list_dict['PREP_UNET'],
        scripts_list_dict['PREP_PIXEL'],
        scripts_list_dict['PRED_VUNET'],
        scripts_list_dict['PRED_UNET'],
        scripts_list_dict['PRED_PIXEL'],
        scripts_list_dict['EVAL_MODEL'],
        scripts_list_dict['EVAL_CELL'],
    ]
# vUnet AFM
elif option == 1:
    run_list = [
        scripts_list_dict['CROP_IMG'],
        scripts_list_dict['PREP_UNET'],
        scripts_list_dict['PRED_VUNET'],
        scripts_list_dict['EVAL_MODEL'],
        scripts_list_dict['EVAL_CELL'],
    ]
# Unet AFM
elif option == 2:
    run_list = [
        scripts_list_dict['CROP_IMG'],
        scripts_list_dict['PREP_UNET'],
        scripts_list_dict['PRED_UNET'],
        scripts_list_dict['EVAL_MODEL'],
        scripts_list_dict['EVAL_CELL'],
    ]
# Pixel
elif option == 3:
    run_list = [
        scripts_list_dict['CROP_IMG'],
        scripts_list_dict['PREP_PIXEL'],
        scripts_list_dict['PRED_PIXEL'],
        scripts_list_dict['EVAL_MODEL'],
        scripts_list_dict['EVAL_CELL'],
    ]

print('START PROJECT')
for script, msg in run_list:
    time.sleep(2)
    print(f'Start {script}... {msg}\nPlease wait')
    time.sleep(2)
    run_by_operacional_system_ajust(f'{self}{os.sep}{script}')
print('FINISH PROJECT')