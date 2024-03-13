# criar a env
# ativar a env
# Instalar as dependencias
# executar o projeto

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
new_self = self.replace(f'{os.sep}dev{os.sep}apps','')

create_env = f'-m venv {new_self}{os.sep}segmentation_env'

run_by_operacional_system_ajust(create_env)

# instal_requirements = ''

# os.system(activate_env)


# criar ambos os requirements

#requirements_win.txt
#requirements_linux.txt
# tensorflow_intel==2.15.0
