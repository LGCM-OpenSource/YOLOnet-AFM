import platform
import subprocess
import os
import sys
from setuptools import setup, find_packages

def create_virtual_environment():
    # Identificar o sistema operacional
    system = platform.system()

    # Criar o ambiente virtual com a versão do Python 3.11.8
    if system == 'Windows':
        # No Windows, usamos venv para criar o ambiente virtual
        subprocess.run([sys.executable, '-m', 'venv', 'segmentenv'])
    else:
        # Em sistemas baseados em Unix, usamos virtualenv
        subprocess.run(['python', '-m', 'venv', 'segmentenv'])

def install_dependencies():
    # Ativar o ambiente virtual
    system = platform.system()
    if system == 'Windows':
        activate_cmd = f'segmentenv{os.sep}Scripts{os.sep}activate.bat'
    else:
        activate_cmd = f'source segmentenv{os.sep}bin{os.sep}activate'
    subprocess.run([activate_cmd], shell=True)

    # Determinar o arquivo de requisitos adequado com base no sistema operacional
    if system == 'Windows':
        requirements_file = 'requirements_win.txt'
    elif system == 'Linux':
        requirements_file = 'requirements_linux.txt'
    else:
        raise RuntimeError("Operacional System not supported!")

    # Instalar as dependências do arquivo de requisitos no ambiente virtual
    subprocess.run(['pip', 'install', '--upgrade', '--force-reinstall', '-r', requirements_file])

def main():
    create_virtual_environment()
    install_dependencies()

    # Criar o pacote
    setup(
        name='vUnet_AFM',
        version='1.0.0',
        packages=find_packages(),
    )

if __name__ == "__main__":
    main()
