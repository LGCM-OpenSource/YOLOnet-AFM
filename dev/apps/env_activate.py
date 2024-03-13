import os
import subprocess
import platform

self_absolute_path = os.path.abspath(__file__)
self = os.path.dirname(self_absolute_path)
new_self = self.replace(f'{os.sep}dev{os.sep}apps','')

identify = platform.system()

caminho_env = 'segmentation_env/bin/activate'

script_ativacao = f'{new_self}{os.sep}segmentation_env/bin/activate'

subprocess.run(script_ativacao, check=True)
