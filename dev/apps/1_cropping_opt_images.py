import os
from utils import CropImages,build_file_path, CROP_PATH, TerminalStyles
from tqdm import tqdm


term = TerminalStyles()

dir = os.listdir(CROP_PATH['optical_raw'])

for img in tqdm(dir, colour='#0000FF'):
    
    image_opt = os.path.join(CROP_PATH['optical_raw'], img)
    image_bw = build_file_path(CROP_PATH['optical_bw_raw'], img, new_process='_OpticalImg-BW.png')
    txt_dir = build_file_path(CROP_PATH['txt_files'], img, new_process=['_2-reference-force-height-extend.txt', '_2-reference-force-height-extend.jpk-qi-image.txt', '_3-reference-force-height-extend.txt'])
    dest = build_file_path(CROP_PATH['optical_crop_resized'], img, new_process='_optico_crop_resized.png')

    imgTrat = CropImages(image_opt, image_bw)
    imgTrat.run_crop_image(dest, txt_dir)

print(f'Cropped images saved in: {term.SAVE_COLOR}{CROP_PATH["optical_crop_resized"]}{term.RESET}\n')