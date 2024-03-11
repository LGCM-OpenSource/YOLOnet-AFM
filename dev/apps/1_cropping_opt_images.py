import os
import sys
sys.path.append(f'dev{os.sep}scripts')
from image_treatment import CropImages
from tqdm import tqdm


thisDir = os.path.dirname(os.path.realpath('__file__'))+os.sep

'''get images (bw and optical) dirs'''
dirOptImg = thisDir+"data"+os.sep+"raw"+os.sep+"optical_images"+os.sep
dirBwImg = thisDir+"data"+os.sep+"raw"+os.sep+"bw_images"+os.sep
destDir = thisDir+"data"+os.sep+"input"+os.sep+"optical_images_resized"+os.sep
txt_file = thisDir+"data"+os.sep+"raw"+os.sep+"txt_files"+os.sep

#listing dirs
dir = os.listdir(dirOptImg)

for img in tqdm(dir):
    #Trat file names
    image_bw_name = img.replace('_OpticalImg.png','_OpticalImg-BW.png')
    image_dest_name = img.replace('_OpticalImg.png', '_optico_crop_resized.png')
    txt_filename = img.replace('_OpticalImg.png', '_2-reference-force-height-extend.txt')
    
    #set up the respective directories 
    image_opt =  dirOptImg+img
    image_bw = dirBwImg+image_bw_name
    txt_dir = txt_file + txt_filename
    dest = destDir+image_dest_name

    imgTrat = CropImages(image_opt, image_bw)

    # croppging images
    imgTrat.run_crop_image(dest, txt_dir)