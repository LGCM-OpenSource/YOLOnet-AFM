import os
import numpy as np
import cv2
from tqdm import tqdm
from .data_path import create_dir

class Augmenter:
    def __init__(self, input_img_dir, input_mask_dir, output_img_dir, output_mask_dir, target_shape=(256, 256)):
        self.input_img_dir = input_img_dir
        self.input_mask_dir = input_mask_dir
        self.output_img_dir = output_img_dir
        self.output_mask_dir = output_mask_dir
        self.target_shape = target_shape
        self.augmentations = [90, 180, 270, 'flip']

        create_dir(self.output_img_dir)
        create_dir(self.output_mask_dir)

    
    @staticmethod
    def rotate(image, angle):
        H, W = image.shape[:2]
        center = (W // 2, H // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (W, H))

    @staticmethod
    def flip(image):
        return cv2.flip(image, 0)
    
    
    def apply(self):
        img_files = sorted(os.listdir(self.input_img_dir))

        for img_file in tqdm(img_files, desc="Applying augmentations", colour='green'):
            try:
                img_path = os.path.join(self.input_img_dir, img_file)
                mask_path = os.path.join(self.input_mask_dir, img_file)

                img = np.load(img_path)
                mask = np.load(mask_path)

                img = cv2.resize(img, self.target_shape, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, self.target_shape, interpolation=cv2.INTER_LINEAR)
                mask[mask > 0] = 1

                self._save_augmented(img, mask, img_file)

                for aug in self.augmentations:
                    if aug == 'flip':
                        flipped_img = self.flip(img)
                        flipped_mask = self.flip(mask)
                        self._save_augmented(flipped_img, flipped_mask, img_file, suffix='_flip')

                        for angle in self.augmentations[:-1]:
                            rotated_img = self.rotate(flipped_img, angle)
                            rotated_mask = self.rotate(flipped_mask, angle)
                            self._save_augmented(rotated_img, rotated_mask, img_file, suffix=f'_flip_{angle}')
                    else:
                        rotated_img = self.rotate(img, aug)
                        rotated_mask = self.rotate(mask, aug)
                        self._save_augmented(rotated_img, rotated_mask, img_file, suffix=f'_{aug}')
            except Exception as e:
                self.logger.error(f"Error processing file {img_file}: {str(e)}")

    def _save_augmented(self, img, mask, base_filename, suffix=''):
        new_img_filename = base_filename.replace('.npy', f'{suffix}.npy')
        new_mask_filename = base_filename.replace('.npy', f'{suffix}.npy')

        img_dest = os.path.join(self.output_img_dir, new_img_filename)
        mask_dest = os.path.join(self.output_mask_dir, new_mask_filename)

        np.save(img_dest, img)
        np.save(mask_dest, mask)

        self.logger.info(f'[✓] Saved image to: {img_dest}')
        self.logger.info(f'[✓] Saved mask  to: {mask_dest}')
