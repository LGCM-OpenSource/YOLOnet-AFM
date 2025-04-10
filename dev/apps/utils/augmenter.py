import os
import numpy as np
import cv2
from tqdm import tqdm
import traceback # Import traceback for detailed error logging
import logging 
from .data_path import create_dir
from .logger import setup_logger, get_logger


class Augmenter:
    logger = setup_logger('augmenter_process')
    def __init__(self, input_img_dir, input_mask_dir, output_img_dir, output_mask_dir, target_shape=(256, 256)):
        self.input_img_dir = input_img_dir
        self.input_mask_dir = input_mask_dir
        self.output_img_dir = output_img_dir
        self.output_mask_dir = output_mask_dir
        self.target_shape = target_shape
        self.augmentations = [90, 180, 270, 'flip']

        self.logger.info(f"Initializing Augmenter:")
        self.logger.info(f"  Input Image Dir: {self.input_img_dir}")
        self.logger.info(f"  Input Mask Dir: {self.input_mask_dir}")
        self.logger.info(f"  Output Image Dir: {self.output_img_dir}")
        self.logger.info(f"  Output Mask Dir: {self.output_mask_dir}")
        self.logger.info(f"  Target Shape: {self.target_shape}")
        self.logger.info(f"  Augmentations: {self.augmentations}")

        try:
            create_dir(self.output_img_dir)
            self.logger.info(f"Ensured output image directory exists: {self.output_img_dir}")
            create_dir(self.output_mask_dir)
            self.logger.info(f"Ensured output mask directory exists: {self.output_mask_dir}")
        except OSError as e:
            self.logger.error(f"Failed to create output directories: {e}")
            self.logger.error(traceback.format_exc())
            raise # Re-raise the exception if directory creation fails
    
    @staticmethod
    def rotate(image, angle):
        H, W = image.shape[:2]
        center = (W // 2, H // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (W, H))

    @staticmethod
    def flip(image):
        flipped_image = cv2.flip(image, 0) # Flip vertically (axis 0)
        # self.logger.debug("Flipped image vertically.") # Optional: Log flip
        return flipped_image
    
    
    def apply(self):
        self.logger.info("Starting augmentation process...")
        processed_count = 0
        error_count = 0
        skipped_count = 0
        try:
            img_files = sorted(os.listdir(self.input_img_dir))
            self.logger.info(f"Found {len(img_files)} files in input image directory: {self.input_img_dir}")
        except FileNotFoundError:
            self.logger.error(f"Input image directory not found: {self.input_img_dir}")
            return # Stop if input dir doesn't exist
        except Exception as e:
            self.logger.error(f"Error listing files in {self.input_img_dir}: {e}")
            self.logger.error(traceback.format_exc())
            return


        for img_file in tqdm(img_files, desc="Applying augmentations", colour='green'):
            self.logger.debug(f"Processing file: {img_file}")
            try:
                img_path = os.path.join(self.input_img_dir, img_file)
                mask_path = os.path.join(self.input_mask_dir, img_file)

                # Check if corresponding mask exists before loading
                if not os.path.exists(mask_path):
                    self.logger.warning(f"Mask file not found for {img_file}, skipping: {mask_path}")
                    skipped_count += 1
                    continue

                # Load image and mask
                img = np.load(img_path)
                mask = np.load(mask_path)
                self.logger.debug(f"Loaded image {img.shape} and mask {mask.shape} for {img_file}")

                # Resize
                img = cv2.resize(img, self.target_shape, interpolation=cv2.INTER_LINEAR)
                # Use INTER_NEAREST for masks to avoid creating intermediate values
                mask = cv2.resize(mask, self.target_shape, interpolation=cv2.INTER_NEAREST)
                self.logger.debug(f"Resized image and mask to {self.target_shape}")

                mask[mask > 0] = 1
                mask = mask.astype(np.uint8) # Ensure mask is integer type
                self.logger.debug("Binarized mask after resizing")

                self._save_augmented(img, mask, img_file)

                for aug in self.augmentations:
                    if isinstance(aug, int): # Rotation
                        rotated_img = self.rotate(img, aug)
                        rotated_mask = self.rotate(mask, aug)
                        rotated_mask = rotated_mask.astype(np.uint8)
                        self.logger.debug(f"Applied rotation {aug}")
                        self._save_augmented(rotated_img, rotated_mask, img_file, suffix=f'_{aug}')
                    elif aug == 'flip':
                        flipped_img = self.flip(img)
                        flipped_mask = self.flip(mask)
                        # Mask should still be binary after flip, no re-binarization needed here
                        self.logger.debug("Applied flip")
                        self._save_augmented(flipped_img, flipped_mask, img_file, suffix='_flip')

                        # Apply rotations to the flipped version
                        for angle in [a for a in self.augmentations if isinstance(a, int)]:
                            rotated_flipped_img = self.rotate(flipped_img, angle)
                            rotated_flipped_mask = self.rotate(flipped_mask, angle)

                            rotated_flipped_mask = rotated_flipped_mask.astype(np.uint8)
                            self.logger.debug(f"Applied flip + rotation {angle}")
                            self._save_augmented(rotated_flipped_img, rotated_flipped_mask, img_file, suffix=f'_flip_{angle}')
                processed_count += 1
            except FileNotFoundError as e:
                self.logger.error(f"File not found during processing of {img_file}: {e}")
                error_count += 1
            except Exception as e:
                self.logger.error(f"Error processing file {img_file}: {e}")
                self.logger.error(traceback.format_exc()) # Log detailed traceback
                error_count += 1
        self.logger.info(f"Augmentation process finished.")
        self.logger.info(f"Successfully processed files: {processed_count}")
        self.logger.info(f"Files skipped (missing mask): {skipped_count}")
        self.logger.info(f"Files failed during processing: {error_count}")
    def _save_augmented(self, img, mask, base_filename, suffix=''):
        """Saves the augmented image and mask, logging the action."""
        try:
            # Ensure the mask is binary and correct dtype before saving
            mask = mask.astype(np.uint8)
            mask[mask > 0] = 1 # Final check for binary

            base = os.path.splitext(base_filename)[0] # Get filename without extension
            new_img_filename = f"{base}{suffix}.npy"
            new_mask_filename = f"{base}{suffix}.npy"

            img_dest = os.path.join(self.output_img_dir, new_img_filename)
            mask_dest = os.path.join(self.output_mask_dir, new_mask_filename)

            np.save(img_dest, img)
            np.save(mask_dest, mask)

            self.logger.debug(f"Saved augmented image to: {img_dest}")
            self.logger.debug(f"Saved augmented mask to: {mask_dest}")
        except Exception as e:
            self.logger.error(f"Failed to save augmented file {base_filename} with suffix '{suffix}': {e}")
            self.logger.error(traceback.format_exc())
            # Optionally re-raise or handle the saving error further
